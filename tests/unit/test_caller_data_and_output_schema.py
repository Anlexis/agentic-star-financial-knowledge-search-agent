# FIN-C2-005 — Unit Tests: caller-data contract + external output schema
#
# The two hostile boundaries, tested together because they are two ends of the
# same rule: every caller input is validated against explicit bounds before it
# can influence the run (fail CLOSED, never echo), and the external answer
# enforces its documented invariant for every representation.
#
# Sections:
#   1. input_context validation (the ingest boundary contract)
#   2. _finite_in_range (the untrusted-numeric parser)
#   3. Retrieve/Rerank fail-safes (state-seeded scalars, candidate scores)
#   4. Runtime-config plumbing (config/config.yaml -> inner graph, end-to-end)
#   5. Output-boundary layers, layer ORDER, and the both-directions probes
#
# Deterministic — no model call, no network. framework.* / src.* imports only.

import re

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import (
    ADVISORY_DISCLAIMER,
    PostProcessNode,
    _enforce_disclaimer,
    _redact_blocked_fields,
)
from src.nodes.pre_process_node import _validate_input_context
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import _finite_in_range, from_json, to_json

_GROUNDED_QUERY = (
    "what customer verification steps are required before opening a retail "
    "investment account, and how long must the records be retained?"
)


def _inner_state(**fields) -> dict:
    state = {
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(fields)
    return state


def _outer_state(**fields) -> dict:
    state = {
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(fields)
    return state


# -- 1. input_context validation (ingest boundary) ----------------------------


class TestInputContextValidation:
    def test_absent_context_defaults_channel_unknown(self):
        context, error = _validate_input_context(None)
        assert error is None
        assert context == {"channel": "unknown"}

    def test_valid_full_context(self):
        context, error = _validate_input_context({"channel": "web", "category": "aml_kyc", "top_k": 3})
        assert error is None
        assert context == {"channel": "web", "category": "aml_kyc", "top_k": 3}

    def test_unknown_keys_are_ignored(self):
        context, error = _validate_input_context({"unexpected": object()})
        assert error is None
        assert "unexpected" not in context

    def test_non_mapping_context_is_rejected(self):
        context, error = _validate_input_context(["not", "a", "dict"])
        assert error == "input_context must be an object"
        assert context == {}

    @pytest.mark.parametrize(
        "bad_channel",
        ["Web-Portal!", "a" * 33, "", 7, ["web"]],
        ids=["punctuated", "too-long", "empty", "int", "list"],
    )
    def test_invalid_channel_fails_closed(self, bad_channel):
        _, error = _validate_input_context({"channel": bad_channel})
        assert error is not None and "channel" in error

    def test_rejected_channel_value_is_never_echoed(self):
        _, error = _validate_input_context({"channel": "Web-Portal!"})
        assert "Web-Portal!" not in error

    @pytest.mark.parametrize(
        "bad_category",
        ["Sales Conduct", "b" * 33, "", 3.5, {"k": "v"}],
        ids=["spaces-upper", "too-long", "empty", "float", "dict"],
    )
    def test_invalid_category_fails_closed(self, bad_category):
        _, error = _validate_input_context({"category": bad_category})
        assert error is not None and "category" in error

    @pytest.mark.parametrize(
        "bad_top_k",
        ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), 3.5, True, 0, 21, "3"],
        ids=["str-nan", "str-inf", "str-neginf", "raw-nan", "raw-inf", "float", "bool", "zero", "over", "numeric-str"],
    )
    def test_invalid_top_k_fails_closed(self, bad_top_k):
        _, error = _validate_input_context({"top_k": bad_top_k})
        assert error is not None and "top_k" in error

    def test_rejected_top_k_value_is_never_echoed(self):
        _, error = _validate_input_context({"top_k": 12345})
        assert "12345" not in error

    @pytest.mark.parametrize("good_top_k", [1, 20])
    def test_valid_top_k_bounds(self, good_top_k):
        context, error = _validate_input_context({"top_k": good_top_k})
        assert error is None
        assert context["top_k"] == good_top_k


# -- 2. _finite_in_range (the untrusted-numeric parser) -----------------------


class TestFiniteInRange:
    @pytest.mark.parametrize(
        "value",
        [
            "NaN",
            "Infinity",
            "-Infinity",
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            False,
            None,
            [],
            {},
            "abc",
            21.0,
            -1.0,
        ],
        ids=[
            "str-nan",
            "str-inf",
            "str-neginf",
            "raw-nan",
            "raw-inf",
            "raw-neginf",
            "true",
            "false",
            "none",
            "list",
            "dict",
            "text",
            "above",
            "below",
        ],
    )
    def test_rejects_non_finite_and_out_of_range(self, value):
        assert _finite_in_range(value, 0, 20) is None

    @pytest.mark.parametrize("value,expected", [(0, 0.0), (20, 20.0), ("5", 5.0), (2.5, 2.5)])
    def test_accepts_finite_in_range(self, value, expected):
        assert _finite_in_range(value, 0, 20) == expected


# -- 3. Retrieve/Rerank fail-safes --------------------------------------------


class TestRetrieveScalarFailSafe:
    @pytest.mark.parametrize(
        "bad_top_k",
        [float("nan"), float("inf"), "lots", True, -3, 99],
        ids=["nan", "inf", "str", "bool", "negative", "over"],
    )
    def test_malformed_seeded_top_k_falls_back_to_default(self, bad_top_k):
        """A malformed retrieval_top_k in (possibly checkpoint-restored) state
        must fall back to the module default, never crash or run unbounded."""
        result = RetrieveNode()(_inner_state(search_query=_GROUNDED_QUERY, retrieval_top_k=bad_top_k))
        candidates = from_json(result["retrieved_documents"])
        # default top_k=4 -> pool = max(4*3, 10) = 12, capped by matches.
        assert 0 < len(candidates) <= 12


class TestRerankThresholdFailSafe:
    @pytest.mark.parametrize(
        "bad_threshold",
        [float("nan"), float("inf"), "0.1", True, -0.1, 1.5],
        ids=["nan", "inf", "str", "bool", "below", "above"],
    )
    def test_malformed_threshold_never_weakens_the_gate(self, bad_threshold):
        """A malformed seeded threshold falls back to the default (0.25) — a
        NaN would otherwise compare False against every score and drop every
        passage silently (or, inverted, pass every low-relevance one)."""
        low_relevance = [{"id": "kb-x", "title": "t", "category": "c", "source": "s", "score": 0.1, "excerpt": "e"}]
        result = RerankFilterNode()(
            _inner_state(
                retrieved_documents=to_json(low_relevance),
                retrieval_score_threshold=bad_threshold,
            )
        )
        assert from_json(result["ranked_documents"]) == []

    def test_non_finite_candidate_score_never_clears_the_gate(self):
        chunks = [{"id": "kb-x", "title": "t", "category": "c", "source": "s", "score": float("nan"), "excerpt": "e"}]
        result = RerankFilterNode()(_inner_state(retrieved_documents=to_json(chunks)))
        assert from_json(result["ranked_documents"]) == []

    def test_caller_top_k_narrows_but_never_widens(self):
        docs = [
            {"id": f"kb-{i}", "title": "t", "category": "c", "source": "s", "score": 0.9, "excerpt": "e"}
            for i in range(6)
        ]
        narrowed = RerankFilterNode()(
            _inner_state(
                retrieved_documents=to_json(docs),
                retrieval_top_k=4,
                query_filters=to_json({"category": None, "top_k": 2}),
            )
        )
        assert len(from_json(narrowed["ranked_documents"])) == 2

        widened = RerankFilterNode()(
            _inner_state(
                retrieved_documents=to_json(docs),
                retrieval_top_k=4,
                query_filters=to_json({"category": None, "top_k": 6}),
            )
        )
        assert len(from_json(widened["ranked_documents"])) == 4


# -- 4. Runtime-config plumbing -----------------------------------------------


class TestRuntimeConfigPlumbing:
    def test_runtime_config_reads_repo_config(self):
        import src.graph.graph as graph_module

        cfg = graph_module._runtime_config()
        assert cfg.get("max_retry") == 3
        assert cfg.get("retrieval", {}).get("top_k") == 4
        assert cfg.get("retrieval", {}).get("score_threshold") == 0.25

    def test_missing_config_file_degrades_to_empty(self, monkeypatch, tmp_path):
        import src.graph.graph as graph_module

        monkeypatch.setattr(graph_module, "_RUNTIME_CONFIG_PATH", tmp_path / "absent.yaml")
        assert graph_module._runtime_config() == {}

    def test_declared_config_reaches_inner_state_seeding(self):
        from src.graph.graph import FinancialKnowledgeBaseSearchAgent

        set_caller_input_context(None)
        agent = FinancialKnowledgeBaseSearchAgent()
        agent.compile()
        subgraph = agent._nodes["main"].get_subgraph()
        extra = subgraph._extra_initial_state()
        assert extra["retrieval_top_k"] == 4
        assert extra["retrieval_score_threshold"] == 0.25
        assert extra["retrieval_kb_path"] == "config/kb/financial_kb.json"

    def test_caller_top_k_reaches_inner_retrieval_end_to_end(self, monkeypatch, tmp_path):
        """The context bridge, proven through the FULL nested graph: with the
        relevance floor opened, the number of cited sources in the final answer
        equals the caller's top_k override."""
        import src.graph.graph as graph_module
        from framework.schemas.invocation_context import InvocationContext
        from framework.schemas.trust_level import TrustLevel as TL

        open_gate = tmp_path / "config.yaml"
        open_gate.write_text(
            "max_retry: 3\nretrieval:\n  top_k: 5\n  score_threshold: 0.0\n  kb_path: config/kb/financial_kb.json\n"
        )
        monkeypatch.setattr(graph_module, "_RUNTIME_CONFIG_PATH", open_gate)
        agent = graph_module.FinancialKnowledgeBaseSearchAgent(config=graph_module._runtime_config())
        agent.compile()
        ctx = InvocationContext(caller_trust_level=TL.VERIFIED_EXTERNAL)
        for requested, expected in ((1, 1), (2, 2), (None, 5)):
            input_context = {"top_k": requested} if requested else {}
            result = agent.invoke(_GROUNDED_QUERY, ctx=ctx, input_context=input_context)
            assert result["status"] == AgentStatus.SUCCESS.value
            markers = set(re.findall(r"^- \[(\d+)\]", result["output"], re.M))
            assert (
                len(markers) == expected
            ), f"top_k={requested}: expected {expected} cited sources, got {sorted(markers)}"


# -- 5. External output schema enforcement ------------------------------------


class TestBlockedFieldRedaction:
    def test_verbatim_caller_text_is_redacted(self):
        question = "how long must onboarding verification records be retained"
        answer = f"HEADER\n{question}\nFOOTER"
        sanitised, fields = _redact_blocked_fields(answer, {"validated_input": question})
        assert question not in sanitised
        assert "[REDACTED]" in sanitised
        assert fields == ["validated_input"]

    def test_short_incidental_overlap_is_not_redacted(self):
        answer = "Verification records are retained for the statutory period."
        sanitised, fields = _redact_blocked_fields(answer, {"validated_input": "records"})
        assert sanitised == answer
        assert fields == []


class TestDisclaimerInvariant:
    def test_missing_disclaimer_is_appended(self):
        enforced, appended = _enforce_disclaimer("an answer body without the notice")
        assert appended is True
        assert ADVISORY_DISCLAIMER in enforced

    def test_present_disclaimer_is_left_byte_identical(self):
        body = f"an answer body\n\n---\n\n*{ADVISORY_DISCLAIMER}*"
        enforced, appended = _enforce_disclaimer(body)
        assert appended is False
        assert enforced == body


class TestOutputGateLayers:
    def test_embedded_caller_question_is_redacted_on_the_success_path(self):
        question = "how long must onboarding verification records be retained"
        answer = f"# Financial Knowledge Base Search Result\n{question}\n[1] answer body\n"
        result = PostProcessNode()(_outer_state(result=answer, validated_input=question))
        assert question not in result["formatted_output"]
        assert "[REDACTED]" in result["formatted_output"]
        assert result["status"] == AgentStatus.SUCCESS.value

    def test_identifier_in_knowledge_base_content_is_redacted(self):
        answer = "# Financial Knowledge Base Search Result\n[1] escalate to DE89370400440532013000 for review\n"
        result = PostProcessNode()(_outer_state(result=answer))
        assert "DE89370400440532013000" not in result["formatted_output"]
        assert "[REDACTED]" in result["formatted_output"]

    def test_every_representation_carries_the_disclaimer(self):
        """The invariant holds for the grounded answer, the insufficient-coverage
        answer, and the empty-upstream fallback alike."""
        representations = [
            "# Financial Knowledge Base Search Result\n[1] a grounded body\n",
            "# Financial Knowledge Base Search Result\nThe financial knowledge base does not contain "
            "sufficient coverage to answer this question.\n",
            "",
        ]
        for answer in representations:
            result = PostProcessNode()(_outer_state(result=answer))
            assert ADVISORY_DISCLAIMER in result["formatted_output"]

    def test_layer_order_caller_text_removal_precedes_the_pattern_scan(self):
        """A caller question that itself contains an identifier is removed whole.

        If the identifier pattern scan ran first it would rewrite the identifier
        inside the embedding, the exact-substring match would then miss, and the
        rest of the caller's text would stay on the external surface.
        """
        question = "please check account DE89370400440532013000 verification status"
        answer = f"# Financial Knowledge Base Search Result\n{question}\n[1] body\n"
        result = PostProcessNode()(_outer_state(result=answer, validated_input=question))
        out = result["formatted_output"]
        assert "DE89370400440532013000" not in out
        assert "verification status" not in out
        assert "[REDACTED]" in out

    def test_credential_scan_reruns_after_the_rewriting_layers(self):
        """A credential that only becomes contiguous once caller text is removed
        is still caught by the re-scan, and the whole answer is withheld."""
        caller_text = "AAAAAAAAAAAAAAAAAAAAAAAA"  # >10 chars: a blocked-field match
        answer = f"# Answer\n\napi_key={caller_text}super_secret_value_123\n"
        result = PostProcessNode()(_outer_state(result=answer, validated_input=caller_text))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["formatted_output"].startswith("[ANSWER WITHHELD:")


class TestNumericAndIdentifierPassThrough:
    """This template quotes regulatory content; it computes no aggregates, so no
    rounding grid applies. Numbers, section numbers, dates and domain
    identifiers must therefore cross the boundary BYTE-IDENTICAL — a gate that
    rewrote them would falsify a figure the reader is told to verify against
    the primary text.
    """

    @pytest.mark.parametrize(
        "quoted",
        [
            "JPY 1,000",
            "JPY 1,234",
            "the reporting threshold is JPY 2,000,000 per transaction",
            "records are retained for 7 years",
            "Currency: JPY\n\n3. Cash Position",
            "Article 4, paragraph 2",
            "kb-001",
            "SKF-6205",
            "ENE-FAC-20260712-001",
            "INV-2026-0001-A",
            "STU-1234",
            "reference 4012-7788-3300-9",
            "STAR 2026",
            "in 2026",
            "90d",
            "123456",
            "9,999",
        ],
        ids=[
            "on-grid-amount",
            "off-grid-amount",
            "threshold-in-prose",
            "retention-period",
            "section-heading-after-code",
            "article-reference",
            "passage-id",
            "part-number",
            "document-id",
            "invoice-id",
            "student-id",
            "guarded-digit-run-inside-identifier",
            "acronym-year",
            "bare-year",
            "horizon",
            "long-digit-run",
            "grouped-number",
        ],
    )
    def test_quoted_tokens_cross_the_boundary_unchanged(self, quoted):
        answer = f"# Financial Knowledge Base Search Result\n[1] {quoted}\n\n---\n\n*{ADVISORY_DISCLAIMER}*"
        result = PostProcessNode()(_outer_state(result=answer))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == answer

    @pytest.mark.parametrize(
        "leak,marker",
        [
            ("contact compliance.desk@example.com", "[REDACTED]"),
            ("account DE89370400440532013000", "[REDACTED]"),
            ("account 1234-5678-9012", "[REDACTED]"),
        ],
        ids=["email", "iban", "grouped-account"],
    )
    def test_identifier_leak_forms_are_redacted(self, leak, marker):
        answer = f"# Financial Knowledge Base Search Result\n[1] {leak}\n\n---\n\n*{ADVISORY_DISCLAIMER}*"
        result = PostProcessNode()(_outer_state(result=answer))
        assert leak.split()[-1] not in result["formatted_output"]
        assert marker in result["formatted_output"]
