# FIN-C2-005 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (FinancialKnowledgeBaseSearchAgent / Graph)
# end-to-end via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# Mirrors docs/03_test_spec.md section 3 (INT-05..INT-12).
# Deterministic — no model call, no network. framework.* / src.* imports only.

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

import src.graph.graph
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    FinancialKnowledgeBaseSearchAgent,
    Graph,
    KnowledgeBaseSearchGraphNode,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

_AML_QUERY = (
    "What customer verification steps are required before opening a retail "
    "investment account under anti-money-laundering rules?"
)


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(FinancialKnowledgeBaseSearchAgent, AgentBaseGraph)

    def test_int_05_graph_alias(self):
        assert Graph is FinancialKnowledgeBaseSearchAgent

    def test_state_schema_is_state(self):
        assert FinancialKnowledgeBaseSearchAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = FinancialKnowledgeBaseSearchAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], KnowledgeBaseSearchGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in FinancialKnowledgeBaseSearchAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = KnowledgeBaseSearchGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = KnowledgeBaseSearchGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = KnowledgeBaseSearchGraphNode()
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "t", "source": "s"}])
        delta = node.merge_output(
            {},
            {"formatted_answer": "ANSWER", "citations": citations, "status": AgentStatus.SUCCESS.value},
        )
        # The inner formatted_answer surfaces as BOTH knowledge_base_answer and
        # result (PostProcessNode's output gate reads state["result"]).
        assert delta == {
            "knowledge_base_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "status": AgentStatus.SUCCESS.value,
            # carried across the boundary so the outer graph can report a
            # rejection settled before the inner run; blank on a normal pass
            "error_code": "",
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert KnowledgeBaseSearchGraphNode.error_strategy == "propagate"
        assert KnowledgeBaseSearchGraphNode.propagate_hitl is False

    def test_int_10_parent_config_degrades_to_defaults_without_a_config_file(self, tmp_path, monkeypatch):
        """An unreadable runtime-config file forwards nothing rather than junk;
        each node then falls back to its module default."""
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", tmp_path / "absent.yaml")
        cfg = KnowledgeBaseSearchGraphNode()._parent_config()
        assert cfg["configurable"] == {"retrieval": {}, "llm": {}}

    def test_int_10_parent_config_rejects_malformed_numerics(self, tmp_path, monkeypatch):
        """A non-finite or out-of-range declared value is not forwarded — a NaN
        threshold would compare False against every score and silently drop
        every passage."""
        bad = tmp_path / "config.yaml"
        bad.write_text("retrieval:\n  top_k: 999\n  score_threshold: .nan\n  kb_path: config/kb/financial_kb.json\n")
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", bad)
        forwarded = KnowledgeBaseSearchGraphNode()._parent_config()["configurable"]["retrieval"]
        assert "top_k" not in forwarded
        assert "score_threshold" not in forwarded
        assert forwarded["kb_path"] == "config/kb/financial_kb.json"


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no model call."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_AML_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_11_output_is_the_gated_formatted_answer(self):
        output = _run(_AML_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Financial Knowledge Base Search Result")
        assert "[1]" in output
        assert "does not constitute financial, legal, or investment advice" in output

    def test_int_11_e2e_traverses_the_post_process_gate(self):
        history = _run(_AML_QUERY).get("node_history", [])
        for cls_name in ("PreProcessNode", "KnowledgeBaseSearchGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_no_coverage_query_still_terminates_success(self):
        result = _run("quantum telepathy sandwich recipes")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_int_12_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """Trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        main slot (it sees status=error and skips the inner graph) and routes
        past post_process to finalize — no domain answer is ever produced."""
        result = _run(_AML_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """State serialization helpers: to_json() on write, from_json() on read."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-001", "score": 0.69, "title": "customer verification"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "aml_kyc", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
