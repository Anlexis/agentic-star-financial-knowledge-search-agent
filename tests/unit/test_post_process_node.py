# FIN-C2-005 — Unit Tests: PostProcessNode (outer post_process slot; output gate)
#
# Invocation canon: node(state) via BaseNode.__call__. PostProcessNode is the
# second outer gate slot and requires VERIFIED_EXTERNAL (like PreProcessNode),
# so its behavioural tests build the state at that level; the ANONYMOUS
# rejection lives in test_trust_gate.py.
#
# Gate layering: the node's own module-level _security_gate_output() scan runs
# INSIDE execute() and replaces a violating answer with the withheld stub
# (returned dict — no exception). The framework's own output scan then sees
# only the clean stub. Intentional-credential tests assert the raw secret never
# survives into formatted_output OR result.
#
# The full external-schema behaviour (redaction layers, layer ordering, and the
# both-directions boundary probes) lives in
# test_caller_data_and_output_schema.py.
#
# Mirrors docs/03_test_spec.md section 2.7 (POST-01..POST-08).
# Deterministic — no model call, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import ADVISORY_DISCLAIMER, PostProcessNode

_CLEAN_ANSWER = (
    "# Financial Knowledge Base Search Result\n\n"
    "[1] customer verification steps are required before account opening.\n\n"
    "---\n\n"
    f"*{ADVISORY_DISCLAIMER}*"
)

# JWT-shaped token built at runtime so no credential-shaped literal ever sits
# in the repository (CI credential-scan hygiene).
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through_byte_identical(self):
        result = PostProcessNode()(_make_state(_CLEAN_ANSWER))
        assert result["status"] == AgentStatus.SUCCESS.value
        # State carries the plain string, never the enum.
        assert result["status"].__class__ is str
        assert result["formatted_output"] == _CLEAN_ANSWER

    def test_post_02_empty_result_degrades_to_a_qualified_message(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "No answer content was generated" in result["formatted_output"]
        # Every representation carries the disclaimer — including this one.
        assert ADVISORY_DISCLAIMER in result["formatted_output"]

    def test_post_03_disclaimer_is_reattached_when_missing(self):
        bare = "# Financial Knowledge Base Search Result\n\n[1] a body without the standing notice.\n"
        result = PostProcessNode()(_make_state(bare))
        assert ADVISORY_DISCLAIMER in result["formatted_output"]
        assert result["status"] == AgentStatus.SUCCESS.value


class TestPostProcessCredentialWithhold:
    def _assert_withheld(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("credential pattern detected" in str(e) for e in result["error_log"])
        # The raw secret must not survive into either surfaced field.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert result["formatted_output"].startswith("[ANSWER WITHHELD:")

    def test_post_04_api_key_is_withheld(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"# Answer\n\n<!-- debug api_key={secret} -->\n"))
        self._assert_withheld(result, secret)

    def test_post_05_credential_assignment_is_withheld(self):
        secret = "password=super_secret_value_123"
        result = PostProcessNode()(_make_state(f"# Answer\n\ninternal note: {secret}\n"))
        self._assert_withheld(result, "super_secret_value_123")

    def test_post_06_jwt_is_withheld(self):
        result = PostProcessNode()(_make_state(f"# Answer\n\nsession token {_FAKE_JWT}\n"))
        self._assert_withheld(result, _FAKE_JWT)

    def test_post_07_bearer_token_is_withheld(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(f"# Answer\n\nauthorization: {secret}\n"))
        self._assert_withheld(result, secret)

    def test_post_08_withheld_output_carries_no_answer_content(self):
        secret = "sk-ABCDEF0123456789abcdef"
        body = "the deposit insurance ceiling applies per depositor per institution"
        result = PostProcessNode()(_make_state(f"# Answer\n\n{body}\n\napi_key={secret}\n"))
        assert body not in result["formatted_output"]
        assert body not in result["result"]
