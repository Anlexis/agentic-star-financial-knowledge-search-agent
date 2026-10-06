# FIN-C2-005 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# The node contract is execute(self, state) -> dict — no config parameter — so
# retrieval tuning is exercised through the SCALAR state fields the inner
# graph seeds (retrieval_top_k / retrieval_kb_path).
#
# Mirrors docs/03_test_spec.md section 2.3 (RET-01..RET-08).
# Deterministic — keyword scoring over the bundled config/kb/financial_kb.json;
# no model call, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_AML_QUERY = (
    "what customer verification steps are required before opening a retail "
    "investment account under anti-money-laundering rules?"
)


def _make_state(query=_AML_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_verification_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the customer-verification query"
        assert docs[0]["id"] == "kb-001"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {"id", "title", "category", "source", "score", "excerpt"}
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveFilters:
    def test_ret_04_category_filter_restricts_pool(self):
        state = _make_state(
            query="risk disclosure duties",
            query_filters=to_json({"category": "disclosure", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "the disclosure category has bundled entries"
        assert {d["category"] for d in docs} == {"disclosure"}

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []


class TestRetrieveSeededScalars:
    """Tuning arrives as state scalars seeded by the inner graph."""

    def test_ret_06_seeded_kb_path_is_used(self):
        state = _make_state(retrieval_kb_path="config/kb/does_not_exist.json")
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_07_kb_path_outside_the_template_is_refused(self):
        state = _make_state(retrieval_kb_path="../../../etc/hosts")
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("outside the template" in n for n in notes)

    def test_seeded_top_k_widens_the_candidate_pool_bound(self):
        # pool = max(top_k * 3, 10); with top_k=1 the pool bound is 10.
        result = RetrieveNode()(_make_state(retrieval_top_k=1))
        assert len(from_json(result["retrieved_documents"])) <= 10


class TestRetrieveNotesAccumulation:
    def test_ret_08_notes_append_never_clobber(self):
        state = _make_state(
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_kb_path="config/kb/bogus.json",
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2
