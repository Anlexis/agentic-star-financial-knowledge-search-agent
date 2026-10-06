"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# Msgpack safety: structured fields (dict / list[dict]) are stored as JSON
# STRINGS, not bare Python containers - a bare dict/list in a checkpointed
# State field breaks checkpoint serialization. Producers serialize with
# to_json() on write; consumers deserialize with from_json() on read.
# Retrieval TUNING knobs (top_k / score_threshold / kb_path) are plain
# scalars, not JSON - see the note on the retrieval_* fields below.
#
# FIN-C2-005 - Financial Knowledge Base Search Agent (Cat 2 RAG).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# Confidentiality note: direct identifiers (account numbers, IBAN, e-mail) in
# the question are redacted by PreProcessNode before any field is written to
# State. Only the normalised search query, knowledge-base passage summaries,
# and the final grounded answer are persisted - never raw customer
# identifiers.
#
# Node contract: FunctionNode.execute() takes ONLY (self, state) - no config
# parameter. Retrieval tuning knobs flow config/config.yaml ->
# KnowledgeBaseSearchGraphNode._parent_config() -> inner graph config ->
# DomainWorkflowGraph._extra_initial_state() seeds the SCALAR state fields
# below -> RetrieveNode / RerankFilterNode read them directly from state,
# falling back to module defaults (mirroring config/config.yaml) when
# unseeded, e.g. a node instantiated directly in a unit test.

import json
import math
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _finite_in_range(value: Any, lo: float, hi: float) -> Optional[float]:
    """Parse an untrusted numeric: FINITE float within [lo, hi], else None.

    Rejects bools, non-numerics, and - critically - non-finite values: float()
    happily parses "NaN"/"Infinity" (and Python's json accepts bare NaN in
    request bodies), and IEEE NaN comparisons are always False, which turns a
    threshold check into a silent no-op. Every number read from an untrusted
    or serialized source must come through here (or an equivalent explicit
    finite check).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not lo <= parsed <= hi:
        return None
    return parsed


class State(AgentState):
    """Flat TypedDict for FIN-C2-005.

    All shared fields (user_input, input_context, status, session_id,
    node_history, error_log, hitl_*, etc.) are inherited from AgentState.

    dict/list fields use JSON-serialized Optional[str]; the retrieval tuning
    knobs are plain scalars. formatted_output is NOT re-declared here - it is
    inherited from AgentState (re-declaring it with a bare type breaks the
    state contract).
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / KnowledgeBaseSearchGraphNode.merge_output
    # ------------------------------------------------------------------

    # Identifier-redacted, validated question produced by PreProcessNode.
    # Raw input is NOT persisted beyond PreProcessNode.
    validated_input: Optional[str]

    # Final knowledge-base search answer, mapped from the inner graph's
    # formatted_answer output via merge_output.
    knowledge_base_answer: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text search query (whitespace-collapsed, length-capped).
    search_query: Optional[str]

    # JSON STRING (to_json) of the validated caller parameters. Deserialised
    # dict shape: {"category": str | None, "top_k": int | None}.
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: Optional[str]

    # Retrieval tuning knobs, seeded from config/config.yaml by
    # DomainWorkflowGraph._extra_initial_state(). Plain SCALARS (already
    # type/range-validated when they were forwarded), so consumers read them
    # without a JSON round-trip.
    retrieval_top_k: int
    retrieval_score_threshold: float
    retrieval_kb_path: str

    # RetrieveNode output
    # JSON STRING (to_json) of scored knowledge-base candidates. Deserialised
    # shape: list[dict], each entry {"id": str, "title": str, "category": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: Optional[str]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: Optional[str]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers.
    grounded_answer: Optional[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "title": str, "source": str}.
    # Consumers (OutputFormatNode) read it back via from_json().
    citations: Optional[str]

    # OutputFormatNode output
    # Final formatted answer (body + sources + advisory disclaimer). Written by
    # OutputFormatNode; surfaced to the outer graph via get_output() ->
    # merge_output().
    formatted_answer: Optional[str]

    # Validation / parse notes accumulated during intake (no caller values).
    # JSON STRING (to_json) of list[str].
    intake_notes: Optional[str]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    error_code: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
