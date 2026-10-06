"""AgentCore Platform v1.0"""

# FIN-C2-005 - RetrieveNode
# Domain node 2: deterministic keyword retrieval over the bundled financial
# knowledge base (config/kb/financial_kb.json). The current build is fully
# deterministic - no embedding model or vector store; the retrieval contract
# (retrieved_documents JSON) is store-agnostic so a later vector-store upgrade
# only swaps this node's internals.
#
# Node contract: execute(self, state) -> dict ONLY - no config parameter.
# Retrieval tuning (top_k / kb_path) is forwarded by
# KnowledgeBaseSearchGraphNode._parent_config() ->
# DomainWorkflowGraph._extra_initial_state() as the SCALAR state fields
# retrieval_top_k / retrieval_kb_path; this node reads them directly from
# state, falling back to module defaults (mirroring config/config.yaml) when
# unseeded (e.g. a node instantiated directly in a unit test, bypassing the
# inner graph). Serialized state scalars are re-parsed through a
# finite+bounded check - a non-finite value never reaches a comparison.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from framework.schemas.agent_status import AgentStatus
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import _finite_in_range, from_json, to_json

# Defaults mirror the declared `retrieval` block in config/config.yaml.
_DEFAULT_TOP_K = 4
_DEFAULT_KB_PATH = "config/kb/financial_kb.json"

# Repo root: src/nodes/retrieve_node.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal stopword set for query tokenisation (deterministic, no NLP deps).
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "with",
        "under",
        "what",
        "which",
        "when",
        "how",
        "do",
        "does",
        "must",
        "should",
        "before",
        "after",
        "by",
        "at",
        "from",
        "that",
        "this",
        "it",
        "as",
        "was",
        "were",
        "can",
        "may",
        "any",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Per-field match weights: a query token found in the title counts more than
# one found only in the body content.
_TITLE_WEIGHT = 1.0
_TAG_WEIGHT = 0.8
_CONTENT_WEIGHT = 0.5

# Excerpt length carried into retrieved_documents (keeps State small).
_EXCERPT_CHARS = 400


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords and 1-2 char noise removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _load_kb(kb_path: str) -> tuple[List[Dict[str, Any]], List[str]]:
    """Load the bundled knowledge base JSON.

    A missing or malformed file degrades gracefully to an empty corpus with a
    note; the answer path then reports insufficient coverage rather than
    fabricating one. A relative path resolves against the repo root and must
    stay inside it - a misconfigured path cannot be turned into an
    arbitrary-file read.
    """
    notes: List[str] = []
    path = Path(kb_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    try:
        resolved = path.resolve()
        resolved.relative_to(_REPO_ROOT)
    except (OSError, ValueError):
        notes.append("RetrieveNode: knowledge-base path resolves outside the template.")
        return [], notes
    try:
        entries = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        notes.append("RetrieveNode: knowledge base not readable at the configured path.")
        return [], notes
    if not isinstance(entries, list):
        notes.append("RetrieveNode: knowledge base root must be a JSON list.")
        return [], notes
    return [e for e in entries if isinstance(e, dict)], notes


def _score_entry(entry: Dict[str, Any], query_tokens: List[str]) -> float:
    """Per-entry relevance: best field-weight per query token, averaged."""
    if not query_tokens:
        return 0.0
    title_tokens = set(_tokenize(str(entry.get("title", ""))))
    tag_tokens = set(_tokenize(" ".join(str(t) for t in entry.get("tags", []))))
    content_tokens = set(_tokenize(str(entry.get("content", ""))))
    total = 0.0
    for token in query_tokens:
        if token in title_tokens:
            total += _TITLE_WEIGHT
        elif token in tag_tokens:
            total += _TAG_WEIGHT
        elif token in content_tokens:
            total += _CONTENT_WEIGHT
    return round(total / len(query_tokens), 4)


class RetrieveNode(FunctionNode):
    """Score the bundled knowledge base against the query and emit candidates.

    Input state keys:
        search_query:      normalised query (from InputValidateNode)
        query_filters:     JSON dict with optional category filter
        retrieval_top_k:   declared retrieval depth (scalar)
        retrieval_kb_path: declared knowledge-base path (scalar)

    Output state keys (partial dict):
        retrieved_documents: JSON list of scored candidates (score desc)
        intake_notes:        (on knowledge-base anomalies) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        query = state.get("search_query") or state.get("validated_input") or state.get("user_input", "")
        filters = from_json(state.get("query_filters"), {}) or {}

        # Strict typing on seeded scalars: the config path forwards only
        # validated numerics, so a string here is corruption - fall back.
        raw_top_k = state.get("retrieval_top_k", _DEFAULT_TOP_K)
        parsed_top_k = _finite_in_range(raw_top_k, 1, 20) if isinstance(raw_top_k, (int, float)) else None
        top_k = int(parsed_top_k) if parsed_top_k is not None and parsed_top_k == int(parsed_top_k) else _DEFAULT_TOP_K

        raw_kb_path = state.get("retrieval_kb_path")
        kb_path = raw_kb_path if isinstance(raw_kb_path, str) and raw_kb_path else _DEFAULT_KB_PATH
        entries, notes = _load_kb(kb_path)

        category = filters.get("category")
        if category:
            entries = [e for e in entries if str(e.get("category", "")).lower() == str(category).lower()]

        query_tokens = _tokenize(query if isinstance(query, str) else "")

        candidates: List[Dict[str, Any]] = []
        for entry in entries:
            score = _score_entry(entry, query_tokens)
            if score <= 0.0:
                continue
            candidates.append(
                {
                    "id": str(entry.get("id", "")),
                    "title": str(entry.get("title", "")),
                    "category": str(entry.get("category", "")),
                    "source": str(entry.get("source", "")),
                    "score": score,
                    "excerpt": str(entry.get("content", ""))[:_EXCERPT_CHARS],
                }
            )

        # Deterministic ordering: score desc, then id asc for stable ties.
        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        # Keep a candidate pool wider than top_k - RerankFilterNode makes
        # the final cut after the category boost + threshold.
        pool_size = max(top_k * 3, 10)
        candidates = candidates[:pool_size]

        # Audit: retrieval pass completed.
        emit_trace_event(
            "retrieve_complete",
            {
                "candidates": len(candidates),
                "kb_entries": len(entries),
                "query_tokens": len(query_tokens),
                "top_k": top_k,
            },
            state,
        )

        out: Dict[str, Any] = {"retrieved_documents": to_json(candidates)}
        if notes:
            # Append to (never clobber) the notes accumulated upstream.
            prior = from_json(state.get("intake_notes"), []) or []
            out["intake_notes"] = to_json(list(prior) + notes)
        return out
