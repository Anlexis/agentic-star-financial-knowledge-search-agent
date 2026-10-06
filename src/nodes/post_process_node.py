"""AgentCore Platform v1.0"""

# FIN-C2-005 - PostProcessNode
# Outer backbone post_process slot: the external-output boundary for the
# knowledge-base search answer.
#
# The external output schema this template states, and this gate enforces for
# EVERY representation it can emit (grounded answer, insufficient-coverage
# answer, empty-upstream fallback):
#
#   an emitted answer contains knowledge-base-derived content plus the fixed
#   template structure, ALWAYS carries the advisory disclaimer, and NEVER
#   carries caller text, direct customer identifiers, or credentials.
#
# Four independent layers, in this order:
#
#   (1) credential scan - API keys, JWTs, Bearer tokens, password assignments
#       anywhere in the assembled answer withhold the output entirely
#       (sanitised stub, status=ERROR);
#   (2) verbatim caller-text redaction - the answer is built from retrieved
#       knowledge-base passages only, so a verbatim embedding of the caller's
#       question (or a derived query form) is a leak, not a feature: any such
#       embedding is replaced with [REDACTED];
#   (3) direct-identifier redaction - account numbers, IBANs and e-mail
#       addresses are redacted wherever they appear, including from
#       knowledge-base content that a deployment may have replaced;
#   (4) disclaimer invariant - the advisory disclaimer is re-asserted, so no
#       representation can reach a reader without it.
#
# LAYER ORDER matters. Both PATTERN scans (1) and (3) are ordered against the
# rewriting layers deliberately: the credential scan runs on the untouched
# text, and the verbatim caller-text redaction (2) - which removes whole
# strings - runs BEFORE the identifier pattern scan (3), because rewriting an
# identifier INSIDE a caller-text embedding would break the exact-substring
# match and leave the rest of that caller text on the surface. After the
# rewriting layers the credential scan is RE-RUN, so a pattern that only
# emerges once text has been removed is still caught.
#
# This template renders no computed monetary aggregates - it quotes regulatory
# passages - so it enforces no numeric rounding grid: a gate that rewrote
# numbers would falsify a quoted threshold, which is the opposite of the
# grounding guarantee. Numbers, section numbers and domain identifiers
# therefore pass through byte-identical, and the boundary tests pin that in
# both directions.
#
# The domain output gate is the module-level function `_security_gate_output`
# called from inside execute() - NOT an instance method on the node class
# (the framework auto-wraps node instance methods on the real invoke path,
# which would raise at class definition).  The agent class
# (FinancialKnowledgeBaseSearchAgent) exposes the same credential scanner as
# the canonical output-gate entry point and delegates to this module (single
# source of truth).
#
# Node contract: execute(self, state) -> dict ONLY - no config parameter.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG
from src.nodes.screens import strip_direct_identifiers

logger = logging.getLogger(__name__)

# Standing advisory disclaimer - carried by EVERY answer this template emits.
# Rendered by OutputFormatNode (which imports this constant) and re-asserted
# here, so renderer and gate can never drift apart.
ADVISORY_DISCLAIMER = (
    "This answer is generated from the bundled financial knowledge base for "
    "informational purposes only and does not constitute financial, legal, "
    "or investment advice. Verify against the primary regulatory text and "
    "consult your compliance officer before acting on it."
)

# Credential patterns that MUST NOT appear in the formatted answer.
_CREDENTIAL_PATTERNS: List[Tuple[str, str]] = [
    (r"(?:sk|pk|ak)-[A-Za-z0-9]{16,}", "api_key_pattern"),
    (r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "jwt_pattern"),
    (r"Bearer\s+[A-Za-z0-9_\-\.]{8,}", "bearer_token"),
    (
        r"(?:password|passwd|secret|api_key|token|access_key|private_key)" r"\s*[:=]\s*\S{8,}",
        "credential_assignment",
    ),
]

# State fields that must NEVER be embedded verbatim in the external answer.
# The answer is assembled from retrieved knowledge-base passages and their
# citations only - caller-derived text (the question, its normalised forms)
# re-appearing verbatim means caller-controlled content reached the external
# surface.
_BLOCKED_FIELDS = frozenset(
    {
        "user_input",
        "validated_input",
        "search_query",
    }
)

_WITHHELD_TEMPLATE = "[ANSWER WITHHELD: output contained a disallowed pattern ({violation}). Retry the question.]"


def _security_gate_output(content: str) -> Optional[str]:
    """Scan output for disallowed credential/secret patterns.

    Returns the first violation name, or None if the output is clean.
    Module-level function (not a node instance method) - the framework
    auto-wraps node instance methods on the real invoke path, so the gate
    must live at module level.
    """
    for pattern, name in _CREDENTIAL_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return name
    return None


def _redact_blocked_fields(result: str, state: AgentState) -> tuple[str, List[str]]:
    """Replace verbatim embeddings of caller-derived state text with [REDACTED].

    Returns (sanitised_result, redacted_field_names). Only substantial values
    (len > 10) are matched so short incidental overlaps are not redacted.
    """
    redacted: List[str] = []
    sanitised = result
    for field in sorted(_BLOCKED_FIELDS):
        value = state.get(field)
        if isinstance(value, str) and len(value) > 10 and value in sanitised:
            sanitised = sanitised.replace(value, "[REDACTED]")
            redacted.append(field)
    return sanitised, redacted


def _enforce_disclaimer(result: str) -> tuple[str, bool]:
    """Guarantee the advisory disclaimer is present on the emitted answer.

    Returns (result, appended). Every representation must carry it; if an
    upstream path produced an answer without it, the gate appends it rather
    than letting the answer reach a reader unqualified.
    """
    if ADVISORY_DISCLAIMER in result:
        return result, False
    return f"{result.rstrip()}\n\n---\n\n*{ADVISORY_DISCLAIMER}*", True


# Reason code -> the sentence the caller reads. A code with no entry falls
# back to the generic one rather than leaking the code itself.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Apply the output gate and expose the final knowledge-base answer.

    Outer backbone post_process slot. Declared VERIFIED_EXTERNAL - the same
    trust level as the ingest boundary, so the answer is only surfaced to a
    caller the entry point vouched for.

    Input state keys:
        result:                 str - rendered answer from the inner OutputFormatNode
        knowledge_base_answer:  str - same rendered answer (fallback source)

    Output state keys (partial dict):
        formatted_output: str
        result:           str
        status:           str
        error_log:        list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A run declined upstream has nothing to format. Render the reason as
        # the caller-facing body and carry the marker onward.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
                "output": message,
                "formatted_output": message,
            }
        answer: str = str(state.get("result") or state.get("knowledge_base_answer") or "")

        # -- Fallback for an empty upstream answer -----------------------------
        if not answer.strip():
            logger.warning("PostProcessNode: result is empty - using fallback message")
            answer = (
                "No answer content was generated for this request. Check error_log "
                "for upstream failures and retry the question."
            )

        # -- Layer 1: credential scan (withhold entirely) ----------------------
        violation = _security_gate_output(answer)
        if violation:
            logger.error("PostProcessNode: credential pattern detected in output - %s", violation)
            emit_trace_event("post_process_credential_violation", {"violation": violation}, state)
            withheld = _WITHHELD_TEMPLATE.format(violation=violation)
            return {
                "formatted_output": withheld,
                "result": withheld,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: credential pattern detected - {violation}"],
            }

        # -- Layer 2: verbatim caller-text redaction ---------------------------
        sanitised_output, redacted_fields = _redact_blocked_fields(answer, state)
        if redacted_fields:
            logger.error(
                "PostProcessNode: caller-derived text embedded verbatim in output - %s",
                ", ".join(redacted_fields),
            )
            emit_trace_event("post_process_blocked_field_redaction", {"fields": redacted_fields}, state)

        # -- Layer 3: direct-identifier redaction ------------------------------
        sanitised_output, identifier_redactions = strip_direct_identifiers(sanitised_output)
        if identifier_redactions:
            logger.warning(
                "PostProcessNode: %d direct identifier(s) redacted from the answer",
                identifier_redactions,
            )
            emit_trace_event(
                "post_process_identifier_redaction",
                {"redaction_count": identifier_redactions},
                state,
            )

        # -- Layer 4: disclaimer invariant -------------------------------------
        sanitised_output, disclaimer_appended = _enforce_disclaimer(sanitised_output)
        if disclaimer_appended:
            logger.warning("PostProcessNode: advisory disclaimer was missing - re-attached")
            emit_trace_event("post_process_disclaimer_reattached", {}, state)

        # -- Re-scan: the rewriting layers must not have produced a new match --
        residual = _security_gate_output(sanitised_output)
        if residual:
            logger.error("PostProcessNode: credential pattern present after redaction - %s", residual)
            emit_trace_event("post_process_credential_violation", {"violation": residual}, state)
            withheld = _WITHHELD_TEMPLATE.format(violation=residual)
            return {
                "formatted_output": withheld,
                "result": withheld,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: credential pattern detected - {residual}"],
            }

        logger.info("PostProcessNode: output gate passed - length=%d", len(sanitised_output))
        emit_trace_event("post_process_complete", {"output_chars": len(sanitised_output)}, state)

        return {
            "formatted_output": sanitised_output,
            "result": sanitised_output,
            "status": AgentStatus.SUCCESS.value,
        }
