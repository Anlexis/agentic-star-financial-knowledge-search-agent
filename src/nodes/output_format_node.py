"""AgentCore Platform v1.0"""

# FIN-C2-005 - OutputFormatNode
# Domain node 5 (terminal): compose the final formatted answer - the grounded
# answer body, the Sources list, and the standing advisory disclaimer. The
# disclaimer is part of THIS node's domain output contract, not of the outer
# post_process slot (post_process only gates, it does not compose).
#
# Output schema: the external answer contains knowledge-base-derived content
# plus this fixed structure, and ALWAYS carries the advisory disclaimer.
# The answer RENDERS that contract here; the outer output gate independently
# ENFORCES it (see post_process_node.py) - renderer and gate share the
# disclaimer constant so they cannot drift apart.
#
# Regulatory figures quoted from a passage are rendered verbatim on purpose:
# the answer's value is that a reader can check it against the primary text,
# and this template computes no aggregates of its own, so there is nothing to
# round. See docs/02_design.md, "External output schema".
#
# Node contract: execute(self, state) -> dict ONLY - no config parameter.
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + status to the outer merge_output().
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.nodes.post_process_node import ADVISORY_DISCLAIMER
from src.schemas.state import from_json

ANSWER_HEADING = "# Financial Knowledge Base Search Result"


class OutputFormatNode(FunctionNode):
    """Compose the final answer: body + sources + advisory disclaimer.

    Input state keys:
        grounded_answer: answer body with [n] citation markers
        citations:       JSON list [{ref, id, title, source}]

    Output state keys (partial dict):
        formatted_answer: final rendered answer string
        status:           AgentStatus.SUCCESS.value (plain string - never
                          write the bare enum to State)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        grounded_answer = state.get("grounded_answer") or "No answer is available for this request."
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []

        lines: List[str] = []
        lines.append(ANSWER_HEADING)
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source = str(citation.get("source", "")).strip()
                suffix = f" ({source})" if source else ""
                lines.append(f"- [{ref}] {title}{suffix}")
        else:
            lines.append("- none (no knowledge-base passage cleared the relevance threshold)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{ADVISORY_DISCLAIMER}*")

        formatted_answer = "\n".join(lines)

        # Audit: final answer composed (disclaimer attached).
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
