"""AgentCore Platform v1.0"""

# FIN-C2-005 - shared content screens
#
# Two screens are applied at more than one boundary, so their patterns live
# here once rather than being restated (and drifting) per node:
#
#   direct-identifier redaction  - PreProcessNode runs it on the incoming
#       question so raw account/customer identifiers never reach State or the
#       domain nodes; PostProcessNode runs it again on the assembled answer so
#       an identifier that entered from any other source (a knowledge-base
#       passage, a citation line) cannot reach the external surface either.
#
#   instruction-override refusal - PreProcessNode refuses at the ingest
#       boundary and InputValidateNode refuses again inside the domain
#       workflow, so a direct inner-graph invocation gets the same rule.
#       The platform input gate refuses such payloads too, but the template
#       MUST NOT depend on that: where the gate is absent or configured off,
#       an unchecked payload would otherwise reach the answer path and return
#       success. Refusal is defined in terms of BEHAVIOUR (error status, no
#       query carried forward), never any gate's wording.

import re
from typing import List

# Identifier guards. `\b` alone treats a hyphen as a boundary, so a structured
# document identifier such as "ENE-FAC-20260712-001" would have its digit run
# read as a standalone account number and mangled into
# "ENE-FAC-[REDACTED]". These single-character guards require the match to be
# free-standing: not preceded or followed by a letter, digit or hyphen. They
# are general - no advance list of the identifier shapes a deployment might use
# is needed - and they leave a genuinely standalone number fully matchable.
_ID_LEFT = r"(?<![A-Za-z0-9-])"
_ID_RIGHT = r"(?![A-Za-z0-9-])"

# Surface-level identifier patterns. Within the guards the redaction is
# deliberately blunt: a false positive costs a masked token in a regulatory
# answer, a false negative puts a customer identifier on the external surface.
_IDENTIFIER_PATTERNS: List[re.Pattern[str]] = [
    # IBAN: 2 letters + 2 check digits + up to 30 alphanumerics.
    re.compile(rf"{_ID_LEFT}[A-Z]{{2}}\d{{2}}[A-Z0-9]{{10,30}}{_ID_RIGHT}"),
    # Long bank / account numbers: 10-19 digits, optionally grouped.
    re.compile(rf"{_ID_LEFT}\d{{4}}[- ]?\d{{4}}[- ]?\d{{2,11}}{_ID_RIGHT}"),
    # E-mail addresses.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
]

IDENTIFIER_REPLACEMENT = "[REDACTED]"

# Instruction-override phrasings: text addressed to the MODEL rather than a
# question addressed to the knowledge base. Deliberately narrow - a genuine
# question that merely contains these words ("what must a system prompt
# disclosure policy cover?") does not match, because every alternative
# requires the imperative override shape.
_INSTRUCTION_OVERRIDE_RE = re.compile(
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions)"
    r"|disregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts|rule|rules)"
    r"|forget\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts)"
    r"|(?:reveal|show|print|repeat|output|disclose)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system\s+prompt|system\s+message|instructions|initial\s+prompt)"
    r"|you\s+are\s+now\s+(?:a|an)\s"
    r"|act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+|an\s+)?(?:developer|admin|root)\s+mode"
    r"|override\s+(?:your|the)\s+(?:instruction|instructions|rules|safety)",
    re.IGNORECASE,
)


def strip_direct_identifiers(text: str) -> tuple[str, int]:
    """Redact direct-identifier tokens from a free-text string.

    Returns (redacted_text, redaction_count) so a caller can audit that a
    redaction happened without logging the redacted value itself.
    """
    redactions = 0
    for pattern in _IDENTIFIER_PATTERNS:
        text, count = pattern.subn(IDENTIFIER_REPLACEMENT, text)
        redactions += count
    return text, redactions


def is_instruction_override(text: str) -> bool:
    """True when the text carries instruction-override (prompt-injection) content."""
    return bool(text) and bool(_INSTRUCTION_OVERRIDE_RE.search(text))
