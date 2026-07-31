"""Chapter 6: Guardrail and escalation logic."""

from dataclasses import dataclass


BLOCKED_TERMS = {"password", "token", "secret"}


@dataclass
class GuardrailResult:
    allowed: bool
    escalation_required: bool
    reason: str


def check_guardrails(text: str) -> GuardrailResult:
    lower = text.lower()
    for term in BLOCKED_TERMS:
        if term in lower:
            return GuardrailResult(False, True, f"Blocked sensitive term: {term}")
    if "delete" in lower or "shutdown" in lower:
        return GuardrailResult(True, True, "High-risk action requires human approval")
    return GuardrailResult(True, False, "Allowed")


if __name__ == "__main__":
    print(check_guardrails("Please delete tenant data"))
    print(check_guardrails("Summarize this release note"))
