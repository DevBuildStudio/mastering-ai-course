"""Chapter 9: Incident triage helper."""

from dataclasses import dataclass


@dataclass
class IncidentSignal:
    retrieval_error_rate: float
    tool_timeout_rate: float
    model_refusal_rate: float


def classify_incident(signal: IncidentSignal) -> str:
    if signal.tool_timeout_rate >= 0.2:
        return "Tool/API incident"
    if signal.retrieval_error_rate >= 0.25:
        return "Retrieval incident"
    if signal.model_refusal_rate >= 0.3:
        return "Model/policy incident"
    return "Minor or mixed incident"


def first_response_steps(category: str) -> list[str]:
    base = ["Acknowledge incident", "Estimate blast radius", "Enable safe fallback mode"]
    if category == "Tool/API incident":
        return base + ["Disable unstable tool", "Raise provider escalation"]
    if category == "Retrieval incident":
        return base + ["Switch to strict source mode", "Rebuild affected index"]
    if category == "Model/policy incident":
        return base + ["Tighten output policy", "Route sensitive traffic to HITL"]
    return base + ["Increase monitoring and continue investigation"]


if __name__ == "__main__":
    s = IncidentSignal(retrieval_error_rate=0.12, tool_timeout_rate=0.27, model_refusal_rate=0.08)
    category = classify_incident(s)
    print(category)
    print(first_response_steps(category))
