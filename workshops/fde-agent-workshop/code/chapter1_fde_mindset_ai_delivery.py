"""Chapter 1: FDE mindset framing utility."""

from dataclasses import dataclass


@dataclass
class FDEBrief:
    use_case: str
    user_impact: str
    launch_metric: str
    rollback_trigger: str


def summarize_brief(brief: FDEBrief) -> str:
    return (
        f"Use case: {brief.use_case}\n"
        f"User impact: {brief.user_impact}\n"
        f"Launch metric: {brief.launch_metric}\n"
        f"Rollback trigger: {brief.rollback_trigger}"
    )


if __name__ == "__main__":
    sample = FDEBrief(
        use_case="Policy Q and A assistant",
        user_impact="Faster support resolution",
        launch_metric="Grounded answer rate >= 85%",
        rollback_trigger="Critical policy error in production",
    )
    print(summarize_brief(sample))
