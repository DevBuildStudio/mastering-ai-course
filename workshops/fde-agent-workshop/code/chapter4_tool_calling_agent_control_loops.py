"""Chapter 4: Schema-safe tool call example."""

from dataclasses import dataclass


@dataclass
class ToolRequest:
    tool_name: str
    service: str


def validate_tool_request(req: ToolRequest) -> None:
    if req.tool_name not in {"status_check", "policy_lookup"}:
        raise ValueError("Unsupported tool")
    if not req.service.strip():
        raise ValueError("Service is required")


def status_check(service: str) -> str:
    return f"{service}: healthy"


def run_tool(req: ToolRequest) -> str:
    validate_tool_request(req)
    if req.tool_name == "status_check":
        return status_check(req.service)
    return f"Policy lookup result for {req.service}"


if __name__ == "__main__":
    request = ToolRequest(tool_name="status_check", service="customer-api")
    print(run_tool(request))
