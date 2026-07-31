"""
FDE Agent Workshop - Reference implementation.

This script demonstrates a production-minded agent skeleton with:
- Retrieval over internal knowledge snippets
- Tool calling against a simulated enterprise API
- Structured outputs
- Guardrails and fallback behavior
- Basic observability metrics

Environment variables (optional):
- WORKSHOP_MODEL: model label used in logs (default: "mock-llm")
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = os.getenv("WORKSHOP_MODEL", "mock-llm")


@dataclass
class AgentMetrics:
    request_id: str
    model: str
    started_at_utc: str
    ended_at_utc: str
    latency_ms: int
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    used_tool: bool
    tool_name: Optional[str]
    retrieval_hits: int
    guardrail_triggered: bool


@dataclass
class AgentResponse:
    answer: str
    sources: List[str]
    confidence: float
    used_tool: bool
    tool_result: Optional[Dict[str, Any]]
    metrics: AgentMetrics


KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "id": "kb-001",
        "title": "FDE Role Definition",
        "content": "Forward Deployed Engineers connect product capabilities to real customer environments and constraints.",
    },
    {
        "id": "kb-002",
        "title": "RAG Failure Modes",
        "content": "Common RAG failures include missing context, stale index content, poor chunking, and metadata filtering issues.",
    },
    {
        "id": "kb-003",
        "title": "Production Trade-offs",
        "content": "Production AI design requires balancing latency, cost, reliability, security, and answer quality.",
    },
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def estimate_tokens(text: str) -> int:
    # Rough approximation suitable for workshop demos.
    return max(1, len(text.split()))


def retrieve_context(query: str, top_k: int = 2) -> List[Dict[str, str]]:
    q = normalize(query)
    scored: List[Tuple[int, Dict[str, str]]] = []
    for doc in KNOWLEDGE_BASE:
        score = 0
        for word in q.split():
            if word in normalize(doc["content"]) or word in normalize(doc["title"]):
                score += 1
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k] if score > 0]


def enterprise_status_tool(service_name: str) -> Dict[str, Any]:
    # Simulated enterprise API integration.
    statuses = ["healthy", "degraded", "maintenance"]
    return {
        "service": service_name,
        "status": random.choice(statuses),
        "checked_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def tool_router(user_query: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    q = normalize(user_query)
    if "status" in q or "health" in q:
        service_match = re.search(r"service\s+([a-zA-Z0-9_-]+)", user_query)
        service_name = service_match.group(1) if service_match else "core-api"
        return "enterprise_status_tool", enterprise_status_tool(service_name)
    return None


def guardrail_check(user_query: str) -> bool:
    blocked_patterns = ["password", "secret key", "private token", "credit card"]
    q = normalize(user_query)
    return any(p in q for p in blocked_patterns)


def synthesize_answer(user_query: str, context_docs: List[Dict[str, str]], tool_data: Optional[Dict[str, Any]]) -> Tuple[str, float]:
    if not context_docs and tool_data is None:
        return (
            "I do not have enough grounded context yet. Please provide more details or connect additional data sources.",
            0.35,
        )

    context_summary = " ".join(doc["content"] for doc in context_docs)
    answer_parts = [f"Based on retrieved context: {context_summary}"]

    confidence = 0.55 + (0.1 * len(context_docs))

    if tool_data is not None:
        answer_parts.append(
            f"Live tool check: service '{tool_data['service']}' is currently '{tool_data['status']}'."
        )
        confidence += 0.15

    answer_parts.append(
        "Recommended next step: add eval checks for retrieval quality and monitor latency, failures, and token usage."
    )

    return "\n".join(answer_parts), min(confidence, 0.95)


def run_agent(user_query: str, request_id: str = "req-001") -> AgentResponse:
    started = time.time()
    started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    if guardrail_check(user_query):
        ended = time.time()
        metrics = AgentMetrics(
            request_id=request_id,
            model=MODEL_NAME,
            started_at_utc=started_at,
            ended_at_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            latency_ms=int((ended - started) * 1000),
            estimated_prompt_tokens=estimate_tokens(user_query),
            estimated_completion_tokens=estimate_tokens("Request blocked by guardrail."),
            used_tool=False,
            tool_name=None,
            retrieval_hits=0,
            guardrail_triggered=True,
        )
        return AgentResponse(
            answer="I cannot process sensitive credential data. Please remove secrets and retry.",
            sources=[],
            confidence=0.99,
            used_tool=False,
            tool_result=None,
            metrics=metrics,
        )

    context_docs = retrieve_context(user_query)
    tool_call = tool_router(user_query)

    tool_name: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
    if tool_call is not None:
        tool_name, tool_result = tool_call

    answer, confidence = synthesize_answer(user_query, context_docs, tool_result)

    ended = time.time()
    ended_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    metrics = AgentMetrics(
        request_id=request_id,
        model=MODEL_NAME,
        started_at_utc=started_at,
        ended_at_utc=ended_at,
        latency_ms=int((ended - started) * 1000),
        estimated_prompt_tokens=estimate_tokens(user_query),
        estimated_completion_tokens=estimate_tokens(answer),
        used_tool=tool_result is not None,
        tool_name=tool_name,
        retrieval_hits=len(context_docs),
        guardrail_triggered=False,
    )

    return AgentResponse(
        answer=answer,
        sources=[doc["id"] for doc in context_docs],
        confidence=round(confidence, 2),
        used_tool=tool_result is not None,
        tool_result=tool_result,
        metrics=metrics,
    )


def pretty_print_response(response: AgentResponse) -> None:
    payload = {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "used_tool": response.used_tool,
        "tool_result": response.tool_result,
        "metrics": asdict(response.metrics),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    demo_queries = [
        "How should an FDE reason about RAG failures?",
        "Check status for service billing-api and explain reliability trade-offs.",
        "My private token is ABC123. Can you store it?",
    ]

    for i, query in enumerate(demo_queries, start=1):
        print(f"\n--- Demo Query {i} ---")
        print(f"User: {query}")
        result = run_agent(query, request_id=f"req-{i:03d}")
        pretty_print_response(result)
