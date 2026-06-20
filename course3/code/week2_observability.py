# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Course 3, Week 2: Observability and Debugging
# This notebook covers production-grade observability for LLM applications using OpenTelemetry,
# Arize Phoenix, and structured logging. You will learn to trace Mistral API calls, record custom
# metrics, diagnose failure patterns, and build a real-time dashboard from captured telemetry data.

# %% [markdown]
# ## 1. Setup
# Install all required packages and configure the Mistral client with environment-based credentials.
# OpenTelemetry provides vendor-neutral tracing/metrics; Arize Phoenix offers an LLM-specific UI.

# %%
# pip install mistralai python-dotenv opentelemetry-sdk opentelemetry-exporter-otlp arize-phoenix structlog

import os
import time
import json
import uuid
import random
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=MISTRAL_API_KEY)

# Cost table (USD per 1M tokens): input / output
COST_TABLE = {
    "mistral-large-latest":  (2.00, 6.00),
    "mistral-small-latest":  (0.20, 0.60),
    "codestral-latest":      (0.20, 0.60),
    "open-mistral-nemo":     (0.15, 0.15),
}

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost for a single Mistral API call."""
    in_rate, out_rate = COST_TABLE.get(model, (2.00, 6.00))
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000

print("Setup complete. Mistral client initialised.")

# %% [markdown]
# ## 2. OpenTelemetry for Mistral Calls
# OpenTelemetry spans capture every Mistral call with model name, token counts, latency, and cost.
# A helper `traced_chat()` wraps `client.chat.complete()` and attaches all attributes automatically.
# Spans are exported to the console exporter here; swap in OTLP for production pipelines.

# %%
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# In-memory exporter so we can inspect spans later in this notebook
memory_exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
# Uncomment to also print spans to stdout:
# provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("mistral-app")


def traced_chat(messages: list[dict], model: str = "mistral-large-latest") -> Optional[str]:
    """Call Mistral chat with a root OpenTelemetry span capturing tokens, cost, and latency.

    Args:
        messages: List of role/content dicts.
        model: Mistral model identifier.

    Returns:
        Assistant reply text, or None on failure.
    """
    with tracer.start_as_current_span("mistral.chat") as span:
        span.set_attribute("model", model)
        span.set_attribute("num_messages", len(messages))
        t0 = time.time()
        try:
            response = client.chat.complete(model=model, messages=messages)
            latency_ms = (time.time() - t0) * 1000
            usage = response.usage
            cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)

            span.set_attribute("prompt_tokens", usage.prompt_tokens)
            span.set_attribute("completion_tokens", usage.completion_tokens)
            span.set_attribute("total_tokens", usage.total_tokens)
            span.set_attribute("cost_usd", round(cost, 6))
            span.set_attribute("latency_ms", round(latency_ms, 2))
            span.set_attribute("status", "ok")

            return response.choices[0].message.content
        except Exception as exc:
            span.set_attribute("status", "error")
            span.set_attribute("error", str(exc))
            span.record_exception(exc)
            raise


reply = traced_chat(
    messages=[{"role": "user", "content": "In one sentence, what is observability?"}],
    model="mistral-small-latest",
)
print("Reply:", reply)

spans = memory_exporter.get_finished_spans()
s = spans[-1]
print(f"\nSpan attributes: {dict(s.attributes)}")
assert s.attributes["status"] == "ok", "Expected successful span"
print("OpenTelemetry span recorded successfully.")

# %% [markdown]
# ## 3. Custom AI Metrics
# OpenTelemetry metrics complement traces with aggregated statistics across many requests.
# A histogram captures latency distributions; a counter accumulates total cost; a gauge holds
# the latest quality score. All three are recorded inside a single `record_metrics()` helper.

# %%
from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
    InMemoryMetricReader,
)

metric_reader = InMemoryMetricReader()
meter_provider = MeterProvider(metric_readers=[metric_reader])
otel_metrics.set_meter_provider(meter_provider)

meter = otel_metrics.get_meter("mistral-app")
llm_latency  = meter.create_histogram("llm.latency",     unit="ms",  description="Chat round-trip latency")
llm_cost     = meter.create_counter("llm.cost_usd",      unit="USD", description="Cumulative API spend")
llm_quality  = meter.create_gauge("llm.quality_score",               description="Latest response quality 0-1")


def record_metrics(latency_ms: float, cost: float, quality: float, model: str) -> None:
    """Record per-request latency, cost, and quality into OpenTelemetry meters.

    Args:
        latency_ms: Wall-clock time for the API call in milliseconds.
        cost: Estimated USD cost for the call.
        quality: Heuristic quality score between 0.0 and 1.0.
        model: Model identifier used as a metric label.
    """
    attrs = {"model": model}
    llm_latency.record(latency_ms, attrs)
    llm_cost.add(cost, attrs)
    llm_quality.set(quality, attrs)


def traced_chat_with_metrics(
    messages: list[dict],
    model: str = "mistral-large-latest",
    quality_score: float = 1.0,
) -> Optional[str]:
    """Wrap traced_chat and emit OTel metrics for every successful call.

    Args:
        messages: Prompt messages.
        model: Mistral model identifier.
        quality_score: Caller-supplied quality score (0-1) for the response.

    Returns:
        Assistant reply text.
    """
    t0 = time.time()
    reply = traced_chat(messages, model)
    latency_ms = (time.time() - t0) * 1000
    spans_now = memory_exporter.get_finished_spans()
    last_span = spans_now[-1]
    cost = float(last_span.attributes.get("cost_usd", 0.0))
    record_metrics(latency_ms, cost, quality_score, model)
    return reply


# Run 3 sample calls to populate metrics
for i in range(3):
    answer = traced_chat_with_metrics(
        messages=[{"role": "user", "content": f"Give me tip #{i+1} for debugging LLMs."}],
        model="mistral-small-latest",
        quality_score=random.uniform(0.7, 1.0),
    )
    print(f"Tip {i+1}: {answer[:80]}...")

metrics_data = metric_reader.get_metrics_data()
print(f"\nMetric instruments collected: {len(metrics_data.resource_metrics)}")
print("Custom AI metrics recorded successfully.")

# %% [markdown]
# ## 4. Arize Phoenix Integration
# Arize Phoenix provides an LLM-native observability UI with span waterfall views, prompt/response
# diffs, and automatic retrieval-quality scoring. `MistralInstrumentor` auto-patches the client so
# every call generates structured spans without manual wrapping. A simulated RAG pipeline shows the
# three-level span hierarchy: retrieve → inject_context → generate.

# %%
# NOTE: Phoenix requires `arize-phoenix` installed. If unavailable in your environment,
# the cell falls back to a stub that prints the same structured trace to stdout.

try:
    import phoenix as px
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    _phoenix_available = True
    print("Arize Phoenix available — launching app (visit http://localhost:6006).")
    session = px.launch_app()
    from phoenix.otel import register
    phoenix_provider = register(project_name="mistral-rag-demo", auto_instrument=False)
    rag_tracer = trace.get_tracer("mistral-rag", tracer_provider=phoenix_provider)
except ImportError:
    _phoenix_available = False
    print("Arize Phoenix not installed — using stub tracer (InMemorySpanExporter).")
    rag_tracer = tracer  # fall back to the in-memory tracer defined in section 2


def rag_pipeline_traced(query: str, documents: list[str]) -> str:
    """Execute a three-stage RAG pipeline with nested OpenTelemetry spans.

    Stages:
        1. retrieve   – score and select relevant documents
        2. inject     – format context string
        3. generate   – call Mistral with the enriched prompt

    Args:
        query: User question.
        documents: Candidate document strings to retrieve from.

    Returns:
        Generated answer string.
    """
    with rag_tracer.start_as_current_span("rag.pipeline") as root:
        root.set_attribute("query", query)
        root.set_attribute("num_candidates", len(documents))

        # Stage 1: retrieve
        with rag_tracer.start_as_current_span("rag.retrieve") as retrieve_span:
            t0 = time.time()
            # Simulate BM25 retrieval by keyword overlap score
            scored = sorted(
                documents,
                key=lambda d: sum(w in d.lower() for w in query.lower().split()),
                reverse=True,
            )
            top_k = scored[:2]
            retrieve_span.set_attribute("latency_ms", round((time.time() - t0) * 1000, 2))
            retrieve_span.set_attribute("docs_returned", len(top_k))
            retrieve_span.set_attribute("top_doc_preview", top_k[0][:60] if top_k else "")

        # Stage 2: inject context
        with rag_tracer.start_as_current_span("rag.inject_context") as inject_span:
            context = "\n\n".join(f"[Doc {i+1}] {d}" for i, d in enumerate(top_k))
            augmented_prompt = (
                f"Answer using ONLY the provided context.\n\nContext:\n{context}\n\nQuestion: {query}"
            )
            inject_span.set_attribute("context_chars", len(context))
            inject_span.set_attribute("total_prompt_chars", len(augmented_prompt))

        # Stage 3: generate
        with rag_tracer.start_as_current_span("rag.generate") as gen_span:
            t0 = time.time()
            answer = traced_chat(
                messages=[{"role": "user", "content": augmented_prompt}],
                model="mistral-small-latest",
            )
            gen_span.set_attribute("latency_ms", round((time.time() - t0) * 1000, 2))
            gen_span.set_attribute("answer_chars", len(answer) if answer else 0)

        root.set_attribute("status", "ok")
        return answer or ""


SAMPLE_DOCS = [
    "OpenTelemetry is a vendor-neutral observability framework for traces, metrics, and logs.",
    "Arize Phoenix provides LLM-specific tracing with prompt/response diff views.",
    "Mistral models support tool calling, JSON mode, and streaming responses.",
    "Structured logging with structlog emits machine-parseable JSON events.",
    "RAG pipelines combine retrieval from a knowledge base with generative responses.",
]

answer = rag_pipeline_traced("What is OpenTelemetry?", SAMPLE_DOCS)
print("RAG answer:", answer[:120])
if _phoenix_available:
    print(f"View traces at: {session.url}")
else:
    finished = memory_exporter.get_finished_spans()
    rag_spans = [s for s in finished if s.name.startswith("rag.")]
    print(f"\nRAG spans captured: {[s.name for s in rag_spans]}")

# %% [markdown]
# ## 5. Structured Logging
# Structured logging emits machine-parseable JSON events instead of free-form strings, enabling
# log aggregators (Datadog, Splunk, ELK) to index every field without brittle regex parsers.
# `structlog` provides a Pythonic API that automatically serialises keyword arguments to JSON.

# %%
import structlog
import logging

# Configure structlog to output JSON lines
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger("mistral-app")


def logged_chat(
    messages: list[dict],
    model: str = "mistral-large-latest",
    session_id: Optional[str] = None,
    cached: bool = False,
) -> Optional[str]:
    """Call Mistral and emit structured log events for success and failure.

    Args:
        messages: Prompt messages.
        model: Mistral model identifier.
        session_id: Caller-supplied session identifier for log correlation.
        cached: Whether the response was served from a cache layer.

    Returns:
        Assistant reply text, or None on unrecoverable error.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]

    t0 = time.time()
    attempt = 0
    max_retries = 2

    while attempt <= max_retries:
        try:
            response = client.chat.complete(model=model, messages=messages)
            latency_ms = round((time.time() - t0) * 1000, 2)
            usage = response.usage
            cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)

            log.info(
                "llm_call",
                model=model,
                tokens=usage.total_tokens,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=round(cost, 6),
                latency_ms=latency_ms,
                cache_hit=cached,
                session_id=session_id,
                attempt=attempt,
            )
            return response.choices[0].message.content

        except Exception as exc:
            attempt += 1
            log.error(
                "llm_failure",
                error=str(exc),
                error_type=type(exc).__name__,
                retry_attempt=attempt,
                session_id=session_id,
                model=model,
            )
            if attempt > max_retries:
                return None
            time.sleep(0.5 * attempt)  # exponential back-off stub

    return None


reply = logged_chat(
    messages=[{"role": "user", "content": "Why is structured logging better than print()?"}],
    model="mistral-small-latest",
    session_id="demo-001",
)
print("\nReply:", (reply or "")[:120])

# %% [markdown]
# ## 6. Debugging Failure Patterns
# LLM pipelines fail in five repeatable ways: empty retrieval, oversized context, rate limiting,
# malformed JSON output, and irrelevant context. Each failure leaves a distinct fingerprint in the
# span attributes that `diagnose_failure()` decodes into an actionable `FailureDiagnosis`.

# %%
@dataclass
class SpanData:
    """Lightweight mock of an OTel span for offline failure injection testing."""
    name: str
    attributes: dict = field(default_factory=dict)
    status: str = "ok"


@dataclass
class FailureDiagnosis:
    """Diagnosis result produced by the failure analyser.

    Attributes:
        failure_type: Short machine-readable label.
        root_cause: Human-readable explanation.
        suggested_fix: Recommended remediation step.
        fingerprint: Deduplication hash string.
    """
    failure_type: str
    root_cause: str
    suggested_fix: str
    fingerprint: str


FAILURE_PATTERNS = {
    "empty_retrieval": {
        "check": lambda s: s.attributes.get("docs_returned", 1) == 0,
        "root_cause": "Retrieval stage returned zero documents — query had no keyword overlap.",
        "fix": "Lower similarity threshold or expand query with synonyms.",
    },
    "context_too_long": {
        "check": lambda s: s.attributes.get("context_chars", 0) > 12_000,
        "root_cause": "Context string exceeds ~12 k chars; prompt likely exceeded model context window.",
        "fix": "Truncate or summarise retrieved documents before injection.",
    },
    "rate_limit_hit": {
        "check": lambda s: "RateLimitError" in s.attributes.get("error", ""),
        "root_cause": "Mistral API rate limit exceeded — too many requests per minute.",
        "fix": "Implement exponential back-off and request queuing.",
    },
    "malformed_json_output": {
        "check": lambda s: s.attributes.get("json_parse_error", False),
        "root_cause": "Model returned text that failed JSON.loads() despite json_object mode.",
        "fix": "Add a retry with stricter system prompt enforcing valid JSON.",
    },
    "irrelevant_context": {
        "check": lambda s: s.attributes.get("relevance_score", 1.0) < 0.2,
        "root_cause": "Retrieved documents have very low relevance to the query.",
        "fix": "Switch to dense (embedding) retrieval instead of sparse BM25.",
    },
}


def failure_fingerprint(span: SpanData) -> str:
    """Produce a stable deduplication key from span name and key attributes.

    Args:
        span: The span to fingerprint.

    Returns:
        A colon-separated string suitable for grouping identical failures.
    """
    model    = span.attributes.get("model", "unknown")
    err      = span.attributes.get("error", "none")[:30]
    docs     = span.attributes.get("docs_returned", "-")
    ctx_size = "large" if span.attributes.get("context_chars", 0) > 12_000 else "ok"
    return f"{span.name}:{model}:{err}:{docs}:{ctx_size}"


def diagnose_failure(span: SpanData) -> FailureDiagnosis:
    """Match a span against known failure patterns and return a diagnosis.

    Args:
        span: Completed span with attributes from the pipeline stage.

    Returns:
        FailureDiagnosis with root cause and fix recommendation.
    """
    for ftype, pattern in FAILURE_PATTERNS.items():
        if pattern["check"](span):
            return FailureDiagnosis(
                failure_type=ftype,
                root_cause=pattern["root_cause"],
                suggested_fix=pattern["fix"],
                fingerprint=failure_fingerprint(span),
            )
    return FailureDiagnosis(
        failure_type="unknown",
        root_cause="No known failure pattern matched.",
        suggested_fix="Inspect full span attributes manually.",
        fingerprint=failure_fingerprint(span),
    )


# Inject and diagnose each failure type
INJECTED_FAILURES = [
    SpanData("rag.retrieve",  {"docs_returned": 0, "model": "mistral-small-latest"}),
    SpanData("rag.inject",    {"context_chars": 15_000, "model": "mistral-small-latest"}),
    SpanData("mistral.chat",  {"error": "RateLimitError: quota exceeded", "model": "mistral-large-latest"}),
    SpanData("mistral.chat",  {"json_parse_error": True, "model": "mistral-large-latest"}),
    SpanData("rag.retrieve",  {"relevance_score": 0.05, "docs_returned": 2, "model": "mistral-small-latest"}),
]

print("=== Failure Diagnosis Report ===\n")
for span in INJECTED_FAILURES:
    dx = diagnose_failure(span)
    print(f"Span       : {span.name}")
    print(f"Type       : {dx.failure_type}")
    print(f"Root cause : {dx.root_cause}")
    print(f"Fix        : {dx.suggested_fix}")
    print(f"Fingerprint: {dx.fingerprint}\n")

# %% [markdown]
# ## 7. Lab Exercise: End-to-End Instrumented RAG with Dashboard
# Instrument a complete Mistral RAG pipeline, run 20 queries (including 3 injected failures),
# capture all telemetry, and render a text-based dashboard showing p50/p95 latency, error rate,
# and average cost per request. All data is also exported as a JSON metrics file.

# %%
import math

# --- Knowledge base for the lab ---
LAB_DOCS = [
    "OpenTelemetry (OTel) provides APIs, SDKs, and tools to instrument applications for observability.",
    "Traces represent the end-to-end journey of a request through a distributed system.",
    "Spans are the basic units of work within a trace, capturing start time, duration, and attributes.",
    "Metrics aggregate numerical measurements over time, such as request counts and latencies.",
    "Structured logs pair each event with key-value metadata enabling programmatic querying.",
    "Arize Phoenix offers an LLM-native UI for inspecting prompt/response pairs and retrieval quality.",
    "Mistral-large-latest is the flagship model for complex reasoning and instruction following.",
    "RAG stands for Retrieval-Augmented Generation and combines search with LLM generation.",
    "Token costs accrue per API call; monitoring spend prevents budget overruns.",
    "Exponential back-off with jitter is the standard retry strategy for rate-limited APIs.",
]

LAB_QUERIES = [
    "What is OpenTelemetry?",
    "How do spans relate to traces?",
    "What are structured logs?",
    "Why use Arize Phoenix for LLMs?",
    "What does RAG stand for?",
    "How are token costs calculated?",
    "What is a good retry strategy?",
    "Describe metrics in observability.",
    "What is mistral-large-latest used for?",
    "How does RAG improve LLM accuracy?",
    "What attributes should a span capture?",
    "When should you use mistral-small-latest?",
    "How do you avoid rate limits?",
    "What is the difference between traces and logs?",
    "What is prompt injection?",
    "How do you monitor LLM cost over time?",
    "What is a MeterProvider in OTel?",
    "Explain context window limits.",
    "What is a histogram metric?",
    "How do you debug a failing RAG pipeline?",
]

# 3 queries that will trigger injected failures (indices 5, 11, 17)
FAILURE_INJECTION_INDICES = {5, 11, 17}

@dataclass
class QueryRecord:
    """Telemetry record for a single lab query execution."""
    query_id: int
    query: str
    model: str
    latency_ms: float
    cost_usd: float
    tokens: int
    is_error: bool
    failure_type: str
    answer_preview: str


def run_lab_query(idx: int, query: str) -> QueryRecord:
    """Execute one RAG query, injecting a failure at predetermined indices.

    Args:
        idx: Zero-based query index.
        query: Natural language question.

    Returns:
        QueryRecord with all captured telemetry.
    """
    model = "mistral-small-latest"
    t0 = time.time()

    # Simulate an injected failure
    if idx in FAILURE_INJECTION_INDICES:
        failure_names = ["empty_retrieval", "rate_limit_hit", "malformed_json_output"]
        ftype = failure_names[sorted(FAILURE_INJECTION_INDICES).index(idx)]
        latency_ms = round(random.uniform(50, 150), 2)
        return QueryRecord(
            query_id=idx,
            query=query,
            model=model,
            latency_ms=latency_ms,
            cost_usd=0.0,
            tokens=0,
            is_error=True,
            failure_type=ftype,
            answer_preview=f"[SIMULATED FAILURE: {ftype}]",
        )

    # Normal execution via RAG pipeline
    try:
        scored = sorted(
            LAB_DOCS,
            key=lambda d: sum(w in d.lower() for w in query.lower().split()),
            reverse=True,
        )
        top_k = scored[:2]
        context = "\n".join(f"[{i+1}] {d}" for i, d in enumerate(top_k))
        prompt = f"Answer briefly using only the context.\n\nContext:\n{context}\n\nQ: {query}"

        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = round((time.time() - t0) * 1000, 2)
        usage = response.usage
        cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        answer = response.choices[0].message.content or ""

        return QueryRecord(
            query_id=idx,
            query=query,
            model=model,
            latency_ms=latency_ms,
            cost_usd=round(cost, 6),
            tokens=usage.total_tokens,
            is_error=False,
            failure_type="none",
            answer_preview=answer[:80],
        )

    except Exception as exc:
        latency_ms = round((time.time() - t0) * 1000, 2)
        return QueryRecord(
            query_id=idx,
            query=query,
            model=model,
            latency_ms=latency_ms,
            cost_usd=0.0,
            tokens=0,
            is_error=True,
            failure_type=type(exc).__name__,
            answer_preview=f"[ERROR: {str(exc)[:60]}]",
        )


# --- Run all 20 queries ---
print("Running 20 lab queries (3 injected failures at indices 5, 11, 17)...\n")
records: list[QueryRecord] = []
for i, q in enumerate(LAB_QUERIES):
    rec = run_lab_query(i, q)
    status = "FAIL" if rec.is_error else "OK  "
    print(f"  [{status}] Q{i:02d}: {rec.query[:45]:<45} {rec.latency_ms:>7.1f}ms  ${rec.cost_usd:.5f}")
    records.append(rec)

# --- Compute dashboard metrics ---
ok_records    = [r for r in records if not r.is_error]
err_records   = [r for r in records if r.is_error]
latencies     = [r.latency_ms for r in ok_records]
costs         = [r.cost_usd   for r in ok_records]

latencies_sorted = sorted(latencies)

def percentile(data: list[float], pct: float) -> float:
    """Compute the p-th percentile of a sorted list using linear interpolation."""
    if not data:
        return 0.0
    idx = (len(data) - 1) * pct / 100
    lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)

p50  = percentile(latencies_sorted, 50)
p95  = percentile(latencies_sorted, 95)
err_rate  = len(err_records) / len(records) * 100
avg_cost  = statistics.mean(costs) if costs else 0.0
total_cost = sum(costs)

failure_summary = {}
for r in err_records:
    failure_summary[r.failure_type] = failure_summary.get(r.failure_type, 0) + 1

# --- Text dashboard ---
DASH = "=" * 58
print(f"\n{DASH}")
print("  OBSERVABILITY DASHBOARD — RAG Lab (20 queries)")
print(DASH)
print(f"  Total queries    : {len(records)}")
print(f"  Successful        : {len(ok_records)}")
print(f"  Failed            : {len(err_records)}")
print(f"  Error rate        : {err_rate:.1f}%")
print(f"  p50 latency       : {p50:.1f} ms")
print(f"  p95 latency       : {p95:.1f} ms")
print(f"  Avg cost/request  : ${avg_cost:.5f}")
print(f"  Total cost        : ${total_cost:.5f}")
print(f"\n  Failure breakdown :")
for ftype, count in failure_summary.items():
    print(f"    {ftype:<30} {count}")
print(DASH)

# --- Export metrics as JSON ---
export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".", "lab_metrics.json")
metrics_export = {
    "summary": {
        "total_queries": len(records),
        "success_count": len(ok_records),
        "error_count": len(err_records),
        "error_rate_pct": round(err_rate, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "avg_cost_usd": round(avg_cost, 6),
        "total_cost_usd": round(total_cost, 6),
    },
    "failure_summary": failure_summary,
    "records": [asdict(r) for r in records],
}

with open(export_path, "w", encoding="utf-8") as fh:
    json.dump(metrics_export, fh, indent=2)

print(f"\nMetrics exported to: {export_path}")

# Assertions to verify correctness
assert len(records) == 20, "Expected 20 query records"
assert len(err_records) == 3, "Expected exactly 3 injected failures"
assert p50 > 0, "p50 latency must be positive"
print("\nAll lab assertions passed.")

# %% [markdown]
# ## Key Takeaways
# - OpenTelemetry spans attach model name, token counts, latency, and cost to every Mistral call,
#   creating a full audit trail without modifying business logic.
# - Custom OTel metrics (histogram, counter, gauge) enable statistical analysis — p50/p95 latency
#   and cumulative cost — across thousands of requests in production.
# - Arize Phoenix provides LLM-native observability by auto-instrumenting the Mistral client and
#   visualising the retrieve → inject → generate span hierarchy in a dedicated UI.
# - Structured logging with `structlog` emits JSON-line events that log aggregators can index
#   without regex parsing, enabling reliable alerting on error rates and cost anomalies.
# - The five canonical failure patterns (empty retrieval, oversized context, rate limits, malformed
#   JSON, irrelevant context) each leave a distinct fingerprint in span attributes, making root-cause
#   diagnosis automatable without manual log inspection.
