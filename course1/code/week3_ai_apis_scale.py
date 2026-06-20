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
# # Week 3: Working with AI APIs at Scale
# This notebook covers production patterns for building robust, scalable AI applications.
# We explore conversation management, async concurrency, retry logic, cost tracking, and
# serving AI features via FastAPI with streaming support.

# %% [markdown]
# ## 1. Setup
# Install dependencies with: `pip install mistralai python-dotenv fastapi uvicorn httpx tenacity`
# We import everything needed across all sections up front.

# %%
import os
import time
import json
import asyncio
import logging
from typing import Any

from dotenv import load_dotenv
from mistralai import Mistral, AsyncMistral
from mistralai.models import SDKError
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

print("Setup complete. Mistral SDK and dependencies loaded.")
print(f"API key configured: {'yes' if MISTRAL_API_KEY != 'your-key-here' else 'NO - set MISTRAL_API_KEY'}")

# %% [markdown]
# ## 2. Conversation Manager
# Manages multi-turn chat history with rolling-window truncation and summary compression.
# This prevents context windows from growing unbounded in long sessions.

# %%
class ConversationManager:
    """Manages chat history for a single conversation session.

    Supports rolling-window truncation and LLM-based summary compression
    to keep token usage bounded across long conversations.
    """

    def __init__(self, session_id: str, api_key: str = MISTRAL_API_KEY):
        """Initialize a conversation session.

        Args:
            session_id: Unique identifier for this conversation.
            api_key: Mistral API key.
        """
        self.session_id = session_id
        self._messages: list[dict[str, str]] = []
        self._client = Mistral(api_key=api_key)

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history.

        Args:
            role: 'user' or 'assistant'.
            content: Text content of the message.
        """
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: {role}")
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        """Return a copy of the full message history."""
        return list(self._messages)

    def rolling_window(self, max_turns: int = 10) -> list[dict[str, str]]:
        """Return the most recent max_turns exchange pairs (user+assistant).

        Args:
            max_turns: Maximum number of back-and-forth turns to retain.

        Returns:
            Trimmed message list with at most max_turns*2 messages.
        """
        max_messages = max_turns * 2
        return self._messages[-max_messages:] if len(self._messages) > max_messages else list(self._messages)

    def summary_compress(self, keep_last_turns: int = 4) -> None:
        """Compress older messages into a summary using Mistral.

        Summarizes all but the most recent keep_last_turns exchange pairs,
        replacing them with a single system message containing the summary.

        Args:
            keep_last_turns: Number of recent turns to keep verbatim.
        """
        keep_messages = keep_last_turns * 2
        if len(self._messages) <= keep_messages:
            return

        to_summarize = self._messages[:-keep_messages]
        recent = self._messages[-keep_messages:]

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in to_summarize
        )
        prompt = (
            f"Summarize the following conversation history concisely "
            f"in 3-5 sentences, preserving key facts:\n\n{history_text}"
        )
        try:
            response = self._client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.choices[0].message.content
            self._messages = [{"role": "system", "content": f"[Summary of earlier conversation]: {summary}"}] + recent
            logger.info("Compressed %d messages into summary for session %s", len(to_summarize), self.session_id)
        except SDKError as e:
            logger.error("Summary compression failed: %s", e)

    @property
    def token_count(self) -> int:
        """Rough token estimate: total characters divided by 4."""
        total_chars = sum(len(m["content"]) for m in self._messages)
        return total_chars // 4

    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self._messages.clear()
        logger.info("Cleared conversation session %s", self.session_id)


# Demo
mgr = ConversationManager("demo-session")
mgr.add_message("user", "What is the capital of France?")
mgr.add_message("assistant", "The capital of France is Paris.")
mgr.add_message("user", "What language do they speak?")
mgr.add_message("assistant", "They speak French.")

print(f"Messages in history: {len(mgr.get_messages())}")
print(f"Estimated tokens: {mgr.token_count}")
print(f"Rolling window (last 1 turn): {mgr.rolling_window(max_turns=1)}")
assert len(mgr.rolling_window(max_turns=1)) == 2, "Rolling window should return 2 messages for 1 turn"
print("ConversationManager assertions passed.")

# %% [markdown]
# ## 3. Async and Concurrent Calls
# Using AsyncMistral we can fire multiple API requests concurrently with asyncio.gather().
# A semaphore limits the maximum concurrent connections to respect rate limits.

# %%
async def chat_async(client: AsyncMistral, prompt: str, semaphore: asyncio.Semaphore) -> str:
    """Send a single chat request with semaphore-controlled concurrency.

    Args:
        client: An AsyncMistral client instance.
        prompt: User prompt to send.
        semaphore: Asyncio semaphore limiting concurrent API calls.

    Returns:
        The assistant's reply text.
    """
    async with semaphore:
        response = await client.chat.complete_async(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


async def run_concurrent_demo() -> None:
    """Demonstrate concurrent vs sequential API calls with timing comparison."""
    prompts = [
        "Name one planet in our solar system.",
        "Name one ocean on Earth.",
        "Name one programming language.",
        "Name one famous scientist.",
        "Name one country in Asia.",
    ]

    semaphore = asyncio.Semaphore(10)
    async_client = AsyncMistral(api_key=MISTRAL_API_KEY)

    # Concurrent
    start = time.time()
    results = await asyncio.gather(
        *[chat_async(async_client, p, semaphore) for p in prompts]
    )
    concurrent_time = time.time() - start

    print(f"\nConcurrent ({len(prompts)} calls): {concurrent_time:.2f}s")
    for prompt, result in zip(prompts, results):
        print(f"  Q: {prompt[:40]:<40} A: {result.strip()[:50]}")

    # Sequential estimate
    print(f"\nEstimated sequential time: ~{concurrent_time * len(prompts):.1f}s (if each took the same)")
    print(f"Speedup factor: ~{len(prompts):.1f}x with full concurrency")
    assert len(results) == len(prompts), "Should receive one result per prompt"


# Run the async demo
asyncio.run(run_concurrent_demo())

# %% [markdown]
# ## 4. Retry and Error Handling
# Production APIs occasionally return transient errors. Tenacity's @retry decorator
# implements exponential back-off, and a simple circuit breaker prevents cascading failures.

# %%
class CircuitBreaker:
    """Simple circuit breaker to halt calls after repeated failures.

    States: CLOSED (normal), OPEN (blocking calls), HALF_OPEN (testing recovery).
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening the circuit.
            recovery_timeout: Seconds to wait before attempting recovery.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failures = 0
        self._opened_at: float | None = None
        self.state = "CLOSED"

    def record_success(self) -> None:
        """Reset failure count on a successful call."""
        self._failures = 0
        self.state = "CLOSED"
        self._opened_at = None

    def record_failure(self) -> None:
        """Increment failure count; open circuit if threshold is reached."""
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self.state = "OPEN"
            self._opened_at = time.time()
            logger.warning("Circuit breaker OPENED after %d failures", self._failures)

    def allow_request(self) -> bool:
        """Check whether a request should be allowed through.

        Returns:
            True if the circuit is closed or in half-open probe state.
        """
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if self._opened_at and (time.time() - self._opened_at) > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False
        # HALF_OPEN: allow one probe request
        return True


_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((SDKError, ConnectionError)),
    reraise=True,
)
def resilient_chat(client: Mistral, prompt: str, model: str = "mistral-large-latest") -> str:
    """Send a chat request with exponential back-off retry and circuit breaker.

    Args:
        client: Mistral synchronous client.
        prompt: User prompt.
        model: Model ID to use.

    Returns:
        Assistant reply text.

    Raises:
        SDKError: After all retry attempts are exhausted.
        RuntimeError: When the circuit breaker is open.
    """
    if not _circuit_breaker.allow_request():
        raise RuntimeError("Circuit breaker is OPEN — too many recent failures.")
    try:
        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        _circuit_breaker.record_success()
        return response.choices[0].message.content
    except SDKError as e:
        _circuit_breaker.record_failure()
        logger.error("API error (attempt will retry): %s", e)
        raise


# Demo resilient call
sync_client = Mistral(api_key=MISTRAL_API_KEY)
try:
    start = time.time()
    answer = resilient_chat(sync_client, "In one sentence, what is exponential back-off?")
    elapsed = time.time() - start
    print(f"Resilient call succeeded in {elapsed:.2f}s")
    print(f"Answer: {answer.strip()}")
    print(f"Circuit breaker state: {_circuit_breaker.state}")
except (SDKError, RuntimeError) as e:
    print(f"Call failed after retries: {e}")

# %% [markdown]
# ## 5. Cost Tracker
# Tracking token usage and mapping it to USD cost is essential for production budgeting.
# The CostTracker accumulates usage per session and raises alerts when budgets are exceeded.

# %%
MISTRAL_PRICING: dict[str, dict[str, float]] = {
    "mistral-large-latest":  {"input_per_1m": 3.00,  "output_per_1m": 9.00},
    "mistral-small-latest":  {"input_per_1m": 0.20,  "output_per_1m": 0.60},
    "codestral-latest":      {"input_per_1m": 0.30,  "output_per_1m": 0.90},
    "mistral-embed":         {"input_per_1m": 0.10,  "output_per_1m": 0.00},
    "pixtral-12b-2409":      {"input_per_1m": 0.15,  "output_per_1m": 0.15},
    "open-mistral-nemo":     {"input_per_1m": 0.15,  "output_per_1m": 0.15},
}


class CostTracker:
    """Tracks token usage and computes USD cost across sessions.

    Supports per-session cost breakdown, daily totals, pre-call cost estimation,
    and configurable budget alerts.
    """

    def __init__(self) -> None:
        """Initialize an empty cost tracker."""
        self._usage: list[dict[str, Any]] = []

    def add_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        timestamp: float | None = None,
    ) -> None:
        """Record token usage for one API call.

        Args:
            session_id: Identifier for the conversation session.
            model: Model used for the call.
            input_tokens: Number of prompt tokens consumed.
            output_tokens: Number of completion tokens generated.
            timestamp: Unix timestamp; defaults to now.
        """
        pricing = MISTRAL_PRICING.get(model, {"input_per_1m": 0.0, "output_per_1m": 0.0})
        cost = (input_tokens * pricing["input_per_1m"] + output_tokens * pricing["output_per_1m"]) / 1_000_000
        self._usage.append({
            "session_id": session_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "timestamp": timestamp or time.time(),
        })

    @property
    def total_cost(self) -> float:
        """Total USD cost across all recorded usage."""
        return sum(r["cost_usd"] for r in self._usage)

    def cost_per_session(self) -> dict[str, float]:
        """Return a dict mapping session_id to total USD cost."""
        result: dict[str, float] = {}
        for record in self._usage:
            sid = record["session_id"]
            result[sid] = result.get(sid, 0.0) + record["cost_usd"]
        return result

    def daily_cost(self) -> dict[str, float]:
        """Return a dict mapping ISO date string to total USD cost for that day."""
        from datetime import datetime, timezone
        result: dict[str, float] = {}
        for record in self._usage:
            day = datetime.fromtimestamp(record["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d")
            result[day] = result.get(day, 0.0) + record["cost_usd"]
        return result

    def estimate_cost(self, messages: list[dict[str, str]], model: str) -> float:
        """Estimate cost before sending a request using character-based token approximation.

        Args:
            messages: List of message dicts with 'content' fields.
            model: Model ID to look up pricing for.

        Returns:
            Estimated USD cost as a float.
        """
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_input_tokens = total_chars // 4
        estimated_output_tokens = 256  # conservative default
        pricing = MISTRAL_PRICING.get(model, {"input_per_1m": 0.0, "output_per_1m": 0.0})
        return (estimated_input_tokens * pricing["input_per_1m"] + estimated_output_tokens * pricing["output_per_1m"]) / 1_000_000

    def budget_alert(self, threshold_usd: float) -> bool:
        """Check whether total cost has exceeded a budget threshold.

        Args:
            threshold_usd: Budget limit in US dollars.

        Returns:
            True if the threshold has been exceeded, False otherwise.
        """
        exceeded = self.total_cost > threshold_usd
        if exceeded:
            logger.warning("BUDGET ALERT: $%.4f spent, threshold is $%.4f", self.total_cost, threshold_usd)
        return exceeded


# Demo cost tracker
tracker = CostTracker()
tracker.add_usage("session-1", "mistral-large-latest", input_tokens=500, output_tokens=150)
tracker.add_usage("session-1", "mistral-small-latest", input_tokens=200, output_tokens=80)
tracker.add_usage("session-2", "mistral-embed",        input_tokens=1000, output_tokens=0)

print(f"Total cost: ${tracker.total_cost:.6f}")
print(f"Cost per session: {tracker.cost_per_session()}")
print(f"Daily cost: {tracker.daily_cost()}")

sample_messages = [{"role": "user", "content": "Explain transformers in detail."}]
est = tracker.estimate_cost(sample_messages, "mistral-large-latest")
print(f"Estimated cost for sample prompt: ${est:.6f}")

alert = tracker.budget_alert(threshold_usd=0.001)
print(f"Budget alert triggered: {alert}")
assert tracker.total_cost > 0, "Total cost should be positive after adding usage"
print("CostTracker assertions passed.")

# %% [markdown]
# ## 6. FastAPI Endpoint
# We expose the AI assistant as an HTTP API with streaming SSE responses, session memory,
# cost tracking, and a /stats endpoint for observability.

# %%
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="AI Assistant API", version="1.0.0")

# In-memory stores
_sessions: dict[str, ConversationManager] = {}
_cost_tracker = CostTracker()
_stats = {"total_requests": 0, "total_tokens": 0}
_api_client = Mistral(api_key=MISTRAL_API_KEY)


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    session_id: str
    message: str


class StatsResponse(BaseModel):
    """Response body for the /stats endpoint."""

    total_requests: int
    total_tokens: int
    total_cost_usd: float
    cost_per_session: dict[str, float]


def _get_or_create_session(session_id: str) -> ConversationManager:
    """Retrieve an existing session or create a new one.

    Args:
        session_id: Session identifier.

    Returns:
        The ConversationManager for that session.
    """
    if session_id not in _sessions:
        _sessions[session_id] = ConversationManager(session_id, api_key=MISTRAL_API_KEY)
    return _sessions[session_id]


async def _stream_chat(session_id: str, user_message: str):
    """Async generator yielding SSE-formatted chunks from Mistral streaming API.

    Args:
        session_id: Session to use for conversation history.
        user_message: New user message to respond to.
    """
    session = _get_or_create_session(session_id)
    session.add_message("user", user_message)

    messages = session.rolling_window(max_turns=10)
    est_cost = _cost_tracker.estimate_cost(messages, "mistral-large-latest")
    logger.info("Session %s — estimated call cost: $%.6f", session_id, est_cost)

    full_reply = []
    input_tokens = 0
    output_tokens = 0

    try:
        async_client = AsyncMistral(api_key=MISTRAL_API_KEY)
        async with async_client.chat.stream_async(
            model="mistral-large-latest",
            messages=messages,
        ) as stream:
            async for event in stream:
                delta = event.data.choices[0].delta.content or ""
                full_reply.append(delta)
                output_tokens += len(delta) // 4
                yield f"data: {json.dumps({'delta': delta})}\n\n"

        reply_text = "".join(full_reply)
        session.add_message("assistant", reply_text)

        input_tokens = sum(len(m["content"]) for m in messages) // 4
        _cost_tracker.add_usage(session_id, "mistral-large-latest", input_tokens, output_tokens)
        _stats["total_requests"] += 1
        _stats["total_tokens"] += input_tokens + output_tokens

        if len(session.get_messages()) > 20:
            session.summary_compress(keep_last_turns=4)

        yield f"data: {json.dumps({'done': True, 'total_tokens': input_tokens + output_tokens})}\n\n"

    except SDKError as e:
        logger.error("Streaming error for session %s: %s", session_id, e)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/chat", summary="Stream a chat response using SSE")
async def chat_endpoint(request: ChatRequest) -> StreamingResponse:
    """Handle a chat request and return a streaming SSE response.

    Args:
        request: ChatRequest with session_id and message.

    Returns:
        StreamingResponse with text/event-stream content type.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return StreamingResponse(
        _stream_chat(request.session_id, request.message),
        media_type="text/event-stream",
    )


@app.get("/stats", response_model=StatsResponse, summary="Get API usage statistics")
async def stats_endpoint() -> StatsResponse:
    """Return aggregate usage statistics across all sessions."""
    return StatsResponse(
        total_requests=_stats["total_requests"],
        total_tokens=_stats["total_tokens"],
        total_cost_usd=_cost_tracker.total_cost,
        cost_per_session=_cost_tracker.cost_per_session(),
    )


print("FastAPI app defined with /chat (POST) and /stats (GET) endpoints.")
print("Routes:", [route.path for route in app.routes])

# %% [markdown]
# ## 7. Lab Exercise
# Build and test the complete AI assistant locally.
# The exercise wires together all components: streaming, session memory, cost tracking,
# retry logic, and the /stats endpoint. Run the server with uvicorn and test it with httpx.

# %%
import httpx

LAB_INSTRUCTIONS = """
LAB EXERCISE: Complete AI Assistant
====================================
Goal: Run the FastAPI server and interact with it via httpx.

Steps:
1. Start the server in a terminal:
       uvicorn week3_ai_apis_scale:app --reload --port 8000

2. Run this cell to send test requests and verify the full pipeline.

Expected outcomes:
- Streaming chunks arrive incrementally from /chat
- /stats returns non-zero request and token counts
- Budget alert triggers when cost threshold is exceeded
"""
print(LAB_INSTRUCTIONS)


async def lab_test_client() -> None:
    """Test the running FastAPI server end-to-end using httpx async client.

    Sends two chat messages across the same session, verifies streaming output,
    then checks the /stats endpoint for accumulated usage.
    """
    base_url = "http://localhost:8000"
    session_id = "lab-session-001"

    prompts = [
        "What are the top 3 benefits of async programming?",
        "How does that relate to AI API calls specifically?",
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- Request {i}: {prompt[:50]} ---")
            payload = {"session_id": session_id, "message": prompt}

            start = time.time()
            full_response = []

            async with client.stream("POST", f"{base_url}/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data:
                            print(data["delta"], end="", flush=True)
                            full_response.append(data["delta"])
                        elif data.get("done"):
                            elapsed = time.time() - start
                            print(f"\n[done in {elapsed:.2f}s, tokens: {data.get('total_tokens', '?')}]")

            assert full_response, f"Expected non-empty response for prompt {i}"

        # Check stats
        stats = (await client.get(f"{base_url}/stats")).json()
        print(f"\n/stats: {json.dumps(stats, indent=2)}")
        assert stats["total_requests"] >= len(prompts), "Expected at least 2 requests recorded in /stats"
        print("\nLab test PASSED: streaming, session memory, and /stats all verified.")


# Uncomment and run after starting the server:
# asyncio.run(lab_test_client())

print("Lab client defined. Start the server with uvicorn, then call: asyncio.run(lab_test_client())")

# Standalone unit tests that don't require a running server
def test_cost_tracker_lab() -> None:
    """Unit test: verify CostTracker budget alert and session breakdown."""
    t = CostTracker()
    t.add_usage("lab-1", "mistral-large-latest", 10000, 3000)
    assert t.total_cost > 0
    assert "lab-1" in t.cost_per_session()
    assert t.budget_alert(threshold_usd=0.0001) is True
    print("test_cost_tracker_lab PASSED")


def test_conversation_rolling_window() -> None:
    """Unit test: verify rolling window returns correct number of messages."""
    cm = ConversationManager("test")
    for i in range(10):
        cm.add_message("user", f"msg {i}")
        cm.add_message("assistant", f"reply {i}")
    window = cm.rolling_window(max_turns=3)
    assert len(window) == 6, f"Expected 6 messages, got {len(window)}"
    print("test_conversation_rolling_window PASSED")


test_cost_tracker_lab()
test_conversation_rolling_window()
print("\nAll standalone lab tests passed.")

# %% [markdown]
# ## Key Takeaways
# - Use `asyncio.gather()` with `AsyncMistral` and a `Semaphore` to run concurrent API calls
#   and reduce total wall-clock time by up to N times versus sequential execution.
# - A `ConversationManager` with rolling-window truncation and LLM-based summary compression
#   keeps context within token limits without losing conversational coherence.
# - Tenacity's `@retry` with exponential back-off and a circuit breaker pattern prevents
#   cascading failures and handles transient API errors gracefully in production.
# - A `CostTracker` that maps token counts to USD using per-model pricing lets you estimate
#   costs before a call, set budget alerts, and audit spend per session or per day.
# - FastAPI with `StreamingResponse` and Server-Sent Events (SSE) lets clients receive
#   AI-generated text token-by-token, improving perceived latency significantly.
