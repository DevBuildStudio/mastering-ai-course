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
# # Course 1, Week 1: How LLMs Actually Work
#
# This notebook introduces the Mistral AI API and core LLM concepts.
# You will learn how to make API calls, stream responses, run async requests,
# compare models, and build an interactive CLI chatbot — all from first principles.

# %% [markdown]
# ## 1. Setup and Installation
#
# Install the Mistral client and dotenv, then authenticate.
# We list available models to confirm the connection is live.

# %%
# !pip install mistralai python-dotenv --quiet

import os
import time
import asyncio
from dotenv import load_dotenv
from mistralai import Mistral, AsyncMistral

load_dotenv()

API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
assert API_KEY != "your-key-here", "Set MISTRAL_API_KEY in your .env file or environment."

client = Mistral(api_key=API_KEY)

# List available models
try:
    models_response = client.models.list()
    model_ids = [m.id for m in models_response.data]
    print("Available models:")
    for mid in sorted(model_ids):
        print(f"  - {mid}")
    print(f"\nTotal models available: {len(model_ids)}")
except Exception as e:
    print(f"Error listing models: {e}")

# %% [markdown]
# ## 2. Your First API Call
#
# `client.chat.complete()` sends a synchronous request and returns a full response object.
# We inspect the response structure: choices, message content, and token usage.

# %%
def first_api_call(prompt: str, model: str = "mistral-small-latest") -> dict:
    """
    Send a single chat completion request and return content plus usage stats.

    Args:
        prompt: The user message to send.
        model: Mistral model identifier.

    Returns:
        dict with keys: content, prompt_tokens, completion_tokens, total_tokens
    """
    try:
        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        usage = response.usage
        return {
            "content": content,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    except Exception as e:
        print(f"API call failed: {e}")
        raise


prompt = "Explain what a large language model is in exactly two sentences."
result = first_api_call(prompt)

print("=== Response ===")
print(result["content"])
print("\n=== Token Usage ===")
print(f"  Prompt tokens    : {result['prompt_tokens']}")
print(f"  Completion tokens: {result['completion_tokens']}")
print(f"  Total tokens     : {result['total_tokens']}")

assert isinstance(result["content"], str) and len(result["content"]) > 0, "Empty response"
assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
print("\nAll assertions passed.")

# %% [markdown]
# ## 3. Streaming Responses
#
# `client.chat.stream()` yields delta chunks as they arrive, enabling real-time display.
# We measure time-to-first-token (TTFT) and total latency to understand the tradeoff.

# %%
def stream_response(prompt: str, model: str = "mistral-small-latest") -> dict:
    """
    Stream a chat completion and print tokens as they arrive.

    Args:
        prompt: The user message to send.
        model: Mistral model identifier.

    Returns:
        dict with keys: full_text, ttft_seconds, total_seconds, chunk_count
    """
    full_text = ""
    ttft = None
    chunk_count = 0
    start = time.time()

    print("=== Streaming Output ===")
    try:
        with client.chat.stream(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                delta = event.data.choices[0].delta.content
                if delta:
                    if ttft is None:
                        ttft = time.time() - start
                    print(delta, end="", flush=True)
                    full_text += delta
                    chunk_count += 1
    except Exception as e:
        print(f"\nStreaming failed: {e}")
        raise

    total = time.time() - start
    print("\n")
    return {
        "full_text": full_text,
        "ttft_seconds": round(ttft or 0, 3),
        "total_seconds": round(total, 3),
        "chunk_count": chunk_count,
    }


stream_prompt = "Describe the transformer architecture in three short paragraphs."
stats = stream_response(stream_prompt)

print("=== Latency Stats ===")
print(f"  Time-to-first-token : {stats['ttft_seconds']}s")
print(f"  Total time          : {stats['total_seconds']}s")
print(f"  Chunks received     : {stats['chunk_count']}")
print(f"  Characters received : {len(stats['full_text'])}")

# %% [markdown]
# ## 4. Async API Calls
#
# `AsyncMistral` enables non-blocking calls via Python's `asyncio`.
# We run 3 concurrent requests and compare the wall-clock time against sequential sync calls.

# %%
async def async_chat(client_async: AsyncMistral, prompt: str, label: str) -> dict:
    """
    Make a single async chat completion call.

    Args:
        client_async: An AsyncMistral client instance.
        prompt: User message.
        label: Identifier for logging.

    Returns:
        dict with keys: label, content, elapsed_seconds
    """
    start = time.time()
    try:
        response = await client_async.chat.complete_async(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
    except Exception as e:
        content = f"Error: {e}"
    elapsed = round(time.time() - start, 3)
    return {"label": label, "content": content, "elapsed_seconds": elapsed}


async def run_concurrent_requests() -> None:
    """Run 3 async requests concurrently and report total wall-clock time."""
    async_client = AsyncMistral(api_key=API_KEY)
    prompts = [
        ("What is attention mechanism?", "Request-1"),
        ("What is tokenization?", "Request-2"),
        ("What is temperature in LLMs?", "Request-3"),
    ]

    print("=== Running 3 Concurrent Async Requests ===")
    wall_start = time.time()
    tasks = [async_chat(async_client, p, label) for p, label in prompts]
    results = await asyncio.gather(*tasks)
    wall_total = round(time.time() - wall_start, 3)

    for r in results:
        print(f"\n[{r['label']}] ({r['elapsed_seconds']}s)")
        print(r["content"][:200] + ("..." if len(r["content"]) > 200 else ""))

    print(f"\nTotal wall-clock time (concurrent): {wall_total}s")
    individual_sum = sum(r["elapsed_seconds"] for r in results)
    print(f"Sum of individual latencies       : {individual_sum}s")
    print(f"Concurrency speedup               : ~{round(individual_sum / wall_total, 1)}x")


asyncio.run(run_concurrent_requests())

# %% [markdown]
# ## 5. Model Comparison
#
# We benchmark `mistral-large-latest` vs `mistral-small-latest` on the same prompt.
# Metrics: latency, token count, response length, and estimated cost.

# %%
# Pricing as of mid-2025 (USD per 1M tokens) — check mistral.ai for current rates
PRICING = {
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
    "mistral-small-latest": {"input": 0.20, "output": 0.60},
}


def compare_models(prompt: str) -> None:
    """
    Run the same prompt on large and small models, then print a comparison table.

    Args:
        prompt: The user message to send to both models.
    """
    results = {}
    for model in ["mistral-large-latest", "mistral-small-latest"]:
        start = time.time()
        try:
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = round(time.time() - start, 3)
            content = response.choices[0].message.content
            usage = response.usage
            pricing = PRICING[model]
            cost = (
                usage.prompt_tokens / 1_000_000 * pricing["input"]
                + usage.completion_tokens / 1_000_000 * pricing["output"]
            )
            results[model] = {
                "elapsed": elapsed,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": round(cost, 6),
                "content": content,
            }
        except Exception as e:
            print(f"Error with {model}: {e}")
            results[model] = None

    print(f"{'Metric':<25} {'mistral-large':>20} {'mistral-small':>20}")
    print("-" * 67)
    metrics = [
        ("Latency (s)", "elapsed"),
        ("Prompt tokens", "prompt_tokens"),
        ("Completion tokens", "completion_tokens"),
        ("Total tokens", "total_tokens"),
        ("Estimated cost ($)", "cost_usd"),
    ]
    for label, key in metrics:
        large_val = results.get("mistral-large-latest", {}).get(key, "N/A")
        small_val = results.get("mistral-small-latest", {}).get(key, "N/A")
        print(f"{label:<25} {str(large_val):>20} {str(small_val):>20}")

    print("\n=== Response Quality ===")
    for model, data in results.items():
        if data:
            print(f"\n[{model}]")
            print(data["content"][:300] + ("..." if len(data["content"]) > 300 else ""))


comparison_prompt = "What are the key differences between GPT-style and BERT-style language models?"
compare_models(comparison_prompt)

# %% [markdown]
# ## 6. CLI Chatbot
#
# A minimal interactive chatbot that maintains conversation history.
# Responses are streamed character by character. Press Ctrl+C to exit.

# %%
def run_chatbot(model: str = "mistral-small-latest") -> None:
    """
    Run a simple CLI chatbot with streaming output and persistent conversation history.

    Args:
        model: Mistral model identifier to use for responses.
    """
    messages = []
    print("=== Mistral CLI Chatbot ===")
    print("Type your message and press Enter. Press Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            print("Bot: ", end="", flush=True)

            assistant_reply = ""
            try:
                with client.chat.stream(model=model, messages=messages) as stream:
                    for event in stream:
                        delta = event.data.choices[0].delta.content
                        if delta:
                            print(delta, end="", flush=True)
                            assistant_reply += delta
            except Exception as e:
                print(f"\n[Error during streaming: {e}]")
                messages.pop()
                continue

            print("\n")
            messages.append({"role": "assistant", "content": assistant_reply})

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


# Uncomment the line below to run the chatbot interactively:
# run_chatbot()
print("Chatbot defined. Call run_chatbot() to start an interactive session.")

# %% [markdown]
# ## 7. Lab Exercise: Production Chatbot
#
# Build a complete streaming chatbot with conversation history, token counter,
# cost tracker, `/clear` command, `/history` command, and graceful exit.
# This is a self-contained challenge — read through it, then run it.

# %%
class ChatSession:
    """
    A stateful chat session with token counting, cost tracking, and command handling.

    Supports special commands:
        /clear   - Reset conversation history
        /history - Show all turns so far
        /quit    - Exit the session
    """

    COST_PER_1M = {
        "mistral-large-latest": {"input": 2.00, "output": 6.00},
        "mistral-small-latest": {"input": 0.20, "output": 0.60},
    }

    def __init__(self, model: str = "mistral-small-latest") -> None:
        """
        Initialize the chat session.

        Args:
            model: Mistral model to use for all turns.
        """
        self.model = model
        self.messages: list[dict] = []
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.session_cost_usd: float = 0.0

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate the cost of a single API call.

        Args:
            prompt_tokens: Number of tokens in the prompt.
            completion_tokens: Number of tokens in the completion.

        Returns:
            Estimated cost in USD.
        """
        pricing = self.COST_PER_1M.get(self.model, {"input": 0.0, "output": 0.0})
        return (
            prompt_tokens / 1_000_000 * pricing["input"]
            + completion_tokens / 1_000_000 * pricing["output"]
        )

    def _show_history(self) -> None:
        """Print the full conversation history with turn labels."""
        if not self.messages:
            print("[No conversation history yet]")
            return
        print("\n=== Conversation History ===")
        for i, msg in enumerate(self.messages, 1):
            role = msg["role"].upper()
            print(f"[{i}] {role}: {msg['content'][:120]}{'...' if len(msg['content']) > 120 else ''}")
        print("=" * 30 + "\n")

    def _show_stats(self) -> None:
        """Print cumulative token usage and cost for the session."""
        total = self.total_prompt_tokens + self.total_completion_tokens
        print(f"\n[Session Stats] Prompt: {self.total_prompt_tokens} | "
              f"Completion: {self.total_completion_tokens} | "
              f"Total: {total} | "
              f"Cost: ${self.session_cost_usd:.6f}")

    def _handle_command(self, cmd: str) -> bool:
        """
        Handle a slash command.

        Args:
            cmd: The command string (e.g. '/clear').

        Returns:
            True if the session should continue, False if it should exit.
        """
        cmd = cmd.strip().lower()
        if cmd == "/clear":
            self.messages.clear()
            print("[Conversation history cleared]")
        elif cmd == "/history":
            self._show_history()
        elif cmd in ("/quit", "/exit", "/q"):
            return False
        else:
            print(f"[Unknown command: {cmd}. Available: /clear, /history, /quit]")
        return True

    def send(self, user_input: str) -> None:
        """
        Send a user message, stream the response, and update accounting.

        Args:
            user_input: The raw text from the user.
        """
        self.messages.append({"role": "user", "content": user_input})
        print(f"[{self.model}] Bot: ", end="", flush=True)

        assistant_reply = ""
        start = time.time()
        # Use a non-streaming call here to get accurate token counts
        try:
            response = client.chat.complete(
                model=self.model,
                messages=self.messages,
            )
            assistant_reply = response.choices[0].message.content
            usage = response.usage
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens
            call_cost = self._estimate_cost(usage.prompt_tokens, usage.completion_tokens)
            self.session_cost_usd += call_cost

            # Simulate streaming output for display
            for char in assistant_reply:
                print(char, end="", flush=True)
                time.sleep(0.002)
            print()

            elapsed = round(time.time() - start, 2)
            print(f"  [{elapsed}s | +{usage.completion_tokens} tokens | +${call_cost:.6f}]")

        except Exception as e:
            print(f"\n[API Error: {e}]")
            self.messages.pop()
            return

        self.messages.append({"role": "assistant", "content": assistant_reply})

    def run(self) -> None:
        """Start the interactive chat loop."""
        print(f"\n=== Production Chatbot (model: {self.model}) ===")
        print("Commands: /clear  /history  /quit\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.startswith("/"):
                    should_continue = self._handle_command(user_input)
                    if not should_continue:
                        break
                    continue
                self.send(user_input)

            except KeyboardInterrupt:
                print("\n[Interrupted]")
                break

        self._show_stats()
        print("Session ended.")


# Demo: run one automated turn to show the system works without interactive input
print("=== Lab Exercise Demo (single automated turn) ===")
demo_session = ChatSession(model="mistral-small-latest")
demo_session.send("In one sentence, what is a token in the context of LLMs?")
demo_session._show_stats()

print("\nTo run the full interactive chatbot, call:")
print("  session = ChatSession(model='mistral-small-latest')")
print("  session.run()")

# %% [markdown]
# ## Key Takeaways
#
# - **Token-based pricing**: LLMs charge per input and output token; tracking usage is essential for cost control in production applications.
# - **Streaming reduces perceived latency**: Time-to-first-token is far more important for UX than total generation time — stream whenever the UI can display incremental output.
# - **Async multiplies throughput**: Running concurrent async requests with `AsyncMistral` can cut wall-clock time by 2-4x compared to sequential sync calls.
# - **Model size is a cost-quality tradeoff**: `mistral-large-latest` costs ~10x more per token than `mistral-small-latest`; benchmark on your actual task before committing to the larger model.
# - **Conversation history is stateful context**: The API is stateless — you must pass the full `messages` list on every call to maintain multi-turn coherence, which means prompt tokens grow with each turn.
