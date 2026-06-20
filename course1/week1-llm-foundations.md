# Week 1: How LLMs Actually Work

> **Theme: Build intuition before writing code.** Before you write a single line of production AI code, you need a mental model of what is actually happening inside the systems you are orchestrating. This week strips away the marketing language and builds genuine intuition for transformers, the model landscape, and your first authenticated API call.

---

## 1.1 The Transformer Demystified

### What Is a Transformer, Really?

When engineers say "LLM," they almost always mean a model built on the **transformer architecture**, introduced in the 2017 paper *Attention Is All You Need* by Vaswani et al. Understanding what transformers do — not at the mathematical level, but at the intuitive level — is the single most important foundation for everything else in this curriculum.

A transformer is a function that takes a sequence of tokens and produces a probability distribution over the next token. That sentence is deceptively simple. Everything that feels magical about ChatGPT, Claude, or Gemini is an emergent consequence of applying that same function billions of times across hundreds of billions of examples during training.

### Attention: The Core Intuition

The transformer's central innovation is the **attention mechanism**. Here is the clearest intuition: attention lets every token in a sequence look at every other token and decide how much to "pay attention" to each one when computing its own meaning.

Consider the sentence: *"The animal didn't cross the street because it was too tired."* What does "it" refer to? As a human, you resolve this by attending to "animal" rather than "street." The attention mechanism learns to do exactly this — and it does so for every word, simultaneously, across every layer of the model.

More precisely, each token is represented as three learned vectors:
- A **Query (Q)**: "What am I looking for?"
- A **Key (K)**: "What do I contain?"
- A **Value (V)**: "What will I contribute if selected?"

The model computes dot products between the Query of one token and the Keys of all other tokens to produce **attention scores** — a measure of relevance. Those scores are normalized (via softmax) into weights, and then the weighted sum of all Value vectors becomes the new representation of that token. This is called **scaled dot-product attention**.

Modern transformers use **multi-head attention**, meaning this process runs in parallel across many independent "heads," each learning to attend to different kinds of relationships (syntactic, semantic, positional, etc.).

> **Key Insight:** Attention is not magic — it is a learned routing mechanism. The model learns *which* tokens are relevant to *which* other tokens for *which* purposes. After training on enough text, it learns that pronouns attend to their antecedents, verbs attend to their subjects, and so on.

### Tokens, Not Words

Before attention can operate, text must be converted to **tokens**. This is done by a **tokenizer**, and understanding it prevents a class of common bugs.

Tokens are not words. They are subword units produced by an algorithm called **Byte Pair Encoding (BPE)**. BPE starts with individual characters and iteratively merges the most frequent adjacent pairs until a target vocabulary size is reached (typically 32,000–100,000 tokens). The result: common words become single tokens, rare words get split, and punctuation is often its own token.

Concrete examples using GPT-4's `cl100k_base` tokenizer:
- `"hello"` → 1 token
- `"ChatGPT"` → 3 tokens: `["Chat", "G", "PT"]`
- `"unbelievable"` → 3 tokens: `["un", "believ", "able"]`
- `"2024-01-15"` → 5 tokens: `["2024", "-", "01", "-", "15"]`

This has real engineering consequences. Token counts determine latency, cost, and whether your prompt fits in the context window. A rule of thumb: 1 token ≈ 0.75 English words, or roughly 4 characters. Code and non-English languages are often less efficient.

> **Key Insight:** When you're surprised by a model's behavior with a specific word or name, check the tokenization first. Models have no concept of letters — they see token IDs. "GPT-4" and "GPT4" are entirely different sequences of tokens to the model.

### Parameters as Compressed Knowledge

A language model's **parameters** (also called **weights**) are the numbers adjusted during training to minimize prediction error. GPT-4 is estimated at ~1.8 trillion parameters. Llama 3 70B has 70 billion. What are these numbers actually storing?

Think of parameters as a lossy compression of the training corpus. During training, the model is repeatedly shown text and asked to predict the next token. Every time it gets it wrong, the error is back-propagated through the network and the weights are nudged slightly in the direction that would have made the correct prediction more likely. After trillions of such nudges, the weights encode statistical patterns at every scale: spelling, grammar, facts, reasoning strategies, writing styles.

This is why models "know" things they were never explicitly told — the knowledge is distributed across billions of parameters as implicit patterns, not as a lookup table.

### Context Window and KV-Cache

The **context window** is the maximum number of tokens the model can process in a single forward pass — both the input (prompt) and output (completion) combined. Current models range from 8K tokens (older models) to 1M+ tokens (Gemini 1.5 Pro). Context window size matters enormously: it determines how much conversation history, document content, or codebase the model can "see" simultaneously.

The **KV-cache** (Key-Value cache) is an optimization for inference. During generation, the model produces one token at a time. Without caching, it would need to recompute the Keys and Values for every previous token on every new generation step — O(n²) computation. The KV-cache stores those Key and Value matrices after they are computed, so subsequent tokens only need to compute attention against the cached values. This is why the first token takes longer to generate than subsequent ones, and why hosted APIs charge differently for input vs. output tokens.

> **Key Insight:** The context window is not just a technical limit — it defines the model's "working memory." When context fills up, you must make deliberate choices about what to keep, summarize, or drop. This is one of the core engineering challenges in building production AI systems.

### Transformer Forward Pass Diagram

```mermaid
flowchart TD
    A["Raw Text Input\n'The cat sat on the mat'"] --> B["Tokenizer\nBPE Encoding"]
    B --> C["Token IDs\n[791, 4797, 3139, 389, 279, 7586]"]
    C --> D["Token Embeddings\nLookup Table → Dense Vectors"]
    D --> E["Positional Encoding\nAdd Position Information"]
    E --> F["Transformer Block 1\n(of N layers)"]
    F --> G["Multi-Head Self-Attention\nQ, K, V projections → Attention Scores"]
    G --> H["Add & Layer Norm"]
    H --> I["Feed-Forward Network\n2-layer MLP with activation"]
    I --> J["Add & Layer Norm"]
    J --> K{"More Layers?"}
    K -->|"Yes (repeat N times)"| F
    K -->|"No"| L["Final Layer Norm"]
    L --> M["Linear Projection\nHidden Dim → Vocab Size"]
    M --> N["Softmax\n→ Probability Distribution"]
    N --> O["Sample Next Token\n(temperature, top-p)"]
    O --> P["Append Token, Repeat\nUntil EOS or max_tokens"]

    style A fill:#e8f4f8
    style O fill:#f0f8e8
    style P fill:#f0f8e8
```

### Chapter 1.1 Checkpoint

1. In the sentence "The trophy didn't fit in the suitcase because it was too big," the word "it" refers to "trophy." Describe in plain English how the attention mechanism would resolve this reference. Which token's Query vector would find a high dot-product with which Key vector?

2. The string `"pre-trained"` tokenizes to `["pre", "-", "train", "ed"]` (4 tokens). Estimate how many tokens a 500-word English essay would contain, and explain why code files often use more tokens per word than prose.

3. Why does the KV-cache make inference faster but require more GPU memory? What engineering tradeoff does this represent?

---

## 1.2 The LLM Landscape

### Closed-Source vs. Open-Source: A Real Engineering Decision

The most immediately practical decision in AI engineering is not which model is "smartest" — it is which model is appropriate for your constraints. The landscape divides cleanly into **closed-source frontier models** (accessed via API) and **open-source models** (downloaded and self-hosted).

| Dimension | Closed-Source (Claude, GPT-4, Gemini) | Open-Source (Llama 3, Mistral, Phi-3) |
|---|---|---|
| **Access** | API call, no model weights | Download weights, run anywhere |
| **Cost** | Per-token pricing (ongoing) | Infrastructure cost (one-time + ops) |
| **Privacy** | Data sent to provider | Data stays on your hardware |
| **Customization** | Prompt engineering only | Fine-tune, quantize, modify |
| **Performance** | State-of-the-art, maintained | Slightly behind frontier (closing gap) |
| **Latency** | Network round-trip + queue | Local inference (variable) |
| **Compliance** | Must trust provider's data handling | Full control of data residency |
| **Reliability** | SLA-backed uptime | Your ops team's problem |
| **Context Window** | Up to 1M tokens | Typically 8K–128K |

For enterprise applications handling sensitive data (healthcare records, legal documents, financial PII), open-source self-hosted models are often mandatory regardless of quality tradeoffs. For consumer products where ease of integration and peak quality matter, closed-source APIs win.

### Temperature: The Creativity Dial

**Temperature** is a scalar applied to the logits (raw scores) before the softmax operation during sampling. Lowering temperature makes the distribution sharper (the most likely token becomes even more dominant). Raising it makes the distribution flatter (less likely tokens get more probability mass).

- `temperature=0`: Greedy decoding. Always picks the single most probable token. Fully deterministic, zero creativity. Use for factual extraction, structured output, code generation where correctness matters.
- `temperature=0.7`: The "sweet spot" for most conversational tasks. Some variation, generally coherent.
- `temperature=1.0`: Sample directly from the model's learned distribution. More creative, more likely to go off-track.
- `temperature=2.0`: Very high variance output. Often incoherent. Used in research for diversity sampling.

The key intuition: temperature does not make the model "smarter" or "dumber" — it controls the *randomness of token selection* given the model's probability estimates. A model that assigns 99% probability to the correct next token will still pick it at temperature=2.0 most of the time. Temperature only matters at the margins.

> **Key Insight:** Temperature is not a "quality" knob — it is a "variance" knob. For tasks with objectively correct answers (math, code, data extraction), lower temperature reduces hallucination risk. For creative writing or brainstorming, higher temperature increases diversity at the cost of coherence.

### Top-P: Nucleus Sampling

**Top-p** (also called **nucleus sampling**) is an alternative to temperature for controlling randomness. Instead of scaling the entire distribution, top-p cuts off the "long tail" of low-probability tokens entirely.

With `top_p=0.9`, the model:
1. Sorts all tokens by probability (descending)
2. Sums probabilities until the cumulative total reaches 0.9
3. Discards all remaining tokens
4. Renormalizes the kept tokens and samples from them

The key advantage over temperature: the nucleus size adapts to the model's confidence. When the model is highly confident (one token has 95% probability), the nucleus contains very few tokens. When the model is uncertain (probabilities spread across many tokens), the nucleus stays wide. This avoids both extreme peakiness and extreme diffuseness.

In practice, most production systems use **both** temperature and top-p together. The Anthropic API defaults to `temperature=1.0, top_p=1.0` (no restriction). A common production setting for factual tasks is `temperature=0.0`; for creative tasks, `temperature=0.7, top_p=0.95`.

### Reading a Model Card

A **model card** is the documentation accompanying a model release. Learning to read one quickly is a core skill. Key fields to always check:

**Context Length**: Maximum token input+output combined. Determines what tasks are feasible. `claude-3-5-sonnet-20241022` supports 200K context; `gpt-4o` supports 128K.

**Training Cutoff**: The date after which the model has no training data. A model with a January 2024 cutoff does not know about events after that date. Always check this before deploying for tasks involving recent events.

**Benchmark Scores**: Common benchmarks include MMLU (general knowledge, multiple choice), HumanEval (Python code generation), MATH (competition mathematics), and GPQA (graduate-level science questions). These give rough comparisons but are heavily gamed — treat them as directional, not definitive.

**Pricing**: Closed-source models charge per token, usually with different rates for input vs. output. Output tokens are typically 3-5x more expensive than input tokens because they require sequential generation. At scale, a 1M-token/day application can cost thousands of dollars per month — model selection has direct P&L impact.

> **Key Insight:** Model cards are marketing documents as much as technical ones. The most important number is often not benchmark rank but cost-per-quality-point for your specific task. Run your own evals on representative examples before committing to a model in production.

> **Key Insight:** "Latest" is not always "best for your use case." A smaller, cheaper model fine-tuned on your domain often outperforms a frontier model at general tasks for your specific workload.

### Chapter 1.2 Checkpoint

1. Your company is building a medical records summarization tool. Data cannot leave EU servers, and you have a 10-person ML ops team. Which model category (closed-source vs. open-source) should you choose, and what are the top three constraints driving that decision?

2. You are generating product descriptions for an e-commerce site. The descriptions should be varied and creative, but must always mention the product name accurately. Propose a temperature and top-p configuration and justify your choices.

3. You find two models with similar benchmark scores. Model A costs $15 per million input tokens; Model B costs $0.50 per million input tokens. Your application processes 2 million input tokens per day. Calculate the monthly cost difference and describe what evaluation you would run to determine if Model A is worth the premium.

---

## 1.3 Your First API Call

### Environment Setup

Professional Python development for AI engineering starts with isolated environments. Never install AI packages globally — version conflicts between `anthropic`, `openai`, `langchain`, and their dependencies are common and painful.

```bash
# Create a new project directory
mkdir ai-engineering-week1
cd ai-engineering-week1

# Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# On macOS/Linux
source .venv/bin/activate

# Upgrade pip first (avoids many obscure install errors)
python -m pip install --upgrade pip

# Install the SDKs we will use this week
pip install anthropic openai python-dotenv

# Verify installation
python -c "import anthropic; print(anthropic.__version__)"
```

Store your API keys in a `.env` file, never in source code:

```bash
# .env (add this to .gitignore immediately)
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

### The Anthropic Client and Message Object

The Anthropic Python SDK uses a `client.messages.create()` interface that returns a structured `Message` object. Understanding the full response structure prevents bugs.

```python
# 01_first_call.py
# A complete, annotated first API call to Claude

import os
from dotenv import load_dotenv
import anthropic

# Load API key from .env file
load_dotenv()

# Initialize the client - reads ANTHROPIC_API_KEY from environment automatically
client = anthropic.Anthropic()

# Make a basic completion request
message = client.messages.create(
    model="claude-sonnet-4-5",          # Model identifier (check docs for latest)
    max_tokens=1024,                      # Maximum tokens to generate in response
    messages=[
        {
            "role": "user",
            "content": "Explain what a transformer neural network is in exactly 3 sentences, suitable for a software engineer with no ML background."
        }
    ]
)

# --- Parsing the Message response object ---

# The top-level response has these key fields:
print(f"Model used:      {message.model}")
print(f"Stop reason:     {message.stop_reason}")   # 'end_turn', 'max_tokens', 'stop_sequence'
print(f"Message ID:      {message.id}")

# Content is a list of ContentBlock objects (usually just one TextBlock)
# Always index into content[0] for simple text responses
print(f"\nContent type:    {message.content[0].type}")   # 'text'
print(f"\nResponse text:\n{message.content[0].text}")

# Usage contains token counts - critical for cost tracking
print(f"\n--- Token Usage ---")
print(f"Input tokens:    {message.usage.input_tokens}")
print(f"Output tokens:   {message.usage.output_tokens}")
print(f"Total tokens:    {message.usage.input_tokens + message.usage.output_tokens}")

# Estimate cost (Claude Sonnet pricing as of 2024 — check current docs)
INPUT_COST_PER_MILLION  = 3.00   # USD per 1M input tokens
OUTPUT_COST_PER_MILLION = 15.00  # USD per 1M output tokens

input_cost  = (message.usage.input_tokens  / 1_000_000) * INPUT_COST_PER_MILLION
output_cost = (message.usage.output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
print(f"Estimated cost:  ${input_cost + output_cost:.6f}")
```

### Error Handling

Production code must handle failures gracefully. The Anthropic SDK raises typed exceptions that carry all the information you need.

```python
# 02_error_handling.py
# Robust error handling patterns for production use

import time
import anthropic
from anthropic import APIError, RateLimitError, APIConnectionError, APIStatusError

client = anthropic.Anthropic()

def call_with_retry(
    prompt: str,
    model: str = "claude-sonnet-4-5",
    max_retries: int = 3,
    base_delay: float = 1.0
) -> str | None:
    """
    Make an API call with exponential backoff retry logic.
    Returns the response text, or None if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        except RateLimitError as e:
            # 429: You've exceeded your rate limit (requests per minute or tokens per minute)
            # Always retry with backoff - rate limits are transient
            wait_time = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            print(f"Rate limit hit (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s...")
            time.sleep(wait_time)

        except APIConnectionError as e:
            # Network-level failure - server unreachable, DNS error, etc.
            wait_time = base_delay * (2 ** attempt)
            print(f"Connection error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

        except APIStatusError as e:
            # HTTP 4xx/5xx errors with a response body
            if e.status_code == 529:
                # Anthropic-specific: service overloaded
                print(f"Service overloaded. Retrying...")
                time.sleep(base_delay * (2 ** attempt))
            elif 400 <= e.status_code < 500:
                # Client errors (bad request, auth failure) - do NOT retry
                print(f"Client error {e.status_code}: {e.message}")
                return None
            else:
                # Server errors (5xx) - retry
                print(f"Server error {e.status_code}. Retrying...")
                time.sleep(base_delay * (2 ** attempt))

        except APIError as e:
            # Catch-all for other API errors
            print(f"Unexpected API error: {e}")
            return None

    print(f"All {max_retries} attempts failed.")
    return None


# Test the retry logic
result = call_with_retry("What is 2 + 2?")
if result:
    print(f"Response: {result}")
```

### Streaming: The Right Way to Build Chat UIs

**Streaming** means the model sends tokens to your client as it generates them, rather than waiting until generation is complete. For a response that takes 10 seconds to generate, streaming shows the first words in under 1 second. This is essential for chat interfaces — users tolerate latency much better when they can see progress.

The Anthropic SDK exposes streaming via a context manager that yields events:

```python
# 03_streaming.py
# Streaming response with real-time output

import anthropic

client = anthropic.Anthropic()

def stream_response(prompt: str, model: str = "claude-sonnet-4-5") -> dict:
    """
    Stream a response and return the full text plus token usage.
    """
    print("Assistant: ", end="", flush=True)
    full_text = ""

    # The `stream()` context manager handles the SSE connection
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        # `text_stream` yields string chunks as they arrive
        for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)
            full_text += text_chunk

    print()  # Newline after streaming completes

    # `get_final_message()` blocks until streaming completes
    # and returns the full Message object with usage stats
    final_message = stream.get_final_message()

    return {
        "text": full_text,
        "input_tokens": final_message.usage.input_tokens,
        "output_tokens": final_message.usage.output_tokens,
        "stop_reason": final_message.stop_reason
    }

# Example usage
result = stream_response("Write a haiku about distributed systems.")
print(f"\n[Tokens: {result['input_tokens']} in / {result['output_tokens']} out]")
```

### API Call Lifecycle Diagram

```mermaid
sequenceDiagram
    participant C as Your Client Code
    participant SDK as Anthropic SDK
    participant RL as Rate Limiter<br/>(Anthropic API Gateway)
    participant LB as Load Balancer
    participant M as Model Inference<br/>(GPU Cluster)
    participant KV as KV Cache

    C->>SDK: client.messages.stream(...)
    SDK->>SDK: Validate parameters,<br/>serialize request body
    SDK->>RL: POST /v1/messages<br/>(with API key header)

    alt Rate limit exceeded
        RL-->>SDK: 429 Too Many Requests<br/>(retry-after header)
        SDK-->>C: Raise RateLimitError
    else Within limits
        RL->>LB: Forward request
        LB->>M: Route to available GPU
        M->>KV: Check/populate KV cache<br/>for prompt tokens
        KV-->>M: Cached K,V matrices<br/>(or compute fresh)

        loop Token generation
            M->>M: Forward pass → logit → sample
            M-->>LB: SSE chunk: {"type":"content_block_delta",...}
            LB-->>SDK: Stream token chunk
            SDK-->>C: Yield text chunk<br/>(text_stream iterator)
            C->>C: Print chunk to terminal
        end

        M-->>LB: SSE: {"type":"message_stop",...}
        LB-->>SDK: Final event with usage stats
        SDK-->>C: stream.get_final_message()<br/>returns Message object
    end

    C->>C: Display token count,<br/>append to history
```

### The 50-Line Streaming CLI Chatbot

Here is the complete lab deliverable — a production-quality CLI chatbot that maintains conversation history, streams responses, handles errors, and reports token usage.

```python
# chatbot.py — Streaming CLI chatbot (~50 lines of logic)
# Run: python chatbot.py

import os
import sys
from dotenv import load_dotenv
import anthropic
from anthropic import RateLimitError, APIConnectionError, APIStatusError

load_dotenv()
client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a helpful AI engineering tutor. 
You explain technical concepts clearly with concrete examples. 
When showing code, always include comments."""

def chat(conversation_history: list[dict], user_input: str) -> tuple[str, dict]:
    """Send a message and stream the response. Returns (text, usage)."""
    conversation_history.append({"role": "user", "content": user_input})

    full_response = ""
    try:
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=conversation_history
        ) as stream:
            print("\nAssistant: ", end="", flush=True)
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
                full_response += chunk
            print()

        final = stream.get_final_message()
        usage = {"input": final.usage.input_tokens, "output": final.usage.output_tokens}

    except RateLimitError:
        full_response = "[Rate limit reached. Please wait a moment and try again.]"
        usage = {"input": 0, "output": 0}
        print(f"\n{full_response}")
    except APIConnectionError:
        full_response = "[Connection error. Check your network and try again.]"
        usage = {"input": 0, "output": 0}
        print(f"\n{full_response}")
    except APIStatusError as e:
        full_response = f"[API error {e.status_code}: {e.message}]"
        usage = {"input": 0, "output": 0}
        print(f"\n{full_response}")

    # Only append successful responses to history
    if not full_response.startswith("["):
        conversation_history.append({"role": "assistant", "content": full_response})

    return full_response, usage


def main():
    history = []
    total_input_tokens = 0
    total_output_tokens = 0

    print("AI Engineering Tutor — type 'quit' or 'exit' to stop, 'reset' to clear history")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            history.clear()
            print("[Conversation history cleared]")
            continue

        _, usage = chat(history, user_input)

        total_input_tokens += usage["input"]
        total_output_tokens += usage["output"]

        print(f"\n[Turn tokens: {usage['input']} in / {usage['output']} out | "
              f"Session total: {total_input_tokens} in / {total_output_tokens} out]")


if __name__ == "__main__":
    main()
```

### Chapter 1.3 Checkpoint

1. You call `client.messages.create()` and get back a `Message` object. Write the Python expression to extract the text content of the first content block, and describe what `stop_reason="max_tokens"` tells you about the response.

2. Your chatbot is deployed and receiving 1,000 requests per hour. At 3 AM, you start seeing `RateLimitError` exceptions. Describe three distinct causes this could have (hint: the rate limiter tracks multiple dimensions) and how you would diagnose which one is occurring.

3. Explain why maintaining `conversation_history` as a list of `{"role": ..., "content": ...}` dicts is necessary for multi-turn conversation, and what happens to token costs as the conversation grows longer. What strategy would you use to keep costs bounded in a long-running chat session?

---

## Lab Walkthrough: Building the Streaming CLI Chatbot

### Prerequisites
- Python 3.11 or later installed
- An Anthropic API key (sign up at console.anthropic.com)
- Basic Python familiarity

### Step 1: Project Setup

```bash
mkdir ai-week1-lab
cd ai-week1-lab
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install anthropic python-dotenv
```

Create a `.gitignore` file immediately:

```bash
# .gitignore
.env
.venv/
__pycache__/
*.pyc
```

Create your `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

### Step 2: Verify Your Setup

Before building the chatbot, run a minimal test to confirm authentication works:

```python
# test_connection.py
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

try:
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'connection successful' and nothing else."}]
    )
    print("Status: OK")
    print("Response:", msg.content[0].text)
    print("Tokens:", msg.usage.input_tokens, "in /", msg.usage.output_tokens, "out")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

```bash
python test_connection.py
```

Expected output:
```
Status: OK
Response: Connection successful
Tokens: 21 in / 4 out
```

### Step 3: Build Incrementally

Do not copy-paste the full chatbot at once. Build it in stages, testing each addition:

**Stage 1 — Single non-streaming call:**
Build a function that takes a string and returns a string response. Verify it works.

**Stage 2 — Add streaming:**
Replace `client.messages.create()` with `client.messages.stream()`. Confirm you see character-by-character output.

**Stage 3 — Add conversation history:**
Create the `history` list and append user/assistant turns. Test by asking a follow-up question: "What did I just ask you?" — the model should remember.

**Stage 4 — Add error handling:**
Wrap the API call in try/except blocks for `RateLimitError`, `APIConnectionError`, and `APIStatusError`. Test by temporarily using an invalid API key to trigger an auth error.

**Stage 5 — Add token tracking:**
Extract `stream.get_final_message().usage` and print it after each turn. Add session totals.

**Stage 6 — Add quality-of-life features:**
- The `reset` command to clear history
- Graceful exit on Ctrl+C (`KeyboardInterrupt`)
- The system prompt that specializes the bot's behavior

### Step 4: Test Your Chatbot

Run the full chatbot:

```bash
python chatbot.py
```

Run through this test script manually to verify all features work:

1. Ask: `"What is a transformer?"` — verify streaming output appears
2. Ask: `"Can you give me a code example?"` — verify it uses conversation context
3. Type `reset` — verify history clears
4. Ask `"What did we discuss?"` — verify it no longer remembers (history cleared)
5. Press Ctrl+C — verify graceful exit message

### Step 5: Extend the Lab (Optional Challenges)

- **Add a token budget warning**: Print a warning when session total exceeds 50,000 tokens.
- **Add conversation export**: On exit, save the full conversation history to a JSON file.
- **Add model switching**: Allow the user to type `/model gpt-4o` to switch providers mid-session (requires adding the `openai` SDK).
- **Add response timing**: Print how many seconds each response took using `time.time()`.

---

## Further Reading

1. **"Attention Is All You Need"** — Vaswani et al. (2017). The original transformer paper. The architecture section (Section 3) is remarkably readable for a research paper. Available free on arXiv: arxiv.org/abs/1706.03762

2. **"The Illustrated Transformer"** — Jay Alammar (2018). The single best visual explanation of attention mechanisms. Available at jalammar.github.io/illustrated-transformer/. Read this alongside Section 1.1 of this course.

3. **"Language Models are Few-Shot Learners"** (GPT-3 paper) — Brown et al. (2020). Introduces the concept of in-context learning and demonstrates emergent capabilities from scale. arxiv.org/abs/2005.14165

4. **"Anthropic Model Card for Claude 3"** — Anthropic (2024). A real model card to practice reading: anthropic.com/claude/model-card. Cross-reference the fields discussed in Section 1.2 with this real document.

5. **"Byte Pair Encoding is Suboptimal for Language Model Pretraining"** — Bostrom & Durrett (2020). A deeper dive into why tokenization choices matter and their downstream effects on model performance. arxiv.org/abs/2004.03720

---

## Week Summary

**Five key takeaways from Week 1:**

- **Transformers are learned token routers.** The attention mechanism learns which tokens are relevant to which other tokens, building up representations that encode grammar, facts, and reasoning strategies. There is no hard-coded knowledge — everything is distributed across billions of learned weights.

- **Tokens are not words.** BPE tokenization splits text into subword units, and your cost, latency, and context limits are all denominated in tokens, not words or characters. Develop the habit of checking tokenization for any string that surprises you.

- **Model selection is an engineering decision, not a prestige decision.** The tradeoffs between closed-source and open-source models are real and consequential: privacy, cost, customizability, and compliance requirements often matter more than benchmark rank.

- **Temperature and top-p are variance controls, not quality controls.** Lower temperature for factual/structured tasks; higher temperature for creative tasks. Always evaluate both on representative examples before setting production values.

- **Error handling and token tracking are not optional.** Production AI engineering requires retry logic with exponential backoff, typed exception handling, and per-request cost tracking from day one. The streaming chatbot you built this week is the foundation every subsequent lab will extend.
