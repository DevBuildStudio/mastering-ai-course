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
# # Week 2: Prompt Engineering
# This notebook covers systematic techniques for crafting, testing, and optimising prompts
# with the Mistral API. You will learn how prompt structure, examples, and reasoning
# instructions influence model output quality and consistency.

# %% [markdown]
# ## 1. Setup
# Import dependencies and initialise the synchronous Mistral client. All examples in this
# notebook target `mistral-large-latest` unless a simpler or specialised model is more
# appropriate.

# %%
import os
import re
import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from mistralai import Mistral, AsyncMistral
from mistralai.models import SDKError

load_dotenv()

API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=API_KEY)
MODEL = "mistral-large-latest"
SMALL_MODEL = "mistral-small-latest"

print("Mistral client ready. Model:", MODEL)

# %% [markdown]
# ## 2. Prompt Anatomy
# A `PromptTemplate` separates the *system* instruction (tone, persona, constraints) from
# the *user* turn (the actual query). Placeholders in curly braces are filled via
# `.format(**kwargs)`, keeping templates reusable.

# %%
@dataclass
class PromptTemplate:
    """Reusable prompt template with system and user fields.

    Supports Python-style {placeholder} substitution in both fields via
    the .format() method.
    """

    system: str
    user: str

    def format(self, **kwargs: Any) -> list[dict]:
        """Return a messages list ready for the Mistral chat API.

        Args:
            **kwargs: Values to substitute into {placeholders}.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        return [
            {"role": "system", "content": self.system.format(**kwargs)},
            {"role": "user", "content": self.user.format(**kwargs)},
        ]


def call_model(messages: list[dict], model: str = MODEL) -> str:
    """Send messages to the Mistral API and return the assistant reply.

    Args:
        messages: Chat history list of role/content dicts.
        model: Mistral model identifier.

    Returns:
        Assistant reply string.

    Raises:
        SDKError: On API or network failure.
    """
    try:
        response = client.chat.complete(model=model, messages=messages)
        return response.choices[0].message.content
    except SDKError as exc:
        print(f"[API error] {exc}")
        raise


# --- Three system prompts, same user query ---
user_query = "Explain what a neural network is."

system_prompts = {
    "Professor": (
        "You are a university professor. Use precise academic language, cite concepts "
        "formally, and assume the reader has a STEM background."
    ),
    "Children's tutor": (
        "You are a friendly tutor explaining things to a 10-year-old. Use simple words, "
        "fun analogies, and short sentences."
    ),
    "Stand-up comedian": (
        "You are a stand-up comedian. Explain concepts with jokes, pop-culture references, "
        "and self-deprecating humour, but still be accurate."
    ),
}

for persona, system in system_prompts.items():
    tmpl = PromptTemplate(system=system, user=user_query)
    start = time.time()
    reply = call_model(tmpl.format())
    elapsed = time.time() - start
    print(f"\n=== Persona: {persona} ({elapsed:.1f}s) ===")
    print(reply[:300], "...\n")

# %% [markdown]
# ### Placeholder substitution
# A single template can serve many inputs by filling `{placeholder}` fields per call via
# `.format(**kwargs)`, instead of writing a new system/user pair for each combination.

# %%
explainer_template = PromptTemplate(
    system="You are an expert educator. Explain topics to a {audience} in 2-3 sentences.",
    user="Explain: {topic}",
)

for topic, audience in [
    ("recursion", "first-year CS student"),
    ("recursion", "5-year-old"),
    ("gradient descent", "business executive"),
]:
    messages = explainer_template.format(topic=topic, audience=audience)
    reply = call_model(messages)
    print(f"\n=== {topic} -> {audience} ===")
    print(reply[:300], "...\n")

# %% [markdown]
# ## 3. Zero-Shot and Few-Shot Classification
# *Zero-shot* asks the model to classify without examples. *Few-shot* embeds labelled
# examples directly in the system prompt, dramatically anchoring the output distribution.
# We compare both approaches on ten identical test cases to show the accuracy gap.

# %%
CATEGORIES = ["positive", "negative", "neutral"]

ZERO_SHOT_SYSTEM = (
    "You are a sentiment classifier. "
    "Respond with exactly one word: positive, negative, or neutral."
)

FEW_SHOT_EXAMPLES = """You are a precise sentiment classifier.

Examples:
Text: "I absolutely love this product!" -> positive
Text: "This is the worst experience I've ever had." -> negative
Text: "The package arrived on Tuesday." -> neutral
Text: "Fantastic service, will definitely return!" -> positive
Text: "Mediocre at best, nothing special." -> neutral

Rules:
- Reply with exactly one word: positive, negative, or neutral.
- Do not add punctuation or explanation.
"""


def zero_shot_classify(text: str) -> str:
    """Classify sentiment with no examples (zero-shot).

    Args:
        text: Input text to classify.

    Returns:
        Predicted label string.
    """
    messages = [
        {"role": "system", "content": ZERO_SHOT_SYSTEM},
        {"role": "user", "content": f'Text: "{text}"'},
    ]
    return call_model(messages, model=SMALL_MODEL).strip().lower()


def few_shot_classify(text: str) -> str:
    """Classify sentiment with five labelled examples (few-shot).

    Args:
        text: Input text to classify.

    Returns:
        Predicted label string.
    """
    messages = [
        {"role": "system", "content": FEW_SHOT_EXAMPLES},
        {"role": "user", "content": f'Text: "{text}"'},
    ]
    return call_model(messages, model=SMALL_MODEL).strip().lower()


test_cases = [
    ("This movie was absolutely brilliant!", "positive"),
    ("I regret buying this.", "negative"),
    ("The store opens at 9am.", "neutral"),
    ("Best holiday ever, highly recommended!", "positive"),
    ("Terrible quality, broke on day one.", "negative"),
    ("It came in a brown box.", "neutral"),
    ("Could not be happier with my purchase.", "positive"),
    ("Rude staff, will not return.", "negative"),
    ("The meeting is scheduled for Friday.", "neutral"),
    ("Exceeded all expectations!", "positive"),
]

zero_correct = 0
few_correct = 0

print(f"{'Text':<45} {'Label':<10} {'Zero':<10} {'Few':<10}")
print("-" * 80)
for text, label in test_cases:
    zp = zero_shot_classify(text)
    fp = few_shot_classify(text)
    zero_correct += int(zp == label)
    few_correct += int(fp == label)
    z_mark = "ok" if zp == label else "MISS"
    f_mark = "ok" if fp == label else "MISS"
    print(f"{text[:44]:<45} {label:<10} {zp+' '+z_mark:<10} {fp+' '+f_mark:<10}")

print(f"\nZero-shot accuracy: {zero_correct}/{len(test_cases)}")
print(f"Few-shot  accuracy: {few_correct}/{len(test_cases)}")
assert few_correct >= zero_correct, "Few-shot should be at least as accurate as zero-shot"

# %% [markdown]
# ## 4. Chain-of-Thought Prompting
# Adding "Think step by step:" prompts the model to externalise its reasoning before
# producing a final answer. This substantially improves performance on multi-step problems.
# We extract the numeric answer from the reasoning trace with a regular expression.

# %%
DIRECT_SYSTEM = "Solve the maths problem. Reply with only the numeric answer."

COT_SYSTEM = (
    "Solve the maths problem. "
    "Think step by step, then write your final numeric answer on a line that starts "
    "with 'Answer:'. Do not omit the 'Answer:' line."
)


def direct_solve(problem: str) -> str:
    """Solve a maths word problem directly, without reasoning trace.

    Args:
        problem: Natural-language maths problem.

    Returns:
        Model reply string (expected: bare number).
    """
    messages = [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": problem},
    ]
    return call_model(messages).strip()


def cot_solve(problem: str) -> tuple[str, str | None]:
    """Solve a maths word problem with chain-of-thought reasoning.

    Args:
        problem: Natural-language maths problem.

    Returns:
        Tuple of (full_reasoning, extracted_answer). extracted_answer is
        None if the 'Answer:' line cannot be found.
    """
    messages = [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": problem},
    ]
    reasoning = call_model(messages)
    match = re.search(r"Answer:\s*([\d,.\-]+)", reasoning, re.IGNORECASE)
    answer = match.group(1).replace(",", "") if match else None
    return reasoning, answer


problems = [
    ("A baker makes 48 muffins. She puts them in boxes of 6. "
     "She sells 5 boxes. How many muffins are left?", "18"),
    ("A train travels at 90 km/h for 2.5 hours. How many kilometres does it travel?", "225"),
    ("Maria earns $15/hour. She works 8 hours a day, 5 days a week. "
     "What are her weekly earnings?", "600"),
]

for problem, expected in problems:
    print(f"\nProblem: {problem}")
    direct = direct_solve(problem)
    reasoning, cot_ans = cot_solve(problem)
    print(f"  Direct answer : {direct}")
    print(f"  CoT answer    : {cot_ans}")
    print(f"  Expected      : {expected}")
    print(f"  CoT reasoning snippet: {reasoning[:200]}...")

# %% [markdown]
# ## 5. Output Format Control
# Supplying `response_format={"type":"json_object"}` guarantees a parseable JSON string.
# We compare a free-text response with a schema-constrained JSON response to show how
# format enforcement eliminates post-processing edge cases.

# %%
SCHEMA_SYSTEM = """You extract structured data from product reviews.

Return ONLY valid JSON matching this schema exactly:
{
  "sentiment": "positive" | "negative" | "neutral",
  "score": <integer 1-5>,
  "key_topics": [<string>, ...],
  "summary": "<one sentence>"
}"""

FREE_SYSTEM = (
    "Extract sentiment, a score (1-5), key topics, and a one-sentence summary "
    "from the product review."
)

REVIEW = (
    "The noise-cancelling headphones are phenomenal for flights. "
    "Battery life could be better but the sound quality makes up for it. "
    "Comfortable for long sessions too. Would buy again."
)


def extract_free(review: str) -> str:
    """Extract review insights as unstructured free text.

    Args:
        review: Raw product review text.

    Returns:
        Model reply string (unstructured).
    """
    messages = [
        {"role": "system", "content": FREE_SYSTEM},
        {"role": "user", "content": review},
    ]
    return call_model(messages)


def extract_json(review: str) -> dict:
    """Extract review insights as validated JSON.

    Args:
        review: Raw product review text.

    Returns:
        Parsed dict conforming to the schema.

    Raises:
        json.JSONDecodeError: If the model returns unparseable JSON.
    """
    messages = [
        {"role": "system", "content": SCHEMA_SYSTEM},
        {"role": "user", "content": review},
    ]
    raw = client.chat.complete(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},
    ).choices[0].message.content
    return json.loads(raw)


print("=== Free-text extraction ===")
free_result = extract_free(REVIEW)
print(free_result)

print("\n=== JSON extraction ===")
json_result = extract_json(REVIEW)
print(json.dumps(json_result, indent=2))

# Validate required keys
required_keys = {"sentiment", "score", "key_topics", "summary"}
assert required_keys.issubset(json_result.keys()), "Missing keys in JSON output"
assert isinstance(json_result["score"], int), "Score must be an integer"
assert isinstance(json_result["key_topics"], list), "key_topics must be a list"
print("\nValidation passed: all required keys present with correct types.")

# %% [markdown]
# ## 6. Prompt A/B Testing
# `PromptABTest` runs multiple prompt variants against the same test cases using the async
# Mistral client for efficiency. `compare_results()` tallies wins per prompt based on a
# simple quality heuristic (response length as a proxy), then declares a winner.

# %%
@dataclass
class PromptABTest:
    """Run and compare multiple prompt variants on a shared test suite.

    Uses the AsyncMistral client to fetch all responses in parallel,
    minimising wall-clock time.

    Args:
        prompts: List of system-prompt strings to compare.
        model: Mistral model identifier for all variants.
    """

    prompts: list[str]
    model: str = SMALL_MODEL
    results: dict[int, list[str]] = field(default_factory=dict)

    async def _fetch(
        self,
        async_client: AsyncMistral,
        system: str,
        user: str,
    ) -> str:
        """Fetch a single completion asynchronously.

        Args:
            async_client: Async Mistral client instance.
            system: System prompt string.
            user: User message string.

        Returns:
            Assistant reply content.
        """
        try:
            response = await async_client.chat.complete_async(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content
        except SDKError as exc:
            return f"[error: {exc}]"

    async def run_test(self, test_cases: list[str]) -> None:
        """Run all prompt variants across all test cases concurrently.

        Args:
            test_cases: List of user message strings.
        """
        async_client = AsyncMistral(api_key=API_KEY)
        self.results = {i: [] for i in range(len(self.prompts))}

        tasks = [
            self._fetch(async_client, system, user)
            for user in test_cases
            for i, system in enumerate(self.prompts)
        ]
        flat = await asyncio.gather(*tasks)

        # Re-map flat list back to per-prompt buckets
        n_prompts = len(self.prompts)
        n_cases = len(test_cases)
        for case_idx in range(n_cases):
            for prompt_idx in range(n_prompts):
                self.results[prompt_idx].append(
                    flat[case_idx * n_prompts + prompt_idx]
                )

    def compare_results(self) -> dict[str, Any]:
        """Compare prompts by average response length (quality proxy).

        Returns:
            Dict with per-prompt stats and the index of the winning prompt.
        """
        stats: dict[str, Any] = {}
        for idx, replies in self.results.items():
            avg_len = sum(len(r) for r in replies) / max(len(replies), 1)
            stats[f"prompt_{idx}"] = {
                "avg_response_length": round(avg_len, 1),
                "n_replies": len(replies),
                "sample": replies[0][:120] if replies else "",
            }
        winner = max(self.results, key=lambda i: sum(len(r) for r in self.results[i]))
        stats["winner"] = f"prompt_{winner}"
        return stats


# Define two prompt variants for a summarisation task
ab_prompts = [
    "Summarise the following text in one sentence.",
    (
        "You are an expert editor. Produce a crisp, informative one-sentence summary "
        "that captures the main idea. Avoid starting with 'The text' or 'This article'."
    ),
]

ab_cases = [
    "Climate change refers to long-term shifts in global temperatures and weather patterns. "
    "While some change is natural, human activities—especially burning fossil fuels—have "
    "been the main driver since the 1800s.",

    "The Python programming language was created by Guido van Rossum and first released in "
    "1991. It emphasises code readability and supports multiple programming paradigms.",

    "Exercise has been shown to improve cardiovascular health, boost mood through endorphin "
    "release, and reduce the risk of chronic diseases such as diabetes and hypertension.",
]

ab_test = PromptABTest(prompts=ab_prompts)
start = time.time()
asyncio.run(ab_test.run_test(ab_cases))
elapsed = time.time() - start

comparison = ab_test.compare_results()
print(f"A/B test completed in {elapsed:.1f}s")
print(json.dumps(comparison, indent=2))
print(f"\nWinner: {comparison['winner']}")

# %% [markdown]
# ## 7. Lab Exercise: Customer Support Prompt Library
# Build a five-variant prompt library for a customer support agent. Each variant
# differs in persona, instruction style, or output constraints. Evaluate all variants
# on 20 synthetic support tickets and report the best prompt by average response quality
# (scored by response completeness and adherence to format).

# %%
# --- Prompt Library ---

PROMPT_LIBRARY = {
    "v1_basic": (
        "You are a customer support agent. Help the customer with their issue."
    ),
    "v2_empathetic": (
        "You are a warm, empathetic customer support specialist. "
        "Always acknowledge the customer's frustration, apologise sincerely, "
        "and provide a clear, actionable resolution in 3 steps or fewer."
    ),
    "v3_structured": (
        "You are a customer support agent. Reply in this exact format:\n"
        "ACKNOWLEDGEMENT: <one sentence empathising with the issue>\n"
        "ROOT CAUSE: <brief diagnosis>\n"
        "RESOLUTION: <numbered steps to fix>\n"
        "FOLLOW-UP: <what to do if problem persists>"
    ),
    "v4_concise": (
        "You are a concise customer support bot. Solve the problem in under 60 words. "
        "No pleasantries. Bullet points preferred."
    ),
    "v5_expert": (
        "You are a senior technical support engineer with 10 years of experience. "
        "Diagnose the root cause, explain it clearly to a non-technical user, "
        "and provide a step-by-step resolution with estimated time for each step."
    ),
}

# --- 20 synthetic support tickets ---
SUPPORT_TICKETS = [
    "My order hasn't arrived and it's been 2 weeks.",
    "I was charged twice for the same item.",
    "The app crashes every time I try to log in.",
    "I can't reset my password — the email never arrives.",
    "My subscription was cancelled but I'm still being billed.",
    "The product I received is different from what I ordered.",
    "I need to change my delivery address but the order is already placed.",
    "My account has been locked and I don't know why.",
    "The discount code I applied isn't showing on my invoice.",
    "I returned an item 3 weeks ago but still haven't received a refund.",
    "The website is showing an error when I try to checkout.",
    "I accidentally placed two identical orders — how do I cancel one?",
    "My gift card balance disappeared after the last app update.",
    "The product stopped working after 2 days. How do I get a replacement?",
    "I'm being asked to verify my identity but I don't have the required documents.",
    "My tracking number says delivered but nothing arrived.",
    "I was promised free shipping but was charged for it.",
    "The live chat support told me to email, and email told me to call. Help!",
    "I need an invoice for my last purchase for expense reporting.",
    "The product size guide was wrong and now the item doesn't fit.",
]


def score_response(response: str) -> float:
    """Score a support response on completeness and format adherence (0–10).

    Heuristic scoring:
    - Length (>50 chars) = up to 4 points
    - Contains apology/acknowledgement = 2 points
    - Contains numbered steps or bullets = 2 points
    - Contains follow-up or contact info = 2 points

    Args:
        response: Model-generated support reply.

    Returns:
        Score between 0.0 and 10.0.
    """
    score = 0.0
    length = len(response)
    score += min(length / 50, 4.0)  # up to 4 pts for length
    if re.search(r"(sorry|apologise|apologi[sz]e|understand your)", response, re.I):
        score += 2.0
    if re.search(r"(\d+\.\s|\n[-*•])", response):
        score += 2.0
    if re.search(r"(contact|reach out|follow.?up|don.t hesitate)", response, re.I):
        score += 2.0
    return round(min(score, 10.0), 2)


async def run_support_eval() -> dict[str, Any]:
    """Evaluate all prompt variants across all support tickets asynchronously.

    Returns:
        Dict mapping prompt name to average quality score and sample reply.
    """
    async_client = AsyncMistral(api_key=API_KEY)

    async def fetch(system: str, ticket: str) -> str:
        try:
            resp = await async_client.chat.complete_async(
                model=SMALL_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": ticket},
                ],
            )
            return resp.choices[0].message.content
        except SDKError as exc:
            return f"[error: {exc}]"

    tasks = [
        fetch(system, ticket)
        for ticket in SUPPORT_TICKETS
        for system in PROMPT_LIBRARY.values()
    ]

    start = time.time()
    flat = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    n_prompts = len(PROMPT_LIBRARY)
    n_tickets = len(SUPPORT_TICKETS)
    prompt_names = list(PROMPT_LIBRARY.keys())

    report: dict[str, Any] = {}
    for p_idx, name in enumerate(prompt_names):
        replies = [
            flat[t_idx * n_prompts + p_idx] for t_idx in range(n_tickets)
        ]
        scores = [score_response(r) for r in replies]
        avg = round(sum(scores) / len(scores), 2)
        report[name] = {
            "avg_score": avg,
            "min_score": min(scores),
            "max_score": max(scores),
            "sample_reply": replies[0][:200],
        }

    best = max(report, key=lambda k: report[k]["avg_score"])
    return {
        "elapsed_seconds": round(elapsed, 1),
        "n_tickets": n_tickets,
        "results": report,
        "best_prompt": best,
        "best_avg_score": report[best]["avg_score"],
    }


print("Running customer support prompt evaluation (20 tickets x 5 prompts)...")
eval_report = asyncio.run(run_support_eval())

print(f"\nCompleted in {eval_report['elapsed_seconds']}s — {eval_report['n_tickets']} tickets\n")
print(f"{'Prompt':<20} {'Avg':>6} {'Min':>6} {'Max':>6}")
print("-" * 42)
for name, stats in eval_report["results"].items():
    print(f"{name:<20} {stats['avg_score']:>6} {stats['min_score']:>6} {stats['max_score']:>6}")

print(f"\nBest prompt: {eval_report['best_prompt']} "
      f"(avg score {eval_report['best_avg_score']}/10)")
print(f"\nSample reply from best prompt:\n{eval_report['results'][eval_report['best_prompt']]['sample_reply']}")

# %% [markdown]
# ## Key Takeaways
# - **System prompts control persona and tone**: the same user query yields radically
#   different responses depending on the system instruction — use this to lock in brand
#   voice and guardrails.
# - **Few-shot examples anchor output distribution**: even five labelled examples can
#   lift classification accuracy significantly compared to zero-shot, because examples
#   remove label-space ambiguity.
# - **Chain-of-thought improves multi-step reasoning**: prefixing with "Think step by
#   step" externalises reasoning and reduces arithmetic or logic errors; always extract
#   the final answer with a regex on a labelled line.
# - **JSON response_format eliminates post-processing fragility**: schema-constrained
#   output removes the need for brittle string parsing and makes downstream code
#   deterministic.
# - **Async A/B testing scales evaluation cheaply**: running prompt variants concurrently
#   with `AsyncMistral` collapses wall-clock time from O(n_prompts × n_cases) to
#   O(max_latency), making systematic prompt iteration practical even at scale.
