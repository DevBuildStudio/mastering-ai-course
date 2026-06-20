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
# ## Week 6: Fine-Tuning Models
# Fine-tuning adapts a pretrained model to a specific task or domain by continuing training on
# curated examples. This notebook covers dataset preparation, synthetic data generation, the
# Mistral fine-tuning API, evaluation with an LLM-as-judge, and a LoRA/QLoRA alternative
# using HuggingFace PEFT.

# %% [markdown]
# ## 1. Setup
# Install dependencies and configure the Mistral client.

# %%
# !pip install mistralai python-dotenv datasets matplotlib peft transformers trl bitsandbytes accelerate

import os
import json
import time
import asyncio
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models import SDKError

load_dotenv()

API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=API_KEY)

print("Mistral client initialised.")
print(f"API key present: {bool(API_KEY and API_KEY != 'your-key-here')}")

# %% [markdown]
# ## 2. Dataset Preparation
# Fine-tuning data for Mistral must be in JSONL format where each line contains a `messages`
# array with alternating user/assistant turns. A `DatasetBuilder` class handles cleaning,
# validation, and serialisation.

# %%
import unicodedata


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters from a string."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def validate_pair(user: str, assistant: str) -> bool:
    """Return True when both user prompt and assistant reply meet minimum quality bars."""
    if not user or not assistant:
        return False
    if len(user.split()) < 3 or len(assistant.split()) < 5:
        return False
    if len(user) > 4096 or len(assistant) > 8192:
        return False
    return True


def save_jsonl(data: list[dict], path: str) -> None:
    """Serialise a list of message-dicts to a JSONL file at *path*."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in data:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} records -> {path}")


class DatasetBuilder:
    """Build, validate, and persist a fine-tuning dataset in Mistral JSONL format."""

    def __init__(self) -> None:
        """Initialise with an empty record list."""
        self.records: list[dict] = []

    def add_pair(self, user: str, assistant: str, system: str | None = None) -> bool:
        """Clean and validate a user/assistant pair then append it to the dataset."""
        user = clean_text(user)
        assistant = clean_text(assistant)
        if not validate_pair(user, assistant):
            return False
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": clean_text(system)})
        messages += [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        self.records.append({"messages": messages})
        return True

    def stats(self) -> dict:
        """Return summary statistics for the current dataset."""
        if not self.records:
            return {"count": 0}
        user_lens = [
            len(m["content"])
            for r in self.records
            for m in r["messages"]
            if m["role"] == "user"
        ]
        asst_lens = [
            len(m["content"])
            for r in self.records
            for m in r["messages"]
            if m["role"] == "assistant"
        ]
        return {
            "count": len(self.records),
            "avg_user_len": round(sum(user_lens) / len(user_lens), 1),
            "avg_assistant_len": round(sum(asst_lens) / len(asst_lens), 1),
        }

    def save(self, path: str) -> None:
        """Delegate persistence to the module-level save_jsonl helper."""
        save_jsonl(self.records, path)


# Demo: build a tiny dataset from hand-written pairs
builder = DatasetBuilder()
sample_pairs = [
    ("What is gradient descent?",
     "Gradient descent is an optimisation algorithm that iteratively moves model parameters "
     "in the direction that reduces the loss function by following the negative gradient."),
    ("Explain overfitting in one sentence.",
     "Overfitting occurs when a model learns the training data too closely, including its noise, "
     "and therefore generalises poorly to unseen examples."),
    ("What does a learning rate control?",
     "The learning rate determines the step size taken along the gradient at each update; "
     "too large causes divergence, too small causes slow convergence."),
]
for u, a in sample_pairs:
    builder.add_pair(u, a)

print("Dataset stats:", builder.stats())
assert builder.stats()["count"] == 3, "Expected 3 validated records"

# %% [markdown]
# ## 3. Synthetic Data Generation
# Using `mistral-large-latest` to generate diverse Q&A pairs that will later train a smaller
# model is a common and cost-effective strategy. Deduplication via embedding similarity prevents
# near-duplicate examples from inflating the dataset.

# %%
import numpy as np


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two embedding vectors."""
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


class SyntheticDataGenerator:
    """Generate synthetic fine-tuning pairs using Mistral and deduplicate by embedding."""

    def __init__(self, client: Mistral) -> None:
        """Attach the shared Mistral client."""
        self.client = client

    def generate_qa_pair(self, topic: str, context: str = "") -> dict | None:
        """Call mistral-large-latest to produce one Q&A pair for *topic*.

        Returns a dict with keys ``user`` and ``assistant``, or None on failure.
        """
        ctx_clause = f" Use this context: {context}" if context else ""
        prompt = (
            f"Generate one realistic question-and-answer pair about: {topic}.{ctx_clause}\n"
            "Respond ONLY with valid JSON in this exact schema:\n"
            '{"question": "...", "answer": "..."}\n'
            "The answer must be at least two sentences."
        )
        try:
            response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return {"user": data["question"], "assistant": data["answer"]}
        except (SDKError, json.JSONDecodeError, KeyError) as exc:
            print(f"[generate_qa_pair] error for topic '{topic}': {exc}")
            return None

    def generate_variations(self, example: dict, n: int = 3) -> list[dict]:
        """Produce *n* paraphrased variations of an existing Q&A pair."""
        prompt = (
            f"Given this Q&A:\nQ: {example['user']}\nA: {example['assistant']}\n\n"
            f"Generate {n} paraphrased variations. "
            f"Respond ONLY with a JSON array of objects each with keys 'question' and 'answer'."
        )
        try:
            response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = json.loads(response.choices[0].message.content)
            items = raw if isinstance(raw, list) else raw.get("variations", [])
            return [{"user": i["question"], "assistant": i["answer"]} for i in items[:n]]
        except (SDKError, json.JSONDecodeError, KeyError) as exc:
            print(f"[generate_variations] error: {exc}")
            return []

    def generate_dataset(self, topics: list[str], n_per_topic: int = 2) -> list[dict]:
        """Generate *n_per_topic* Q&A pairs for each topic synchronously."""
        results: list[dict] = []
        for topic in topics:
            for _ in range(n_per_topic):
                pair = self.generate_qa_pair(topic)
                if pair:
                    results.append(pair)
                time.sleep(0.3)  # polite rate-limiting
        return results

    def deduplicate(self, pairs: list[dict], threshold: float = 0.92) -> list[dict]:
        """Remove near-duplicate pairs whose user-question embeddings exceed *threshold*."""
        if not pairs:
            return pairs
        questions = [p["user"] for p in pairs]
        try:
            emb_response = self.client.embeddings.create(
                model="mistral-embed", inputs=questions
            )
            embeddings = [e.embedding for e in emb_response.data]
        except SDKError as exc:
            print(f"[deduplicate] embedding error: {exc}; skipping dedup")
            return pairs

        kept: list[int] = []
        for i, emb_i in enumerate(embeddings):
            duplicate = any(
                _cosine_similarity(emb_i, embeddings[j]) >= threshold for j in kept
            )
            if not duplicate:
                kept.append(i)

        print(f"Deduplicated {len(pairs)} -> {len(kept)} pairs (threshold={threshold})")
        return [pairs[i] for i in kept]


# Demo: generate a small synthetic dataset
generator = SyntheticDataGenerator(client)
topics = ["transformer attention mechanism", "tokenisation in NLP"]
start = time.time()
synthetic_pairs = generator.generate_dataset(topics, n_per_topic=1)
elapsed = time.time() - start
print(f"Generated {len(synthetic_pairs)} pairs in {elapsed:.1f}s")
if synthetic_pairs:
    print("Sample pair:", json.dumps(synthetic_pairs[0], indent=2)[:300])

# %% [markdown]
# ## 4. Mistral Fine-Tuning API
# Mistral exposes a REST-backed fine-tuning workflow: upload a JSONL file, create a job with
# hyperparameters, then poll until the job status becomes `"success"`. The resulting model ID
# can be used like any other Mistral model.

# %%
TRAIN_PATH = r"d:\tmp\ft_train.jsonl"
VAL_PATH = r"d:\tmp\ft_val.jsonl"

# Build a slightly larger dataset by combining hand-written and synthetic pairs
all_pairs = sample_pairs + [(p["user"], p["assistant"]) for p in synthetic_pairs]
train_builder = DatasetBuilder()
val_builder = DatasetBuilder()

for idx, (u, a) in enumerate(all_pairs):
    if idx % 5 == 0:
        val_builder.add_pair(u, a)
    else:
        train_builder.add_pair(u, a)

train_builder.save(TRAIN_PATH)
val_builder.save(VAL_PATH)
print(f"Train: {train_builder.stats()}, Val: {val_builder.stats()}")


def upload_file(path: str, purpose: str = "fine-tune") -> str:
    """Upload *path* to the Mistral files API and return the file ID."""
    try:
        with open(path, "rb") as fh:
            resp = client.files.upload(file=fh, purpose=purpose)
        print(f"Uploaded {path} -> file_id={resp.id}")
        return resp.id
    except SDKError as exc:
        print(f"[upload_file] API error: {exc}")
        return ""


def create_fine_tuning_job(
    training_file_id: str,
    validation_file_id: str,
    base_model: str = "open-mistral-7b",
    epochs: int = 3,
    learning_rate: float = 1e-4,
) -> str:
    """Submit a fine-tuning job and return the job ID."""
    try:
        job = client.fine_tuning.jobs.create(
            training_files=[training_file_id],
            validation_files=[validation_file_id],
            model=base_model,
            hyperparameters={"training_steps": epochs * 10, "learning_rate": learning_rate},
        )
        print(f"Fine-tuning job created: {job.id} | status={job.status}")
        return job.id
    except SDKError as exc:
        print(f"[create_fine_tuning_job] API error: {exc}")
        return ""


def poll_job(job_id: str, interval: int = 30, max_wait: int = 3600) -> str:
    """Poll a fine-tuning job every *interval* seconds until done; return final model ID."""
    if not job_id:
        return ""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            job = client.fine_tuning.jobs.get(job_id)
            print(f"  status={job.status}")
            if job.status == "SUCCESS":
                print(f"Fine-tuned model ready: {job.fine_tuned_model}")
                return job.fine_tuned_model or ""
            if job.status in ("FAILED", "CANCELLED"):
                print("Job ended without a model.")
                return ""
        except SDKError as exc:
            print(f"[poll_job] error: {exc}")
        time.sleep(interval)
    print("Timed out waiting for job.")
    return ""


# NOTE: The upload and job-creation calls below will only succeed with valid credentials
# and sufficient quota. They are shown for instructional completeness.
print("\n-- Fine-tuning API walkthrough (requires valid API key + quota) --")
train_file_id = upload_file(TRAIN_PATH)
val_file_id = upload_file(VAL_PATH)

FINETUNED_MODEL_ID = ""  # populated after job completes
if train_file_id and val_file_id:
    job_id = create_fine_tuning_job(train_file_id, val_file_id)
    if job_id:
        print("Polling job (will timeout quickly in demo mode)...")
        FINETUNED_MODEL_ID = poll_job(job_id, interval=15, max_wait=60)

print(f"Fine-tuned model ID: {FINETUNED_MODEL_ID or '(not yet available)'}")

# %% [markdown]
# ## 5. Evaluating the Fine-Tuned Model
# An LLM-as-judge pattern uses `mistral-large-latest` to score each model response on accuracy,
# format, and helpfulness (1-5 per dimension). Aggregated scores across test cases reveal
# whether fine-tuning improved the model.

# %%
import matplotlib.pyplot as plt

JUDGE_SYSTEM = (
    "You are an expert evaluator. Score the assistant reply on three dimensions, "
    "each from 1 (very poor) to 5 (excellent):\n"
    "  accuracy   - factual correctness\n"
    "  format     - clarity and structure\n"
    "  helpfulness - practical usefulness\n"
    "Return ONLY valid JSON: {\"accuracy\": N, \"format\": N, \"helpfulness\": N}"
)


def judge_response(question: str, answer: str) -> dict:
    """Use mistral-large-latest as a judge and return a scores dict."""
    prompt = f"Question: {question}\n\nAssistant reply:\n{answer}"
    try:
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except (SDKError, json.JSONDecodeError) as exc:
        print(f"[judge_response] error: {exc}")
        return {"accuracy": 0, "format": 0, "helpfulness": 0}


def run_evaluation(model_id: str, test_cases: list[dict]) -> list[dict]:
    """Run *model_id* on each test case and score with the LLM judge.

    Each element of *test_cases* must have a ``user`` key.
    Returns a list of score dicts augmented with the model reply.
    """
    results = []
    for tc in test_cases:
        start = time.time()
        try:
            resp = client.chat.complete(
                model=model_id,
                messages=[{"role": "user", "content": tc["user"]}],
            )
            answer = resp.choices[0].message.content
        except SDKError as exc:
            print(f"[run_evaluation] model error: {exc}")
            answer = ""
        latency = time.time() - start
        scores = judge_response(tc["user"], answer)
        scores["latency"] = round(latency, 2)
        scores["answer_preview"] = answer[:80]
        results.append(scores)
    return results


def compare_models(base_scores: list[dict], finetuned_scores: list[dict]) -> None:
    """Print a comparison table and render a grouped bar chart."""
    dims = ["accuracy", "format", "helpfulness"]

    def avg(scores: list[dict], dim: str) -> float:
        vals = [s[dim] for s in scores if s.get(dim)]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    base_avgs = [avg(base_scores, d) for d in dims]
    ft_avgs = [avg(finetuned_scores, d) for d in dims]

    print(f"\n{'Dimension':<15} {'Base':>6} {'Fine-tuned':>12}")
    print("-" * 35)
    for d, b, f in zip(dims, base_avgs, ft_avgs):
        print(f"{d:<15} {b:>6.2f} {f:>12.2f}")

    x = np.arange(len(dims))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, base_avgs, width, label="Base model", color="steelblue")
    ax.bar(x + width / 2, ft_avgs, width, label="Fine-tuned", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Score (1-5)")
    ax.set_title("Base vs Fine-Tuned Model Evaluation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(r"d:\tmp\ft_comparison.png", dpi=100)
    plt.show()
    print("Chart saved to d:\\tmp\\ft_comparison.png")


# Demo evaluation against base model using the existing test pairs
test_cases = [{"user": q} for q, _ in sample_pairs]

print("Evaluating base model (mistral-small-latest) ...")
base_eval = run_evaluation("mistral-small-latest", test_cases)
print(f"Base evaluation complete: {len(base_eval)} results")

# For the fine-tuned model we fall back to mistral-small if no model was produced
ft_model = FINETUNED_MODEL_ID or "mistral-small-latest"
print(f"\nEvaluating model: {ft_model} ...")
ft_eval = run_evaluation(ft_model, test_cases)

compare_models(base_eval, ft_eval)

# %% [markdown]
# ## 6. LoRA Alternative with HuggingFace
# When you cannot (or prefer not to) use a managed fine-tuning API, Parameter-Efficient
# Fine-Tuning (PEFT) with LoRA/QLoRA lets you adapt a local model using a fraction of the
# memory by training only small rank-decomposition matrices instead of all weights.

# %%
# NOTE: The imports below require: pip install peft transformers trl bitsandbytes accelerate
# They are wrapped in try/except so the rest of the notebook continues without GPU hardware.

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    PEFT_AVAILABLE = True
except ImportError as _e:
    PEFT_AVAILABLE = False
    print(f"PEFT/transformers not available ({_e}); showing code only.")

LORA_MODEL_NAME = "mistralai/Mistral-7B-v0.1"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

bnb_config = None
if PEFT_AVAILABLE and torch.cuda.is_available():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_qlora_model(model_name: str) -> tuple[Any, Any] | tuple[None, None]:
    """Load a model and tokeniser with QLoRA quantisation if CUDA is available."""
    if not PEFT_AVAILABLE:
        print("PEFT not installed; skipping model load.")
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, tokenizer
    except Exception as exc:
        print(f"[load_qlora_model] could not load model: {exc}")
        return None, None


def build_sft_trainer(model: Any, tokenizer: Any, train_data: list[str]) -> Any:
    """Construct an SFTTrainer for a 10-step demo run."""
    from datasets import Dataset

    dataset = Dataset.from_dict({"text": train_data})
    training_args = SFTConfig(
        output_dir=r"d:\tmp\lora_checkpoints",
        max_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=2,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    return trainer


print("\n-- LoRA configuration summary --")
print(f"Rank (r): {lora_config.r}")
print(f"Alpha: {lora_config.lora_alpha}")
print(f"Target modules: {lora_config.target_modules}")
print(f"Dropout: {lora_config.lora_dropout}")
print(f"Task type: {lora_config.task_type}")
print("\nMemory comparison (approximate for Mistral-7B):")
print(f"  Full fine-tune (bf16): ~28 GB VRAM")
print(f"  QLoRA 4-bit + r=16   :  ~6 GB VRAM  (~79% reduction)")

if PEFT_AVAILABLE:
    print("\nAttempting QLoRA model load (requires GPU + ~6 GB VRAM)...")
    lora_model, lora_tokenizer = load_qlora_model(LORA_MODEL_NAME)
    if lora_model:
        train_texts = [f"User: {u}\nAssistant: {a}" for u, a in sample_pairs]
        trainer = build_sft_trainer(lora_model, lora_tokenizer, train_texts)
        print("SFTTrainer ready. Running 10 steps...")
        trainer.train()
        print("10-step demo training complete.")

# %% [markdown]
# ## 7. Lab Exercise
# **Goal**: generate 200 synthetic customer-support Q&A pairs, fine-tune a Mistral model,
# evaluate on 30 held-out test cases, and summarise the results in a metrics dict.
# This is a complete, self-contained challenge you can run end-to-end.

# %%
SUPPORT_TOPICS = [
    "order tracking", "return policy", "product warranty",
    "billing issue", "account login", "shipping delay",
    "refund request", "product damaged", "subscription cancellation",
    "discount code not working",
]


def run_lab_exercise(
    n_train: int = 200,
    n_test: int = 30,
    base_model: str = "open-mistral-7b",
) -> dict:
    """Full fine-tuning lab: generate data, fine-tune, evaluate, and return metrics.

    Steps:
    1. Generate *n_train* synthetic customer-support Q&A pairs.
    2. Split off *n_test* pairs as a held-out test set.
    3. Upload and fine-tune (if API key is valid).
    4. Evaluate base model and fine-tuned model on the test set with an LLM judge.
    5. Return a metrics report dict.
    """
    lab_start = time.time()
    print(f"[Lab] Generating {n_train} synthetic pairs...")
    gen = SyntheticDataGenerator(client)

    # Distribute pairs across topics
    pairs_per_topic = max(1, n_train // len(SUPPORT_TOPICS))
    raw_pairs = gen.generate_dataset(SUPPORT_TOPICS, n_per_topic=pairs_per_topic)

    # Top up to n_train if needed
    while len(raw_pairs) < n_train:
        extra = gen.generate_qa_pair(SUPPORT_TOPICS[len(raw_pairs) % len(SUPPORT_TOPICS)])
        if extra:
            raw_pairs.append(extra)

    raw_pairs = gen.deduplicate(raw_pairs)
    print(f"[Lab] After dedup: {len(raw_pairs)} pairs")

    # Split
    test_pairs = raw_pairs[:n_test]
    train_pairs = raw_pairs[n_test:]

    # Persist
    lab_train_path = r"d:\tmp\lab_train.jsonl"
    lab_val_path = r"d:\tmp\lab_val.jsonl"

    lab_builder = DatasetBuilder()
    for p in train_pairs:
        lab_builder.add_pair(p["user"], p["assistant"])
    lab_builder.save(lab_train_path)

    val_builder = DatasetBuilder()
    for p in test_pairs[:10]:
        val_builder.add_pair(p["user"], p["assistant"])
    val_builder.save(lab_val_path)

    # Fine-tune
    lab_ft_model = ""
    train_fid = upload_file(lab_train_path)
    val_fid = upload_file(lab_val_path)
    if train_fid and val_fid:
        jid = create_fine_tuning_job(train_fid, val_fid, base_model=base_model, epochs=3)
        if jid:
            lab_ft_model = poll_job(jid, interval=30, max_wait=7200)

    # Evaluate
    lab_test_cases = [{"user": p["user"]} for p in test_pairs]
    print("[Lab] Evaluating base model...")
    lab_base_scores = run_evaluation("mistral-small-latest", lab_test_cases[:10])

    eval_model = lab_ft_model or "mistral-small-latest"
    print(f"[Lab] Evaluating {eval_model}...")
    lab_ft_scores = run_evaluation(eval_model, lab_test_cases[:10])

    dims = ["accuracy", "format", "helpfulness"]

    def _avg(scores: list[dict], d: str) -> float:
        v = [s[d] for s in scores if s.get(d)]
        return round(sum(v) / len(v), 2) if v else 0.0

    metrics = {
        "total_pairs_generated": len(raw_pairs),
        "training_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "fine_tuned_model_id": lab_ft_model or "not_created",
        "base_model_scores": {d: _avg(lab_base_scores, d) for d in dims},
        "finetuned_model_scores": {d: _avg(lab_ft_scores, d) for d in dims},
        "improvement": {
            d: round(_avg(lab_ft_scores, d) - _avg(lab_base_scores, d), 2)
            for d in dims
        },
        "elapsed_seconds": round(time.time() - lab_start, 1),
    }
    return metrics


# Execute the lab (use small counts to stay within free-tier limits during class)
lab_metrics = run_lab_exercise(n_train=20, n_test=5)
print("\n=== Lab Exercise Report ===")
print(json.dumps(lab_metrics, indent=2))

assert "total_pairs_generated" in lab_metrics
assert "base_model_scores" in lab_metrics
assert "finetuned_model_scores" in lab_metrics
print("\nAll assertions passed.")

# %% [markdown]
# ## Key Takeaways
# - Fine-tuning requires high-quality, diverse data — the `DatasetBuilder` class enforces minimum
#   length and format constraints before a single example reaches the model.
# - Synthetic data generation with a powerful teacher model (`mistral-large-latest`) is a
#   practical way to bootstrap training data when human-labelled examples are scarce.
# - Embedding-based deduplication prevents near-duplicate examples from skewing training and
#   inflating perceived dataset size.
# - LoRA/QLoRA dramatically reduces memory requirements (up to ~79% for Mistral-7B) by training
#   only low-rank adapter matrices, making local fine-tuning feasible on consumer hardware.
# - Systematic evaluation with an LLM-as-judge closes the loop: without measurable improvement
#   on held-out test cases, fine-tuning may not be worth the cost.
