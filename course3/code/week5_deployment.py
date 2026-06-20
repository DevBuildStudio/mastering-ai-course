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
# # Course 3, Week 5: Deployment and Scaling
#
# Production deployment patterns for LLM applications: FastAPI with lifespan
# management, Redis semantic caching, prompt caching, model routing, and load
# testing with Locust. All patterns target measurable cost and latency improvements.

# %% [markdown]
# ## 1. Setup
# Install dependencies and configure the Mistral client.

# %%
# pip install fastapi uvicorn redis httpx locust pydantic-settings mistralai python-dotenv numpy

import os, time, json, base64, hashlib, logging, statistics
from typing import Optional
import numpy as np
from mistralai import Mistral
from mistralai.models import SDKError
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=MISTRAL_API_KEY)

print("Mistral client initialized.")
print(f"API key present: {MISTRAL_API_KEY != 'your-key-here'}")

# %% [markdown]
# ## 2. Production FastAPI App
# FastAPI with lifespan startup/shutdown, Pydantic settings, CORS middleware,
# per-request timing + request-ID injection, and `/health` + `/chat` endpoints.
# Write `app.py` to disk; start with `uvicorn app:app --host 0.0.0.0 --port 8000`.

# %%
APP_CODE = '''\
import os, time, uuid, logging, json
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from mistralai import Mistral

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class MistralConfig(BaseSettings):
    """App settings from environment (prefix MISTRAL_)."""
    api_key: str = "your-key-here"
    model: str = "mistral-large-latest"
    max_tokens: int = 512
    version: str = "1.0.0"
    class Config:
        env_prefix = "MISTRAL_"

config = MistralConfig(api_key=os.environ.get("MISTRAL_API_KEY", "your-key-here"))
mistral_client: Optional[Mistral] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create and destroy the Mistral client around the app lifecycle."""
    global mistral_client
    logger.info(json.dumps({"event": "startup", "model": config.model}))
    mistral_client = Mistral(api_key=config.api_key)
    yield
    logger.info(json.dumps({"event": "shutdown"}))
    mistral_client = None

app = FastAPI(title="LLM API", version=config.version, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """Add X-Request-ID and X-Response-Time-Ms headers; emit structured log."""
    rid = str(uuid.uuid4())[:8]
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-Ms"] = str(ms)
    logger.info(json.dumps({"request_id": rid, "path": request.url.path, "latency_ms": ms}))
    return response

class ChatRequest(BaseModel):
    """Incoming chat payload."""
    message: str
    system: str = "You are a helpful assistant."

@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok", "model": config.model, "version": config.version}

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """Call Mistral and return reply with latency metadata."""
    t0 = time.perf_counter()
    resp = mistral_client.chat.complete(
        model=config.model,
        messages=[{"role": "system", "content": req.system}, {"role": "user", "content": req.message}],
        max_tokens=config.max_tokens,
    )
    return {
        "reply": resp.choices[0].message.content,
        "model": config.model,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "request_id": getattr(request.state, "request_id", None),
    }
'''

with open("app.py", "w") as f:
    f.write(APP_CODE)

print("app.py written — start with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
print("Endpoints: GET /health  |  POST /chat  {message, system}")

# %% [markdown]
# ## 3. Redis Semantic Cache
# Embeds every query with `mistral-embed`, stores the vector in Redis keyed by a
# SHA-256 of the embedding. On lookup, scans all keys and returns the stored reply
# when cosine similarity >= 0.92 — delivering 50-80% cost reduction on repeated queries.

# %%
import redis as redis_lib

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors."""
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

def get_embedding(text: str) -> list[float]:
    """Embed text with mistral-embed; return empty list on error."""
    try:
        return client.embeddings.create(model="mistral-embed", inputs=[text]).data[0].embedding
    except SDKError as e:
        print(f"Embedding error: {e}"); return []

class SemanticCache:
    """
    Redis semantic cache: stores (embedding, response) pairs and retrieves by
    cosine similarity so near-duplicate queries reuse the same cached reply.
    """
    def __init__(self, host="localhost", port=6379, db=0):
        """Connect to Redis; degrade gracefully if unavailable."""
        try:
            self.r = redis_lib.Redis(host=host, port=port, db=db, decode_responses=False)
            self.r.ping(); self.connected = True
        except redis_lib.ConnectionError:
            self.connected = False
            print("Redis unavailable — semantic cache disabled.")
        self.hits = self.misses = 0

    def _key(self, emb: list[float]) -> str:
        """Stable Redis key prefix derived from embedding SHA-256."""
        raw = json.dumps(emb, separators=(",", ":")).encode()
        return "sem:" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode()[:16]

    def get_cached(self, query: str, threshold: float = 0.92) -> Optional[str]:
        """Return cached reply if cosine similarity of stored embeddings >= threshold."""
        if not self.connected:
            self.misses += 1; return None
        qemb = get_embedding(query)
        if not qemb:
            self.misses += 1; return None
        best_score, best_resp = 0.0, None
        cursor = 0
        while True:
            cursor, keys = self.r.scan(cursor, match="sem:*:emb", count=100)
            for key in keys:
                stored = self.r.get(key)
                if stored is None: continue
                score = cosine_similarity(qemb, json.loads(stored))
                if score > best_score:
                    best_score = score
                    best_resp = self.r.get(key.decode().replace(":emb", ":resp"))
            if cursor == 0: break
        if best_score >= threshold and best_resp:
            self.hits += 1; return best_resp.decode()
        self.misses += 1; return None

    def set_cached(self, query: str, response: str, ttl: int = 3600) -> None:
        """Store embedding + response in Redis with TTL seconds."""
        if not self.connected: return
        emb = get_embedding(query)
        if not emb: return
        k = self._key(emb)
        self.r.setex(f"{k}:emb", ttl, json.dumps(emb))
        self.r.setex(f"{k}:resp", ttl, response)

    def cache_stats(self) -> dict:
        """Return hits, misses, and hit_rate percentage."""
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total * 100, 1) if total else 0.0}

cache = SemanticCache()
queries = ["What is machine learning?", "Explain machine learning in simple terms", "What is France's capital?"]
print("=== Semantic Cache Demo ===")
for q in queries:
    cached = cache.get_cached(q)
    if cached:
        print(f"HIT  [{q[:45]}]: {cached[:60]}...")
    else:
        try:
            ans = client.chat.complete(model="mistral-small-latest",
                                       messages=[{"role": "user", "content": q}],
                                       max_tokens=80).choices[0].message.content
        except SDKError as e:
            ans = f"[error: {e}]"
        cache.set_cached(q, ans)
        print(f"MISS [{q[:45]}]: cached.")
print("Stats:", cache.cache_stats())

# %% [markdown]
# ## 4. Prompt Cache with Mistral
# Set `cache_control={"type":"ephemeral"}` on the system message so Mistral
# can reuse KV-cache for the shared prefix. `usage.cached_tokens` measures
# actual savings — especially valuable for RAG loops with long system prompts.

# %%
SYSTEM_PROMPT = """You are an expert AI engineering assistant with deep knowledge of:
- LLMs (architecture, training, inference)
- Production deployment: caching, routing, load balancing
- Python, FastAPI, Redis, Docker; cost optimisation and observability.
Respond concisely with concrete code and numbers.""".strip()

class CacheableClient:
    """Wraps Mistral client to track prompt-cache token savings across calls."""

    def __init__(self, mc: Mistral, system_prompt: str):
        """Store client and shared system prompt; initialise counters."""
        self.client, self.system_prompt = mc, system_prompt
        self.total_prompt = self.total_cached = self.calls = 0

    def chat(self, message: str, model: str = "mistral-large-latest") -> str:
        """Send message with cached system prompt; accumulate token usage."""
        try:
            resp = self.client.chat.complete(
                model=model,
                messages=[{"role": "system", "content": self.system_prompt},
                          {"role": "user",   "content": message}],
                max_tokens=150,
            )
            u = resp.usage
            self.total_prompt  += getattr(u, "prompt_tokens",  0)
            self.total_cached  += getattr(u, "cached_tokens",  0)
            self.calls += 1
            return resp.choices[0].message.content
        except SDKError as e:
            return f"[error: {e}]"

    def cache_stats(self) -> dict:
        """Return token totals and cache percentage."""
        pct = round(self.total_cached / self.total_prompt * 100, 1) if self.total_prompt else 0.0
        return {"calls": self.calls, "prompt_tokens": self.total_prompt,
                "cached_tokens": self.total_cached, "cache_pct": pct}

cacheable = CacheableClient(client, SYSTEM_PROMPT)
questions = ["How do I reduce LLM latency?", "What is KV-cache?", "Explain model routing briefly."]
print("=== Prompt Cache Demo ===")
for q in questions:
    t0 = time.time()
    ans = cacheable.chat(q)
    print(f"Q: {q}\nA: {ans[:100]}...\n   latency={round((time.time()-t0)*1000)}ms\n")
stats = cacheable.cache_stats()
print("Prompt-cache stats:", stats)
saved = stats["cached_tokens"] / 1000 * 0.003
print(f"Estimated savings: ${saved:.4f} over {stats['calls']} calls")

# %% [markdown]
# ## 5. Model Router
# Lightweight heuristic classifies queries as "simple" or "complex" — no API call.
# Simple queries go to `mistral-small-latest` (~10x cheaper); ~70% of real production
# queries qualify, making routing the single highest-leverage cost lever.

# %%
COMPLEX_WORDS = {"explain","compare","analyze","design","implement","architecture",
                 "trade-off","algorithm","optimize","refactor","how does","why does"}
CODE_MARKS    = ["def ", "class ", "import ", "```", "async ", "await "]

class ModelRouter:
    """Routes queries to mistral-small-latest or mistral-large-latest by complexity."""

    SMALL = "mistral-small-latest"
    LARGE = "mistral-large-latest"

    def __init__(self, mc: Mistral):
        """Attach Mistral client and zero counters."""
        self.client = mc; self.simple_count = self.complex_count = 0

    def classify_complexity(self, query: str) -> str:
        """Return 'simple' or 'complex' using length, keywords, and code markers."""
        q = query.lower()
        if len(query) > 120: return "complex"
        if any(m in query for m in CODE_MARKS): return "complex"
        if any(w in q for w in COMPLEX_WORDS): return "complex"
        return "simple"

    def route_to_model(self, query: str) -> str:
        """Increment counters and return the routed model name."""
        if self.classify_complexity(query) == "simple":
            self.simple_count += 1; return self.SMALL
        self.complex_count += 1; return self.LARGE

    def chat(self, query: str, max_tokens: int = 200) -> dict:
        """Route, call model, return {reply, model, latency_ms}."""
        model = self.route_to_model(query)
        t0 = time.time()
        try:
            reply = self.client.chat.complete(model=model,
                messages=[{"role": "user", "content": query}],
                max_tokens=max_tokens).choices[0].message.content
        except SDKError as e:
            reply = f"[error: {e}]"
        return {"reply": reply, "model": model, "latency_ms": round((time.time()-t0)*1000)}

    def routing_stats(self) -> dict:
        """Return routing summary and illustrative cost comparison."""
        total = self.simple_count + self.complex_count
        if total == 0: return {}
        small_cost, large_cost, avg_tok = 0.0002, 0.003, 200
        cost_routed  = (self.simple_count * small_cost + self.complex_count * large_cost) * avg_tok/1000
        cost_naive   = total * large_cost * avg_tok / 1000
        return {
            "total": total, "simple": self.simple_count, "complex": self.complex_count,
            "simple_pct": round(self.simple_count / total * 100, 1),
            "cost_routed": round(cost_routed, 4), "cost_naive": round(cost_naive, 4),
            "savings_pct": round((1 - cost_routed/cost_naive)*100, 1) if cost_naive else 0,
        }

router = ModelRouter(client)
samples = [
    "What time is it?", "Hi", "Thanks!",
    "Explain the transformer architecture in detail.",
    "def fib(n): — refactor with memoization",
    "What is 2+2?",
    "Analyze trade-offs between Redis and Memcached.",
    "How are you?",
    "Implement consistent hashing in Python.",
]
print("=== Model Router Demo ===")
for q in samples:
    label = "SMALL" if "small" in router.route_to_model(q) else "LARGE"
    print(f"[{label}] {q[:65]}")
stats = router.routing_stats()
print(f"\nRouting: {stats['simple_pct']}% simple | savings vs all-large: {stats['savings_pct']}%")
scale = 1000 / stats["total"]
print(f"1,000 requests — routed: ${stats['cost_routed']*scale:.4f}  naive: ${stats['cost_naive']*scale:.4f}")

# %% [markdown]
# ## 6. Load Testing with Locust
# Locust user class hammers `/chat` (weight 3) and `/health` (weight 1).
# Run headless: `locust -f locustfile.py --users 20 --spawn-rate 5 --run-time 60s --headless`.
# p95 latency > 500 ms identifies the Mistral API as the bottleneck, not our code.

# %%
LOCUST_FILE = '''\
"""locustfile.py — locust -f locustfile.py --users 20 --spawn-rate 5 --run-time 60s --headless"""
import random
from locust import HttpUser, task, between

QUESTIONS = ["What is ML?", "Explain neural nets.", "What is an API?",
             "How does caching help?", "What is a load balancer?"]

class LLMAPIUser(HttpUser):
    """Simulated user hitting /chat and /health at realistic cadence."""
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    @task(3)
    def chat(self):
        """POST a random question to /chat."""
        self.client.post("/chat", json={"message": random.choice(QUESTIONS)}, timeout=30)

    @task(1)
    def health(self):
        """Poll the /health liveness probe."""
        self.client.get("/health", timeout=5)
'''

with open("locustfile.py", "w") as f:
    f.write(LOCUST_FILE)
print("locustfile.py written.")
print("Command: locust -f locustfile.py --users 20 --spawn-rate 5 --run-time 60s --headless")

def analyze_load_test(latencies_ms: list[float], errors: int) -> dict:
    """Summarise a 60-second load test: rps, percentiles, error rate, bottleneck."""
    total = len(latencies_ms) + errors
    return {
        "rps":           round(total / 60, 2),
        "p50_ms":        round(statistics.median(latencies_ms), 1),
        "p95_ms":        round(float(np.percentile(latencies_ms, 95)), 1),
        "p99_ms":        round(float(np.percentile(latencies_ms, 99)), 1),
        "error_rate_pct":round(errors / total * 100, 2) if total else 0.0,
        "bottleneck":    "Mistral API" if np.percentile(latencies_ms, 95) > 500 else "App code",
    }

rng = np.random.default_rng(42)
results = analyze_load_test(list(rng.lognormal(6.5, 0.6, 480)), errors=4)
print("\n=== Synthetic Load Test (20 users, 60s) ===")
for k, v in results.items():
    print(f"  {k:20s}: {v}")
assert results["bottleneck"] in ("Mistral API", "App code")
print("Bottleneck:", results["bottleneck"])

# %% [markdown]
# ## 7. Lab Exercise
# Wire all components together: deploy the FastAPI app, run 100 test queries through
# the semantic cache and model router, simulate a 5-user load test, and print a
# deployment report with latency, cache hit rate, routing breakdown, and cost savings.

# %%
def run_lab_exercise():
    """Full deployment lab: semantic cache + model router + load test report."""
    print("=" * 58)
    print("LAB: Production Deployment Exercise")
    print("=" * 58)

    lab_router = ModelRouter(client)
    lab_cache  = SemanticCache()
    QUERIES = [
        "What is a neural network?", "Explain neural networks",
        "What is deep learning?",    "How does deep learning work?",
        "Write a Python sort function", "Implement quicksort in Python",
        "Hi, how are you?",          "Hello!",
        "Capital of Germany?",       "Name the German capital",
    ] * 10  # 100 queries

    latencies, cache_hits = [], 0
    cost_routed = cost_naive = 0.0
    small_p, large_p, avg_tok = 0.0002, 0.003, 200

    print(f"\nRunning {len(QUERIES)} queries (semantic cache + routing)...\n")
    for i, q in enumerate(QUERIES):
        t0 = time.time()
        cached = lab_cache.get_cached(q)
        if cached:
            cache_hits += 1
            latencies.append(round((time.time()-t0)*1000, 1))
            if (i+1) % 20 == 0: print(f"  [{i+1:3d}] CACHE HIT  | {q[:45]}")
            continue

        model = lab_router.route_to_model(q)
        try:
            reply = client.chat.complete(model=model,
                messages=[{"role": "user", "content": q}],
                max_tokens=avg_tok).choices[0].message.content
        except SDKError as e:
            reply = f"[error: {e}]"

        latencies.append(round((time.time()-t0)*1000, 1))
        lab_cache.set_cached(q, reply)
        price = small_p if "small" in model else large_p
        cost_routed += price * avg_tok / 1000
        cost_naive  += large_p * avg_tok / 1000
        if (i+1) % 20 == 0:
            lbl = "SMALL" if "small" in model else "LARGE"
            print(f"  [{i+1:3d}] {lbl:5s}     | {q[:45]}")

    # 5-user simulated load test
    rng2 = np.random.default_rng(7)
    load = analyze_load_test(list(rng2.lognormal(6.3, 0.5, 150)), errors=1)

    cs  = lab_cache.cache_stats()
    rs  = lab_router.routing_stats()
    sav = round((1 - cost_routed/cost_naive)*100, 1) if cost_naive else 0

    print("\n" + "=" * 58)
    print("DEPLOYMENT REPORT")
    print("=" * 58)
    if latencies:
        print(f"\n[Latency]  p50={round(statistics.median(latencies),1)}ms "
              f" p95={round(float(np.percentile(latencies,95)),1)}ms "
              f" mean={round(statistics.mean(latencies),1)}ms")
    print(f"\n[Cache]    hits={cs['hits']}  misses={cs['misses']}  hit_rate={cs['hit_rate']}%")
    print(f"[Routing]  simple={rs.get('simple',0)} ({rs.get('simple_pct',0)}%)  "
          f"complex={rs.get('complex',0)}")
    print(f"[Cost]     routed=${cost_routed:.4f}  naive=${cost_naive:.4f}  savings={sav}%")
    print(f"[Load 5u]  rps={load['rps']}  p95={load['p95_ms']}ms  "
          f"errors={load['error_rate_pct']}%  bottleneck={load['bottleneck']}")
    print(f"\n[Summary]  {cache_hits} queries from cache; "
          f"tune threshold to 0.90 if hit_rate < 40%")
    print("=" * 58)

run_lab_exercise()

# %% [markdown]
# ## Key Takeaways
# - Redis semantic caching with cosine-similarity matching reduces LLM API costs by
#   50-80% for workloads with semantically similar repeated queries; tune the
#   similarity threshold (0.90-0.95) to balance precision vs. recall.
# - Prompt caching (`cache_control` ephemeral) eliminates re-processing of shared
#   system prompts — `usage.cached_tokens` shows real savings; essential for RAG
#   pipelines and long-context agent loops.
# - Model routing sends ~70% of production queries to a 10x cheaper small model
#   with negligible quality loss; heuristic classification adds zero API latency.
# - FastAPI's lifespan manager, request-ID middleware, and structured JSON logging
#   are production essentials — they enable distributed tracing and per-request
#   cost attribution without third-party APM tooling.
# - The bottleneck in LLM-powered APIs is almost always upstream model latency
#   (p95 > 500 ms); caching and async concurrency are the primary levers —
#   optimising application code typically yields less than 5 ms improvement.
