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
# # Week 4: Embeddings and Vector Databases
#
# This notebook covers how to generate text embeddings with Mistral, measure semantic
# similarity using cosine distance, and store/query vectors using ChromaDB.
# By the end you will have a working semantic search engine over a small document corpus.

# %% [markdown]
# ## 1. Setup
# Install dependencies and initialise the Mistral client. ChromaDB is used as the
# vector store — it runs fully in-process with no external server required.
# The `mistral-embed` model produces 1024-dimensional dense vectors.

# %%
# pip install mistralai python-dotenv chromadb numpy
import os
import time
import math

import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models import SDKError

load_dotenv()

API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=API_KEY)
EMBED_MODEL = "mistral-embed"

print("Mistral client ready.")
print(f"Embedding model : {EMBED_MODEL}")

# %% [markdown]
# ## 2. Generating Embeddings
# `client.embeddings.create` accepts a list of strings and returns a list of
# 1024-dimensional float vectors. We time both single and batch calls so you can
# appreciate the efficiency of batching.

# %%
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using the Mistral embedding model.

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        List of 1024-dimensional float vectors, one per input text.

    Raises:
        SDKError: Propagated from the Mistral API on failure.
    """
    try:
        response = client.embeddings.create(model=EMBED_MODEL, inputs=texts)
        return [item.embedding for item in response.data]
    except SDKError as exc:
        print(f"[ERROR] Embedding API call failed: {exc}")
        raise


# Single embedding
sample_text = "Artificial intelligence is transforming the software industry."
t0 = time.time()
single_vec = embed_texts([sample_text])[0]
elapsed_single = time.time() - t0

print(f"Embedding dimension : {len(single_vec)}")
assert len(single_vec) == 1024, "Expected 1024-dim embedding"
print(f"First 5 values      : {single_vec[:5]}")
print(f"Single embed time   : {elapsed_single:.3f}s")

# Batch embedding of 20 texts
batch_texts = [
    f"Document number {i}: machine learning is used in many applications." for i in range(20)
]
t0 = time.time()
batch_vecs = embed_texts(batch_texts)
elapsed_batch = time.time() - t0

assert len(batch_vecs) == 20
assert all(len(v) == 1024 for v in batch_vecs)
print(f"\nBatch (20 texts) embed time : {elapsed_batch:.3f}s")
print(f"Avg per text                : {elapsed_batch/20:.4f}s")

# %% [markdown]
# ## 3. Cosine Similarity
# Cosine similarity measures the angle between two vectors — 1.0 means identical
# direction, 0.0 means orthogonal. We build a full similarity matrix and a helper
# that finds the closest match for any query string.

# %%
def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        v1: First vector as a list of floats.
        v2: Second vector as a list of floats (same length as v1).

    Returns:
        Cosine similarity score in [-1, 1].
    """
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / norm if norm > 0.0 else 0.0


def compute_similarity_matrix(texts: list[str]) -> np.ndarray:
    """Compute an N x N pairwise cosine similarity matrix for a list of texts.

    Args:
        texts: List of strings to compare.

    Returns:
        NumPy array of shape (N, N) with similarity scores.
    """
    vecs = embed_texts(texts)
    n = len(vecs)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_similarity(vecs[i], vecs[j])
    return matrix


def find_most_similar(query: str, corpus: list[str]) -> tuple[str, float]:
    """Find the most similar text in corpus to the query.

    Args:
        query:  The query string.
        corpus: List of candidate strings.

    Returns:
        Tuple of (best_matching_text, similarity_score).
    """
    all_texts = [query] + corpus
    vecs = embed_texts(all_texts)
    query_vec = vecs[0]
    corpus_vecs = vecs[1:]
    scores = [cosine_similarity(query_vec, cv) for cv in corpus_vecs]
    best_idx = int(np.argmax(scores))
    return corpus[best_idx], scores[best_idx]


# Demonstrate that similar sentences cluster together
sentences = [
    "Dogs are loyal and friendly pets.",
    "Cats are independent domestic animals.",
    "Python is a popular programming language.",
    "JavaScript runs in the browser.",
    "My dog loves to play fetch in the park.",
]

sim_matrix = compute_similarity_matrix(sentences)
print("Cosine similarity matrix:")
header = "".join(f"  S{i+1}  " for i in range(len(sentences)))
print("     " + header)
for i, row in enumerate(sim_matrix):
    row_str = "".join(f"{v:6.3f}" for v in row)
    print(f"  S{i+1} {row_str}")

print()
query = "I have a golden retriever who loves outdoor activities."
match, score = find_most_similar(query, sentences)
print(f"Query  : {query}")
print(f"Match  : {match}")
print(f"Score  : {score:.4f}")
assert score > 0.7, "Expected high similarity for semantically close sentences"

# %% [markdown]
# ## 4. ChromaDB Setup
# ChromaDB is an open-source vector database that can run in-memory (for experiments)
# or persist to disk (for production). We define a custom embedding function so
# ChromaDB calls Mistral automatically when you add or query documents.

# %%
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


class MistralEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding function backed by the Mistral embed model."""

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a batch of documents.

        Args:
            input: List of document strings from ChromaDB.

        Returns:
            List of 1024-dim float vectors.
        """
        return embed_texts(list(input))


mistral_ef = MistralEmbeddingFunction()

# In-memory client (no persistence)
mem_client = chromadb.Client()
mem_col = mem_client.create_collection(
    name="docs_memory",
    embedding_function=mistral_ef,
)

# Persistent client (survives process restarts)
PERSIST_DIR = os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "chroma_db")
disk_client = chromadb.PersistentClient(path=PERSIST_DIR)
# Delete if exists so the notebook is re-runnable
try:
    disk_client.delete_collection("docs_disk")
except Exception:
    pass
disk_col = disk_client.create_collection(
    name="docs_disk",
    embedding_function=mistral_ef,
)

# Add a few documents with metadata
sample_docs = [
    "Vector databases store high-dimensional embeddings for fast retrieval.",
    "ChromaDB is a lightweight, open-source vector store written in Python.",
    "Semantic search finds documents by meaning rather than keyword overlap.",
]
sample_ids = [f"doc_{i}" for i in range(len(sample_docs))]
sample_meta = [{"source": "intro", "index": i} for i in range(len(sample_docs))]

disk_col.add(documents=sample_docs, ids=sample_ids, metadatas=sample_meta)
print(f"Collection count: {disk_col.count()}")
assert disk_col.count() == 3

# %% [markdown]
# ## 5. Chunking Documents
# Long documents must be split into overlapping chunks before embedding so that
# every chunk fits comfortably within the model's context window and no information
# is lost at chunk boundaries. We also store per-chunk metadata for later filtering.

# %%
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping character-level chunks.

    Args:
        text:       Source text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        List of chunk strings. Returns a single-element list if the text is
        shorter than chunk_size.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def process_file(filepath: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
    """Read a text file and return a list of chunk dicts with metadata.

    Args:
        filepath:   Absolute path to a UTF-8 text file.
        chunk_size: Characters per chunk (passed to chunk_text).
        overlap:    Overlap characters between chunks.

    Returns:
        List of dicts with keys: 'text', 'source', 'chunk_index', 'total_chunks'.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        raw = fh.read()
    chunks = chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
    filename = os.path.basename(filepath)
    return [
        {
            "text": chunk,
            "source": filename,
            "chunk_index": idx,
            "total_chunks": len(chunks),
        }
        for idx, chunk in enumerate(chunks)
    ]


def embed_and_store(chunks: list[dict], collection: chromadb.Collection) -> None:
    """Embed a list of chunk dicts and upsert them into a ChromaDB collection.

    Args:
        chunks:     Output of process_file — each dict must have key 'text'.
        collection: Target ChromaDB collection (must use MistralEmbeddingFunction).
    """
    texts = [c["text"] for c in chunks]
    ids = [f"{c['source']}_chunk_{c['chunk_index']}" for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    # Add in batches of 10 to avoid rate limits
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        collection.add(
            documents=texts[i : i + batch_size],
            ids=ids[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


# Quick smoke-test with an in-memory string
demo_text = " ".join([f"Sentence number {n} about machine learning." for n in range(30)])
demo_chunks = chunk_text(demo_text, chunk_size=200, overlap=30)
print(f"Total chunks (size=200, overlap=30): {len(demo_chunks)}")
assert len(demo_chunks) > 1
print(f"Chunk 0 length : {len(demo_chunks[0])} chars")
print(f"Chunk 1 prefix : {demo_chunks[1][:40]}...")

# %% [markdown]
# ## 6. Semantic Search
# The `SemanticSearch` class wraps ChromaDB with a clean `index` / `search` API.
# Metadata filters let you narrow results to a specific source or topic, which is
# essential when your collection spans multiple document types.

# %%
class SemanticSearch:
    """High-level semantic search backed by ChromaDB and Mistral embeddings.

    Usage::

        ss = SemanticSearch(collection_name="my_docs")
        ss.index(documents, metadatas=metadatas)
        results = ss.search("what is gradient descent?", n_results=3)
    """

    def __init__(self, collection_name: str = "semantic_search"):
        """Initialise an in-memory ChromaDB collection.

        Args:
            collection_name: Name for the ChromaDB collection.
        """
        self._client = chromadb.Client()
        self._ef = MistralEmbeddingFunction()
        self._collection = self._client.create_collection(
            name=collection_name,
            embedding_function=self._ef,
        )

    def index(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """Add documents to the collection.

        Args:
            documents: List of raw text strings.
            metadatas: Optional list of metadata dicts (one per document).
        """
        ids = [f"doc_{i}" for i in range(self._collection.count(), self._collection.count() + len(documents))]
        kwargs: dict = {"documents": documents, "ids": ids}
        if metadatas:
            kwargs["metadatas"] = metadatas
        self._collection.add(**kwargs)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Query the collection for semantically similar documents.

        Args:
            query:     Natural-language query string.
            n_results: Maximum number of results to return.
            where:     Optional ChromaDB metadata filter, e.g. {"topic": "python"}.

        Returns:
            List of dicts with keys: 'text', 'metadata', 'distance', 'score'.
        """
        kwargs: dict = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "score": 1.0 - dist,  # ChromaDB returns L2 distance; invert for readability
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ]


# Build a small collection and demonstrate filtering
ss = SemanticSearch(collection_name="demo")
corpus_docs = [
    "Python lists support append, pop, and slice operations.",
    "NumPy arrays enable fast vectorised arithmetic in Python.",
    "JavaScript closures capture variables from the enclosing scope.",
    "Arrow functions in JavaScript provide concise syntax.",
    "Gradient descent minimises loss by following the negative gradient.",
    "Backpropagation computes gradients using the chain rule.",
]
corpus_meta = [
    {"topic": "python"}, {"topic": "python"},
    {"topic": "javascript"}, {"topic": "javascript"},
    {"topic": "ml"}, {"topic": "ml"},
]
ss.index(corpus_docs, metadatas=corpus_meta)

query_str = "how do I work with arrays in Python?"

print("=== Without filter ===")
results_all = ss.search(query_str, n_results=3)
for r in results_all:
    print(f"  [{r['metadata']['topic']:12s}] score={r['score']:.4f}  {r['text'][:60]}")

print("\n=== Filtered to topic=python ===")
results_py = ss.search(query_str, n_results=3, where={"topic": "python"})
for r in results_py:
    print(f"  [{r['metadata']['topic']:12s}] score={r['score']:.4f}  {r['text'][:60]}")

assert all(r["metadata"]["topic"] == "python" for r in results_py), "Filter not applied"

# %% [markdown]
# ## 7. Lab Exercise
# Build a complete semantic search system over 50 synthetic documents drawn from
# three topics (Python, Machine Learning, Web Development). You will:
# 1. Index all documents with metadata.
# 2. Search with and without topic filters.
# 3. Run a minimal CLI-style REPL.
# 4. Compare retrieval quality across chunk sizes (128 / 512 / 1024).

# %%
# --- Dataset generation (no API calls needed here) ---
TOPIC_DOCS: dict[str, list[str]] = {
    "python": [
        "Python uses indentation to define code blocks instead of braces.",
        "List comprehensions provide a concise way to build lists: [x*2 for x in range(10)].",
        "The `with` statement ensures resources like files are closed automatically.",
        "Decorators in Python wrap functions to add behaviour without modifying them.",
        "Generator functions use `yield` to produce values lazily.",
        "Python's GIL limits true parallelism; use multiprocessing for CPU-bound tasks.",
        "Type hints (PEP 484) improve IDE support and static analysis.",
        "Dataclasses reduce boilerplate when creating simple data-holding classes.",
        "The `pathlib` module provides object-oriented filesystem paths.",
        "Virtual environments isolate project dependencies from the system Python.",
        "f-strings (PEP 498) allow inline expressions inside string literals.",
        "The `itertools` module provides efficient looping utilities.",
        "Context managers can be created with `contextlib.contextmanager`.",
        "Python dicts preserve insertion order since version 3.7.",
        "Walrus operator `:=` assigns a value as part of an expression.",
        "asyncio enables concurrent I/O-bound tasks with async/await syntax.",
    ],
    "machine_learning": [
        "Supervised learning trains a model on labelled input-output pairs.",
        "Overfitting occurs when a model memorises training data but fails to generalise.",
        "Regularisation techniques like L1 and L2 penalise large model weights.",
        "Cross-validation estimates model performance on unseen data.",
        "Decision trees split data recursively based on feature thresholds.",
        "Random forests aggregate many decision trees to reduce variance.",
        "Neural networks learn hierarchical representations through stacked layers.",
        "Batch normalisation stabilises training by normalising layer activations.",
        "Dropout randomly deactivates neurons during training to prevent co-adaptation.",
        "Transfer learning adapts a pre-trained model to a new task with less data.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "The transformer architecture replaced RNNs for most NLP tasks.",
        "Embeddings map discrete tokens to continuous vector spaces.",
        "Cosine similarity measures the angle between two embedding vectors.",
        "Fine-tuning updates pre-trained weights on a domain-specific dataset.",
        "RLHF aligns language models with human preferences through reward modelling.",
        "RAG (Retrieval-Augmented Generation) grounds LLM responses with retrieved facts.",
    ],
    "web_development": [
        "REST APIs use HTTP verbs (GET, POST, PUT, DELETE) to manipulate resources.",
        "JSON is the de-facto data exchange format for modern web APIs.",
        "OAuth 2.0 delegates authorisation without sharing credentials.",
        "WebSockets enable full-duplex communication between client and server.",
        "CORS headers control which origins can call a cross-origin API.",
        "HTTP/2 multiplexes multiple requests over a single TCP connection.",
        "Service workers enable offline-capable progressive web apps.",
        "React components compose UIs from reusable, stateful building blocks.",
        "CSS Grid and Flexbox provide powerful layout primitives for modern browsers.",
        "TypeScript adds static typing to JavaScript for safer large-scale codebases.",
        "Docker containers package applications with all their dependencies.",
        "CI/CD pipelines automate testing and deployment on every code push.",
        "GraphQL lets clients request exactly the data they need in one round trip.",
        "Edge functions execute code close to the user for minimal latency.",
        "Content Security Policy headers mitigate XSS attacks.",
        "Web Vitals (LCP, CLS, FID) measure real-user performance.",
        "Semantic HTML improves accessibility and SEO out of the box.",
    ],
}

# Flatten to 50 docs (≈16+17+17)
all_docs: list[str] = []
all_meta: list[dict] = []
for topic, docs in TOPIC_DOCS.items():
    for doc in docs:
        all_docs.append(doc)
        all_meta.append({"topic": topic})

print(f"Total documents: {len(all_docs)}")
assert len(all_docs) >= 50 or len(all_docs) == sum(len(v) for v in TOPIC_DOCS.values())

# --- Index into SemanticSearch ---
lab_ss = SemanticSearch(collection_name="lab_search")
lab_ss.index(all_docs, metadatas=all_meta)
print(f"Indexed {lab_ss._collection.count()} documents.")

# --- 1. Basic search ---
print("\n--- Top 3 results for 'how does attention work?' ---")
for r in lab_ss.search("how does attention work?", n_results=3):
    print(f"  [{r['metadata']['topic']:20s}] {r['text'][:70]}")

# --- 2. Metadata-filtered search ---
print("\n--- Top 3 Python results for 'asynchronous programming' ---")
for r in lab_ss.search("asynchronous programming", n_results=3, where={"topic": "python"}):
    print(f"  [{r['metadata']['topic']:20s}] {r['text'][:70]}")

# --- 3. Simple search REPL (runs 3 demo queries non-interactively) ---
print("\n--- CLI-style demo (non-interactive) ---")
demo_queries = [
    ("transformer architecture", None),
    ("CSS layout techniques", "web_development"),
    ("list comprehension", "python"),
]
for q, topic in demo_queries:
    where = {"topic": topic} if topic else None
    tag = f"[filter: {topic}]" if topic else "[no filter]"
    results = lab_ss.search(q, n_results=2, where=where)
    print(f"\nQuery: '{q}' {tag}")
    for r in results:
        print(f"  -> {r['text'][:75]}")

# --- 4. Compare retrieval quality across chunk sizes ---
long_text = (
    "Machine learning is a branch of artificial intelligence. "
    "It enables systems to learn from data without explicit programming. "
    "Deep learning uses multi-layer neural networks to learn complex patterns. "
    "Natural language processing applies these techniques to text and speech. "
    "Computer vision applies deep learning to images and video. "
    "Reinforcement learning trains agents through reward and punishment signals. "
    "Unsupervised learning discovers hidden structure in unlabelled data. "
    "Semi-supervised learning combines a small amount of labelled data with a large amount of unlabelled data. "
    "Transfer learning leverages knowledge from one domain to improve performance in another. "
    "Federated learning trains models across decentralised devices without sharing raw data. "
) * 5  # ~2 500 chars

print("\n--- Chunk size comparison ---")
for cs in [128, 512, 1024]:
    chunks = chunk_text(long_text, chunk_size=cs, overlap=cs // 10)
    print(f"  chunk_size={cs:4d}  ->  {len(chunks):3d} chunks  (first chunk: {len(chunks[0])} chars)")

print("\nLab exercise complete.")

# %% [markdown]
# ## Key Takeaways
# - `mistral-embed` produces 1024-dimensional vectors that capture semantic meaning,
#   enabling similarity search far beyond keyword matching.
# - Cosine similarity is the standard metric for comparing embedding vectors; values
#   above ~0.85 indicate strong semantic overlap.
# - ChromaDB makes it trivial to add a vector store to any Python project — it runs
#   fully in-process with no external dependencies, and scales to millions of documents
#   with a persistent client.
# - Chunk size and overlap are critical hyperparameters: small chunks improve precision
#   but lose context; large chunks improve recall but may dilute relevance scores.
# - Metadata filters let you scope retrieval to a specific source, topic, or date range,
#   dramatically improving precision in multi-domain document collections.
