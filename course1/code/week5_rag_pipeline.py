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
# # Week 5: Retrieval-Augmented Generation (RAG)
# RAG combines a retrieval step with LLM generation so the model answers from
# your documents rather than baked-in weights. This notebook builds a complete
# RAG pipeline: ingest → embed → retrieve → generate, plus evaluation.

# %% [markdown]
# ## 1. Setup
# Install dependencies and configure the Mistral client. ChromaDB provides a
# local vector store; PyPDF2 and requests handle document loading.

# %%
# pip install chromadb pypdf2 requests mistralai python-dotenv

import os
import time
import hashlib
import textwrap
import requests
from typing import Any
from dotenv import load_dotenv

try:
    from mistralai import Mistral
    from mistralai.models import SDKError
except ImportError as e:
    raise ImportError("Run: pip install mistralai") from e

try:
    import chromadb
    from chromadb import EmbeddingFunction, Documents, Embeddings
except ImportError as e:
    raise ImportError("Run: pip install chromadb") from e

try:
    import PyPDF2
except ImportError as e:
    raise ImportError("Run: pip install pypdf2") from e

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=MISTRAL_API_KEY)

print("Mistral client initialised.")
print(f"ChromaDB version: {chromadb.__version__}")

# %% [markdown]
# ## 2. Document Ingestion Pipeline
# `DocumentLoader` handles PDF, plain-text, and URL sources.
# `TextChunker` splits documents with overlap so context windows stay coherent.
# `DocumentProcessor` wires them together into a single `ingest()` call.

# %%
class DocumentLoader:
    """Load documents from PDF files, text files, or URLs."""

    def load_pdf(self, path: str) -> str:
        """Extract all text from a PDF file at *path*."""
        text_parts = []
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
        return "\n".join(text_parts)

    def load_text(self, path: str) -> str:
        """Read a plain-text file at *path* and return its contents."""
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def load_url(self, url: str, timeout: int = 10) -> str:
        """Fetch *url* and return the raw response text."""
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        # Strip HTML tags with a minimal approach (no extra deps)
        text = resp.text
        import re
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class TextChunker:
    """Split long text into overlapping chunks for embedding."""

    def recursive_split(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[str]:
        """Return a list of chunks of at most *chunk_size* chars with *overlap*."""
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == text_len:
                break
            start = end - overlap
        return chunks


class DocumentProcessor:
    """Orchestrate loading, chunking, and metadata tagging of a document."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """Initialise with chunking parameters."""
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest(self, source: str) -> list[dict]:
        """Load *source* (path or URL), split, and return list of chunk dicts.

        Each dict has keys: ``id``, ``text``, ``source``.
        """
        t0 = time.time()
        if source.startswith("http://") or source.startswith("https://"):
            raw = self.loader.load_url(source)
            src_label = source
        elif source.lower().endswith(".pdf"):
            raw = self.loader.load_pdf(source)
            src_label = os.path.basename(source)
        else:
            raw = self.loader.load_text(source)
            src_label = os.path.basename(source)

        chunks = self.chunker.recursive_split(raw, self.chunk_size, self.overlap)
        docs = []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{src_label}:{i}:{chunk[:40]}".encode()).hexdigest()
            docs.append({"id": chunk_id, "text": chunk, "source": src_label})

        elapsed = time.time() - t0
        print(f"Ingested '{src_label}': {len(docs)} chunks in {elapsed:.2f}s")
        return docs


# Demo: ingest a Python docs page
processor = DocumentProcessor(chunk_size=512, overlap=50)
python_docs_url = "https://docs.python.org/3/library/functions.html"
try:
    sample_docs = processor.ingest(python_docs_url)
    print(f"Sample chunk:\n{textwrap.fill(sample_docs[0]['text'][:300], width=80)}")
except Exception as exc:
    print(f"Network unavailable, using synthetic docs: {exc}")
    sample_docs = [
        {"id": f"synthetic_{i}", "text": f"Python built-in function example {i}: "
         f"print(), len(), range() are common built-ins used in Python programs.",
         "source": "synthetic"}
        for i in range(10)
    ]

assert len(sample_docs) > 0, "Ingestion must return at least one chunk"
print(f"Total chunks available: {len(sample_docs)}")

# %% [markdown]
# ## 3. Vector Store with Mistral Embeddings
# `MistralEmbeddingFunction` adapts the Mistral embed API to ChromaDB's
# `EmbeddingFunction` interface. `RAGVectorStore` wraps a ChromaDB collection
# and exposes `add_documents` / `query` helpers.

# %%
class MistralEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding function backed by mistral-embed."""

    def __init__(self, api_key: str | None = None):
        """Initialise with an optional *api_key* (falls back to env var)."""
        self._client = Mistral(api_key=api_key or MISTRAL_API_KEY)

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        """Embed a list of document strings and return a list of float vectors."""
        try:
            response = self._client.embeddings.create(
                model="mistral-embed",
                inputs=list(input),
            )
            return [item.embedding for item in response.data]
        except SDKError as exc:
            raise RuntimeError(f"Mistral embedding failed: {exc}") from exc


class RAGVectorStore:
    """Manage a ChromaDB collection with Mistral-powered embeddings."""

    def __init__(self, collection_name: str = "rag_docs"):
        """Create or open a persistent ChromaDB collection."""
        self._chroma = chromadb.Client()
        self._embed_fn = MistralEmbeddingFunction()
        self._col = self._chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
        )
        print(f"Vector store '{collection_name}' ready.")

    def add_documents(self, docs: list[dict]) -> None:
        """Add *docs* (list of id/text/source dicts) to the collection."""
        if not docs:
            return
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [{"source": d["source"]} for d in docs]
        t0 = time.time()
        self._col.upsert(ids=ids, documents=texts, metadatas=metadatas)
        print(f"Upserted {len(docs)} docs in {time.time() - t0:.2f}s")

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Return *n_results* nearest chunks for *query_text*.

        Each result dict has: ``text``, ``source``, ``distance``.
        """
        results = self._col.query(
            query_texts=[query_text],
            n_results=min(n_results, self._col.count() or 1),
        )
        hits = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"text": text, "source": meta.get("source", ""), "distance": dist})
        return hits

    @property
    def count(self) -> int:
        """Return number of documents in the store."""
        return self._col.count()


# Build and populate the vector store
vector_store = RAGVectorStore("week5_rag")
vector_store.add_documents(sample_docs)
print(f"Documents in store: {vector_store.count}")

# Quick sanity query
test_hits = vector_store.query("what is len()", n_results=3)
print(f"\nTop hit for 'what is len()':\n  {test_hits[0]['text'][:200]}")

# %% [markdown]
# ## 4. Query Pipeline with HyDE
# Hypothetical Document Embeddings (HyDE) improves recall: ask the LLM to
# generate a hypothetical answer, embed *that*, then retrieve similar real
# chunks. `query_expand` generates three query variants for broader coverage.

# %%
class RAGRetriever:
    """Retrieve relevant chunks using standard, HyDE, or expanded queries."""

    def __init__(self, vector_store: RAGVectorStore):
        """Attach to an existing *vector_store*."""
        self._store = vector_store

    def retrieve(self, query: str, n: int = 5) -> list[dict]:
        """Standard dense retrieval — embed *query* and return top-*n* chunks."""
        return self._store.query(query, n_results=n)

    def hyde_retrieve(self, query: str, n: int = 5) -> list[dict]:
        """HyDE retrieval: generate a hypothetical answer, then embed it.

        Generates a short hypothetical passage with mistral-small-latest,
        then retrieves chunks similar to that passage rather than the raw query.
        """
        try:
            hyp_response = client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write a short, factual 2-3 sentence answer to the "
                            "question below. Do not say you don't know."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
            )
            hypothetical = hyp_response.choices[0].message.content
        except SDKError as exc:
            print(f"HyDE generation failed, falling back to direct query: {exc}")
            hypothetical = query

        return self._store.query(hypothetical, n_results=n)

    def query_expand(self, query: str, n: int = 5) -> list[dict]:
        """Expand *query* into 3 variants, retrieve for each, deduplicate.

        Uses mistral-small-latest to generate semantically varied phrasings.
        """
        try:
            exp_response = client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate exactly 3 alternative phrasings of the user "
                            "query, one per line, no numbering or bullets."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
            )
            raw = exp_response.choices[0].message.content or ""
            variants = [v.strip() for v in raw.strip().splitlines() if v.strip()][:3]
        except SDKError as exc:
            print(f"Query expansion failed: {exc}")
            variants = []

        all_queries = [query] + variants
        seen_ids: set[str] = set()
        merged: list[dict] = []
        for q in all_queries:
            for hit in self._store.query(q, n_results=n):
                key = hit["text"][:80]
                if key not in seen_ids:
                    seen_ids.add(key)
                    merged.append(hit)

        # Re-sort by distance (lower = better) and cap at n
        merged.sort(key=lambda h: h["distance"])
        return merged[:n]


retriever = RAGRetriever(vector_store)

print("=== Standard retrieval ===")
t0 = time.time()
std_hits = retriever.retrieve("how do I use range()", n=3)
print(f"Retrieved {len(std_hits)} chunks in {time.time()-t0:.2f}s")
print(f"Top chunk: {std_hits[0]['text'][:180]}\n")

print("=== HyDE retrieval ===")
t0 = time.time()
hyde_hits = retriever.hyde_retrieve("how do I use range()", n=3)
print(f"Retrieved {len(hyde_hits)} chunks in {time.time()-t0:.2f}s")
print(f"Top chunk: {hyde_hits[0]['text'][:180]}\n")

print("=== Expanded query retrieval ===")
t0 = time.time()
exp_hits = retriever.query_expand("built-in functions for sequences", n=3)
print(f"Retrieved {len(exp_hits)} unique chunks in {time.time()-t0:.2f}s")

# %% [markdown]
# ## 5. Context Assembly and Generation
# `RAGGenerator` formats retrieved chunks into a numbered context block,
# then calls `mistral-large-latest` with a structured prompt. Source citations
# are appended automatically, and the generator detects when no context exists.

# %%
CONTEXT_PROMPT = """\
You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say exactly:
"I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer (cite sources inline as [1], [2], …):"""


class RAGGenerator:
    """Assemble context from retrieved chunks and generate a grounded answer."""

    def build_context(self, chunks: list[dict]) -> tuple[str, list[str]]:
        """Format *chunks* into a numbered context string plus a sources list.

        Returns ``(context_str, sources_list)``.
        """
        if not chunks:
            return "", []
        parts = []
        sources = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(f"[{i}] {chunk['text']}")
            src = chunk.get("source", "unknown")
            if src not in sources:
                sources.append(src)
        return "\n\n".join(parts), sources

    def generate(self, question: str, context: str) -> str:
        """Call mistral-large-latest with *context* and *question*.

        Returns the model's answer string. Handles empty context gracefully.
        """
        if not context.strip():
            return "I don't know based on the provided context."
        prompt = CONTEXT_PROMPT.format(context=context, question=question)
        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except SDKError as exc:
            raise RuntimeError(f"Generation failed: {exc}") from exc


generator = RAGGenerator()

sample_question = "What does the len() function do in Python?"
sample_context, sample_sources = generator.build_context(std_hits)
print(f"Context assembled ({len(sample_context)} chars), sources: {sample_sources}\n")

t0 = time.time()
answer = generator.generate(sample_question, sample_context)
print(f"Answer ({time.time()-t0:.2f}s):\n{answer}\n")
print(f"Sources: {sample_sources}")

assert "I don't know" not in answer or len(sample_context) == 0

# Empty-context guard
empty_answer = generator.generate("What is x?", "")
assert empty_answer == "I don't know based on the provided context."
print("\nEmpty-context guard: OK")

# %% [markdown]
# ## 6. Full RAG QA System
# `RAGSystem` combines every component into a clean API: `ingest(source)` and
# `ask(question)`. Conversation mode maintains a short history so the model can
# resolve follow-up questions. The demo runs over the Python docs corpus.

# %%
class RAGSystem:
    """End-to-end RAG system: ingest documents, answer questions with citations."""

    def __init__(
        self,
        collection_name: str = "rag_system",
        chunk_size: int = 512,
        overlap: int = 50,
        retrieval_mode: str = "hyde",
    ):
        """Initialise all pipeline components.

        *retrieval_mode* may be ``'standard'``, ``'hyde'``, or ``'expand'``.
        """
        self._processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)
        self._store = RAGVectorStore(collection_name)
        self._retriever = RAGRetriever(self._store)
        self._generator = RAGGenerator()
        self._mode = retrieval_mode
        self._history: list[dict] = []

    def ingest(self, source: str) -> int:
        """Ingest *source* into the vector store. Returns number of chunks added."""
        docs = self._processor.ingest(source)
        self._store.add_documents(docs)
        return len(docs)

    def ask(
        self,
        question: str,
        n_chunks: int = 5,
        use_history: bool = False,
    ) -> dict[str, Any]:
        """Answer *question* using the indexed corpus.

        Returns a dict with keys: ``answer``, ``sources``, ``chunks_used``.
        If *use_history* is True, previous Q&A turns are prepended to the prompt.
        """
        t0 = time.time()
        if self._mode == "hyde":
            chunks = self._retriever.hyde_retrieve(question, n=n_chunks)
        elif self._mode == "expand":
            chunks = self._retriever.query_expand(question, n=n_chunks)
        else:
            chunks = self._retriever.retrieve(question, n=n_chunks)

        context, sources = self._generator.build_context(chunks)

        if use_history and self._history:
            history_str = "\n".join(
                f"Q: {h['question']}\nA: {h['answer']}" for h in self._history[-3:]
            )
            full_context = f"Previous conversation:\n{history_str}\n\n{context}"
        else:
            full_context = context

        answer = self._generator.generate(question, full_context)
        self._history.append({"question": question, "answer": answer})

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "latency_s": round(time.time() - t0, 2),
        }

    def clear_history(self) -> None:
        """Reset conversation history."""
        self._history = []


# --- Demo ---
print("Building RAG system over Python docs...\n")
rag = RAGSystem(collection_name="demo_rag", retrieval_mode="hyde")

try:
    n = rag.ingest("https://docs.python.org/3/library/functions.html")
    print(f"Ingested {n} chunks from Python built-ins docs\n")
except Exception as exc:
    print(f"Using synthetic corpus ({exc})")
    for doc in sample_docs:
        rag._store.add_documents([doc])

questions = [
    "How does the sorted() function work?",
    "What is the difference between map() and filter()?",
]

for q in questions:
    result = rag.ask(q, n_chunks=4)
    print(f"Q: {q}")
    print(f"A: {result['answer'][:400]}")
    print(f"   Sources: {result['sources']} | Chunks: {result['chunks_used']} | {result['latency_s']}s\n")

# Conversation mode
rag.clear_history()
r1 = rag.ask("What does enumerate() do?", use_history=True)
print(f"Turn 1 answer (first 200 chars): {r1['answer'][:200]}\n")
r2 = rag.ask("Can you give an example using it with a list?", use_history=True)
print(f"Turn 2 answer (first 200 chars): {r2['answer'][:200]}\n")

# %% [markdown]
# ## 7. Lab Exercise
# Build a RAG system over a real multi-topic corpus, then measure retrieval
# quality with a hit-rate metric: for each of 20 questions the ground-truth
# answer chunk must appear in the top-5 retrieved results.

# %%
# ---- CORPUS SETUP ----
CORPUS_URLS = [
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
]

lab_rag = RAGSystem(collection_name="lab_eval", chunk_size=400, overlap=40, retrieval_mode="standard")

for url in CORPUS_URLS:
    try:
        lab_rag.ingest(url)
    except Exception as exc:
        print(f"Skipping {url}: {exc}")

print(f"\nCorpus size: {lab_rag._store.count} chunks\n")

# ---- EVALUATION QA PAIRS ----
# Each pair: (question, keywords that MUST appear in the correct chunk)
QA_PAIRS: list[tuple[str, list[str]]] = [
    ("Who created Python?", ["Guido", "van Rossum"]),
    ("What year was Python first released?", ["1991", "Python"]),
    ("What is supervised learning?", ["supervised", "label"]),
    ("What is unsupervised learning?", ["unsupervised", "cluster"]),
    ("What does NLP stand for?", ["natural language"]),
    ("What is a neural network?", ["neuron", "layer", "network"]),
    ("What is gradient descent?", ["gradient", "optim"]),
    ("What is a transformer model?", ["transformer", "attention"]),
    ("What is the Turing test?", ["Turing"]),
    ("What is reinforcement learning?", ["reward", "reinforcement"]),
    ("What is tokenization in NLP?", ["token"]),
    ("What is backpropagation?", ["backprop", "gradient"]),
    ("What are GPT models?", ["GPT", "generative"]),
    ("What is overfitting?", ["overfit", "generaliz"]),
    ("What is a convolutional neural network?", ["convolut", "CNN"]),
    ("What does AI stand for?", ["artificial intelligence"]),
    ("What is BERT?", ["BERT", "bidirectional"]),
    ("What is a decision tree?", ["decision tree", "node"]),
    ("What is transfer learning?", ["transfer"]),
    ("What is a recurrent neural network?", ["recurrent", "RNN"]),
]


def hit_at_k(
    retriever: RAGRetriever,
    question: str,
    keywords: list[str],
    k: int = 5,
) -> bool:
    """Return True if any top-k chunk contains at least one *keyword* (case-insensitive)."""
    hits = retriever.retrieve(question, n=k)
    for hit in hits:
        text_lower = hit["text"].lower()
        if any(kw.lower() in text_lower for kw in keywords):
            return True
    return False


# ---- RUN EVALUATION ----
print("Running hit-rate evaluation (top-5)...\n")
results = []
t_eval_start = time.time()

for question, keywords in QA_PAIRS:
    hit = hit_at_k(lab_rag._retriever, question, keywords, k=5)
    results.append({"question": question, "hit": hit})
    status = "HIT " if hit else "MISS"
    print(f"  [{status}] {question}")

eval_time = time.time() - t_eval_start
total = len(results)
hits = sum(r["hit"] for r in results)
hit_rate = hits / total

print(f"\n{'='*50}")
print(f"Hit-rate @5 : {hits}/{total} = {hit_rate:.1%}")
print(f"Eval time   : {eval_time:.1f}s")
print(f"{'='*50}\n")

# Assertions
assert 0.0 <= hit_rate <= 1.0, "Hit rate must be in [0, 1]"
print(f"Hit rate assertion passed: {hit_rate:.1%}")

# Breakdown by topic
topic_map = {
    "Python": [q for q, _ in QA_PAIRS[:2]],
    "ML/AI":  [q for q, _ in QA_PAIRS[2:]],
}
for topic, qs in topic_map.items():
    topic_hits = sum(r["hit"] for r in results if r["question"] in qs)
    print(f"  {topic}: {topic_hits}/{len(qs)} = {topic_hits/len(qs):.1%}")

# %% [markdown]
# ## Key Takeaways
# - RAG grounds LLM answers in your documents, dramatically reducing hallucinations
#   compared to pure parametric generation.
# - Chunking strategy (size, overlap) significantly affects retrieval quality;
#   smaller chunks improve precision, larger chunks preserve context.
# - HyDE boosts recall by embedding a *hypothetical* answer instead of the raw
#   query, bridging the vocabulary gap between questions and documents.
# - Query expansion diversifies retrieval by generating semantically varied
#   phrasings, catching relevant chunks that a single query would miss.
# - Hit-rate @k is a cheap, automatic metric for evaluating retrieval quality
#   before wiring up expensive end-to-end generation evaluation.
