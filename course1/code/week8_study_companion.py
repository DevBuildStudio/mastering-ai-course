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
# # Week 8 Capstone: AI-Powered Study Companion
# This capstone integrates RAG, adaptive explanation, quiz generation, and a CLI interface
# into a complete study assistant powered by Mistral AI and ChromaDB.

# %% [markdown]
# ## 1. Setup
# Install dependencies and configure the environment. ChromaDB stores embeddings locally;
# Mistral handles all LLM and embedding calls; Rich provides colorized terminal output.

# %%
# pip install mistralai python-dotenv chromadb PyMuPDF rich fastapi uvicorn

import os
import json
import time
import uuid
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from mistralai import Mistral
from mistralai.models import SDKError

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("ChromaDB not installed. Run: pip install chromadb")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
    print("Rich not installed. Run: pip install rich")

API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=API_KEY)

print("Setup complete.")
print(f"ChromaDB available: {CHROMA_AVAILABLE}")
print(f"Rich available:     {RICH_AVAILABLE}")

# %% [markdown]
# ## 2. Document Ingestion
# `StudyDocumentIngester` reads PDF or plain-text files, splits them into overlapping
# chunks, embeds each chunk with `mistral-embed`, and stores the vectors in ChromaDB with
# rich metadata so downstream retrieval can filter by source and page number.

# %%
@dataclass
class IngestionReport:
    """Summary produced after ingesting one document."""
    source: str
    total_chunks: int
    total_chars: int
    elapsed_seconds: float
    collection_name: str


class StudyDocumentIngester:
    """Ingest PDF or text study materials into a ChromaDB vector store."""

    def __init__(self, collection_name: str = "study_docs",
                 chunk_size: int = 400, overlap: int = 80):
        """Initialise the ingester with chunking parameters and a ChromaDB collection."""
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.collection_name = collection_name
        if CHROMA_AVAILABLE:
            self._chroma = chromadb.Client()
            self._col = self._chroma.get_or_create_collection(collection_name)
        else:
            self._col = None

    # ------------------------------------------------------------------
    def _chunk(self, text: str) -> list[str]:
        """Split *text* into overlapping chunks of fixed character length."""
        chunks, start = [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end].strip())
            start += self.chunk_size - self.overlap
        return [c for c in chunks if c]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Return Mistral embeddings for a list of strings."""
        try:
            resp = client.embeddings.create(model="mistral-embed", inputs=texts)
            return [item.embedding for item in resp.data]
        except SDKError as exc:
            print(f"[Embed error] {exc}")
            return [[0.0] * 1024 for _ in texts]

    def _store(self, chunks: list[str], source: str, metadatas: list[dict]) -> None:
        """Embed chunks and upsert into ChromaDB."""
        if self._col is None:
            return
        embeddings = self._embed(chunks)
        ids = [f"{source}::{i}" for i in range(len(chunks))]
        self._col.upsert(ids=ids, documents=chunks,
                         embeddings=embeddings, metadatas=metadatas)

    # ------------------------------------------------------------------
    def ingest_text(self, path: str) -> IngestionReport:
        """Ingest a plain-text (.txt) file and return an IngestionReport."""
        t0 = time.time()
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        chunks = self._chunk(raw)
        metas = [{"source": path, "page": 0, "chunk_idx": i} for i in range(len(chunks))]
        self._store(chunks, os.path.basename(path), metas)
        return IngestionReport(source=path, total_chunks=len(chunks),
                               total_chars=len(raw),
                               elapsed_seconds=round(time.time() - t0, 2),
                               collection_name=self.collection_name)

    def ingest_pdf(self, path: str) -> IngestionReport:
        """Ingest a PDF file page-by-page and return an IngestionReport."""
        t0 = time.time()
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("PyMuPDF not installed. Falling back to ingest_text.")
            return self.ingest_text(path)

        doc = fitz.open(path)
        all_chunks, all_metas = [], []
        total_chars = 0
        for page_num, page in enumerate(doc):
            text = page.get_text()
            total_chars += len(text)
            for i, chunk in enumerate(self._chunk(text)):
                all_chunks.append(chunk)
                all_metas.append({"source": path, "page": page_num, "chunk_idx": i})
        self._store(all_chunks, os.path.basename(path), all_metas)
        return IngestionReport(source=path, total_chunks=len(all_chunks),
                               total_chars=total_chars,
                               elapsed_seconds=round(time.time() - t0, 2),
                               collection_name=self.collection_name)

    def query_context(self, query: str, n: int = 5) -> list[dict]:
        """Return the top-n most relevant chunks for *query*."""
        if self._col is None:
            return []
        emb = self._embed([query])[0]
        results = self._col.query(query_embeddings=[emb], n_results=n,
                                  include=["documents", "metadatas", "distances"])
        out = []
        for doc, meta, dist in zip(results["documents"][0],
                                   results["metadatas"][0],
                                   results["distances"][0]):
            out.append({"text": doc, "source": meta.get("source", ""),
                        "page": meta.get("page", 0), "score": round(1 - dist, 4)})
        return out


# Quick smoke-test (no real files required)
ingester = StudyDocumentIngester()
print("StudyDocumentIngester ready. Collection:", ingester.collection_name)

# %% [markdown]
# ## 3. RAG Chat Engine
# `StudyChatEngine` retrieves relevant document chunks, builds a depth-aware tutor prompt,
# maintains per-session history, and cites sources inline so students always know where an
# answer came from.

# %%
class StudyChatEngine:
    """Retrieval-augmented chat engine with depth-level control."""

    DEPTH_INSTRUCTIONS = {
        "simple": "Explain as if to a curious 12-year-old. Use plain language and short sentences.",
        "balanced": "Give a clear explanation suitable for an undergraduate student.",
        "technical": "Provide a rigorous technical explanation with precise terminology.",
    }

    def __init__(self, ingester: StudyDocumentIngester):
        """Attach to an existing StudyDocumentIngester instance."""
        self.ingester = ingester
        self._sessions: dict[str, list[dict]] = {}

    def retrieve_context(self, query: str, n: int = 5) -> list[dict]:
        """Retrieve the top-n relevant chunks for *query* from the vector store."""
        return self.ingester.query_context(query, n=n)

    def build_tutor_prompt(self, question: str, context: list[dict],
                           history: list[dict], depth_level: str = "balanced") -> list[dict]:
        """Construct the full message list for the tutor LLM call."""
        depth_instr = self.DEPTH_INSTRUCTIONS.get(depth_level, self.DEPTH_INSTRUCTIONS["balanced"])
        ctx_text = "\n\n".join(
            f"[Source: {c['source']}, p.{c['page']}]\n{c['text']}" for c in context
        )
        system = (
            f"You are an expert AI tutor. {depth_instr}\n"
            "Use ONLY the provided context to answer. "
            "Cite sources as [Source: <filename>, p.<page>] inline.\n\n"
            f"CONTEXT:\n{ctx_text or 'No documents ingested yet.'}"
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-6:])  # keep last 3 exchanges
        messages.append({"role": "user", "content": question})
        return messages

    def chat(self, question: str, session_id: Optional[str] = None,
             depth: str = "balanced") -> dict:
        """Answer *question* using RAG and return a result dict with answer + sources."""
        if session_id is None:
            session_id = str(uuid.uuid4())
        history = self._sessions.setdefault(session_id, [])
        t0 = time.time()
        context = self.retrieve_context(question)
        messages = self.build_tutor_prompt(question, context, history, depth)
        try:
            resp = client.chat.complete(model="mistral-large-latest", messages=messages)
            answer = resp.choices[0].message.content
        except SDKError as exc:
            answer = f"[API error: {exc}]"
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return {"answer": answer, "sources": context,
                "session_id": session_id,
                "latency_s": round(time.time() - t0, 2)}


chat_engine = StudyChatEngine(ingester)
print("StudyChatEngine ready.")

# %% [markdown]
# ## 4. Adaptive Explanation
# `AdaptiveExplainer` uses Mistral to classify the apparent complexity of a student's
# question, then generates an explanation pitched at the right level (ELI5, intermediate,
# or technical) and builds an analogy to aid intuition.

# %%
class AdaptiveExplainer:
    """Generate explanations pitched at the detected complexity level of the learner."""

    LEVELS = ("eli5", "intermediate", "technical")

    def detect_complexity(self, question: str) -> str:
        """Use Mistral to classify the complexity implied by *question*.

        Returns one of: 'eli5', 'intermediate', 'technical'.
        """
        prompt = (
            "Classify the complexity level implied by this student question into exactly "
            "one word: eli5, intermediate, or technical.\n\n"
            f"Question: {question}\n\nReply with a single word only."
        )
        try:
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip().lower()
            for lvl in self.LEVELS:
                if lvl in raw:
                    return lvl
            return "intermediate"
        except SDKError:
            return "intermediate"

    def explain(self, concept: str, detected_level: Optional[str] = None) -> str:
        """Generate a level-appropriate explanation of *concept*.

        If *detected_level* is None, it is inferred from *concept* itself.
        """
        level = detected_level or self.detect_complexity(concept)
        level_map = {
            "eli5": "Explain to a curious 5-year-old using a simple story or metaphor.",
            "intermediate": "Explain for an undergraduate with some background knowledge.",
            "technical": "Give a precise, rigorous technical explanation with examples.",
        }
        instr = level_map.get(level, level_map["intermediate"])
        prompt = f"{instr}\n\nConcept: {concept}"
        try:
            resp = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except SDKError as exc:
            return f"[API error: {exc}]"

    def generate_analogy(self, concept: str, domain: str = "everyday life") -> str:
        """Return a creative analogy for *concept* drawn from *domain*."""
        prompt = (
            f"Create a vivid, memorable analogy that explains '{concept}' "
            f"using something from '{domain}'. Keep it to 3-4 sentences."
        )
        try:
            resp = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except SDKError as exc:
            return f"[API error: {exc}]"


explainer = AdaptiveExplainer()
print("AdaptiveExplainer ready.")
print("Sample level detection:", explainer.detect_complexity("What is a neural network?"))

# %% [markdown]
# ## 5. Quiz Generator
# `QuizGenerator` creates multiple-choice and short-answer questions grounded in the
# retrieved context. An LLM judge grades free-text answers against a model answer with
# a numeric score and feedback, tracked across a `QuizSession`.

# %%
@dataclass
class QuizQuestion:
    """A single quiz question with metadata."""
    qtype: str          # "mcq" or "short"
    topic: str
    question: str
    choices: list[str]  # empty for short-answer
    model_answer: str
    context_snippet: str = ""


@dataclass
class QuizSession:
    """Tracks score and questions for a single quiz sitting."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    questions: list[QuizQuestion] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    feedbacks: list[str] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        """Sum of all graded scores (0–1 per question)."""
        return sum(self.scores)

    @property
    def max_score(self) -> float:
        """Maximum achievable score for this session."""
        return float(len(self.scores))

    @property
    def pct(self) -> float:
        """Percentage score rounded to one decimal place."""
        if not self.scores:
            return 0.0
        return round(self.total_score / self.max_score * 100, 1)


class QuizGenerator:
    """Generate and grade quiz questions using Mistral AI."""

    def _llm(self, prompt: str, json_mode: bool = False) -> str:
        """Call mistral-large-latest and return response text."""
        kwargs = {"model": "mistral-large-latest",
                  "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.complete(**kwargs)
            return resp.choices[0].message.content
        except SDKError as exc:
            return json.dumps({"error": str(exc)})

    def generate_mcq(self, topic: str, context: str, n: int = 5) -> list[QuizQuestion]:
        """Generate *n* multiple-choice questions about *topic* from *context*."""
        prompt = (
            f"Based on this study material:\n{context}\n\n"
            f"Create {n} multiple-choice questions about '{topic}'. "
            "Return JSON: {\"questions\": [{\"question\": \"...\", "
            "\"choices\": [\"A) ...\", \"B) ...\", \"C) ...\", \"D) ...\"], "
            "\"answer\": \"A) ...\"}]}"
        )
        raw = self._llm(prompt, json_mode=True)
        try:
            data = json.loads(raw)
            return [
                QuizQuestion(qtype="mcq", topic=topic,
                             question=q["question"], choices=q["choices"],
                             model_answer=q["answer"], context_snippet=context[:200])
                for q in data.get("questions", [])
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    def generate_short_answer(self, topic: str, context: str, n: int = 3) -> list[QuizQuestion]:
        """Generate *n* short-answer questions about *topic* from *context*."""
        prompt = (
            f"Based on this study material:\n{context}\n\n"
            f"Create {n} short-answer questions about '{topic}'. "
            "Return JSON: {\"questions\": [{\"question\": \"...\", \"model_answer\": \"...\"}]}"
        )
        raw = self._llm(prompt, json_mode=True)
        try:
            data = json.loads(raw)
            return [
                QuizQuestion(qtype="short", topic=topic,
                             question=q["question"], choices=[],
                             model_answer=q["model_answer"],
                             context_snippet=context[:200])
                for q in data.get("questions", [])
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    def grade_answer(self, question: QuizQuestion,
                     student_answer: str) -> tuple[float, str]:
        """Grade *student_answer* against the model answer; return (score 0-1, feedback)."""
        prompt = (
            f"Question: {question.question}\n"
            f"Model answer: {question.model_answer}\n"
            f"Student answer: {student_answer}\n\n"
            "Score the student answer from 0.0 to 1.0 and give one sentence of feedback. "
            "Return JSON: {\"score\": 0.8, \"feedback\": \"...\"}"
        )
        raw = self._llm(prompt, json_mode=True)
        try:
            data = json.loads(raw)
            return float(data.get("score", 0.0)), data.get("feedback", "")
        except (json.JSONDecodeError, ValueError):
            return 0.0, "Could not parse grading response."


quiz_gen = QuizGenerator()
print("QuizGenerator ready.")

# %% [markdown]
# ## 6. CLI Interface
# `StudyCompanionCLI` wires all components into an interactive terminal application.
# It provides a main menu for chat, quiz, document upload, and session stats, with
# colorized output via Rich when available.

# %%
class StudyCompanionCLI:
    """Interactive command-line study companion."""

    def __init__(self):
        """Initialise with shared components."""
        self.ingester = StudyDocumentIngester()
        self.chat_engine = StudyChatEngine(self.ingester)
        self.explainer = AdaptiveExplainer()
        self.quiz_gen = QuizGenerator()
        self.session_id = str(uuid.uuid4())
        self.chat_count = 0
        self.quiz_sessions: list[QuizSession] = []

    def _print(self, text: str, style: str = "") -> None:
        """Print with optional Rich style, falling back to plain print."""
        if RICH_AVAILABLE and style:
            console.print(text, style=style)
        else:
            print(text)

    def display_welcome(self) -> None:
        """Print a welcome banner."""
        banner = (
            "\n=== AI-Powered Study Companion ===\n"
            "Powered by Mistral AI + ChromaDB\n"
            f"Session: {self.session_id[:8]}\n"
        )
        if RICH_AVAILABLE:
            console.print(Panel(banner, style="bold cyan"))
        else:
            print(banner)

    def handle_chat_loop(self) -> None:
        """Run an interactive chat loop until the user types 'back'."""
        self._print("\n[Chat mode] Type your question. Type 'back' to return to menu.\n",
                    "bold yellow")
        depth = "balanced"
        while True:
            question = input("You: ").strip()
            if question.lower() in ("back", "exit", "quit", ""):
                break
            if question.startswith("/depth "):
                depth = question.split()[1]
                print(f"Depth set to: {depth}")
                continue
            result = self.chat_engine.chat(question, self.session_id, depth=depth)
            self._print(f"\nTutor ({result['latency_s']}s):\n{result['answer']}\n",
                        "green")
            if result["sources"]:
                srcs = ", ".join(
                    f"{s['source']}:p{s['page']}" for s in result["sources"][:2]
                )
                self._print(f"Sources: {srcs}", "dim")
            self.chat_count += 1

    def handle_quiz_loop(self) -> None:
        """Run a quiz session and record the results."""
        topic = input("Quiz topic: ").strip() or "general knowledge"
        self._print(f"\nGenerating quiz on '{topic}'...", "bold yellow")
        context = "No documents loaded." if not CHROMA_AVAILABLE else (
            " ".join(c["text"] for c in self.ingester.query_context(topic, n=3))
            or "No documents loaded."
        )
        questions = self.quiz_gen.generate_mcq(topic, context, n=3)
        questions += self.quiz_gen.generate_short_answer(topic, context, n=2)
        if not questions:
            self._print("Could not generate questions.", "red")
            return
        session = QuizSession()
        for i, q in enumerate(questions, 1):
            print(f"\nQ{i}: {q.question}")
            for ch in q.choices:
                print(f"  {ch}")
            answer = input("Your answer: ").strip()
            score, feedback = self.quiz_gen.grade_answer(q, answer)
            session.questions.append(q)
            session.scores.append(score)
            session.feedbacks.append(feedback)
            self._print(f"Score: {score:.0%} — {feedback}", "cyan")
        self.quiz_sessions.append(session)
        self._print(f"\nQuiz complete! Final score: {session.pct}%", "bold green")

    def handle_upload(self) -> None:
        """Prompt for a file path and ingest it."""
        path = input("File path (.txt or .pdf): ").strip()
        if not os.path.exists(path):
            self._print("File not found.", "red")
            return
        if path.endswith(".pdf"):
            report = self.ingester.ingest_pdf(path)
        else:
            report = self.ingester.ingest_text(path)
        self._print(
            f"Ingested {report.total_chunks} chunks from '{report.source}' "
            f"in {report.elapsed_seconds}s", "green"
        )

    def show_session_stats(self) -> None:
        """Display a summary of the current session."""
        avg_quiz = (
            round(sum(s.pct for s in self.quiz_sessions) / len(self.quiz_sessions), 1)
            if self.quiz_sessions else 0
        )
        stats = (
            f"Chat turns   : {self.chat_count}\n"
            f"Quizzes done : {len(self.quiz_sessions)}\n"
            f"Avg quiz pct : {avg_quiz}%\n"
            f"Session ID   : {self.session_id[:8]}"
        )
        if RICH_AVAILABLE:
            console.print(Panel(stats, title="Session Stats", style="bold blue"))
        else:
            print("\n--- Session Stats ---")
            print(stats)

    def run(self) -> None:
        """Launch the main menu loop."""
        self.display_welcome()
        menu = (
            "\n1) Chat with tutor\n"
            "2) Take a quiz\n"
            "3) Upload document\n"
            "4) Session stats\n"
            "5) Quit\n"
        )
        while True:
            print(menu)
            choice = input("Choose [1-5]: ").strip()
            if choice == "1":
                self.handle_chat_loop()
            elif choice == "2":
                self.handle_quiz_loop()
            elif choice == "3":
                self.handle_upload()
            elif choice == "4":
                self.show_session_stats()
            elif choice == "5":
                self._print("Goodbye!", "bold cyan")
                break
            else:
                self._print("Invalid choice.", "red")


print("StudyCompanionCLI defined. Run StudyCompanionCLI().run() to start the app.")

# %% [markdown]
# ## 7. Lab Exercise: Integration Test
# This section performs a complete end-to-end test without interactive input:
# ingest sample documents, run RAG chat questions, generate and grade a quiz,
# then print a full session report.

# %%
import tempfile, pathlib

def run_integration_test() -> None:
    """Full integration test: ingest docs, chat, quiz, grade, and report.

    Creates three in-memory text files, ingests them, runs 10 chat questions,
    generates a 5-question quiz, grades answers with the LLM judge, and prints
    a structured session report.
    """
    print("\n" + "=" * 60)
    print("LAB EXERCISE: Integration Test")
    print("=" * 60)

    # --- 1. Create sample documents -----------------------------------------
    sample_docs = {
        "ml_basics.txt": (
            "Machine learning is a subset of artificial intelligence that enables systems "
            "to learn from data. Supervised learning uses labelled examples. "
            "Unsupervised learning finds hidden patterns without labels. "
            "Reinforcement learning trains agents via reward signals. "
            "Gradient descent minimises a loss function by iteratively updating weights."
        ),
        "neural_nets.txt": (
            "A neural network consists of layers of interconnected nodes called neurons. "
            "The input layer receives raw features. Hidden layers extract intermediate "
            "representations. The output layer produces predictions. "
            "Backpropagation computes gradients via the chain rule. "
            "Activation functions such as ReLU introduce non-linearity."
        ),
        "transformers.txt": (
            "Transformers use self-attention to weigh the importance of each token "
            "relative to every other token in a sequence. "
            "Multi-head attention runs several attention operations in parallel. "
            "Positional encodings inject sequence-order information. "
            "BERT is encoder-only; GPT is decoder-only; T5 is encoder-decoder."
        ),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ing = StudyDocumentIngester(collection_name="lab_test")
        for fname, content in sample_docs.items():
            fpath = pathlib.Path(tmpdir) / fname
            fpath.write_text(content, encoding="utf-8")
            report = ing.ingest_text(str(fpath))
            print(f"  Ingested: {fname} — {report.total_chunks} chunks in {report.elapsed_seconds}s")

        engine = StudyChatEngine(ing)
        session_id = "lab-session"

        # --- 2. Run 10 RAG chat questions -----------------------------------
        questions = [
            "What is supervised learning?",
            "How does gradient descent work?",
            "What is backpropagation?",
            "What does ReLU do?",
            "Explain self-attention in transformers.",
            "What is the difference between BERT and GPT?",
            "What are positional encodings?",
            "What is reinforcement learning?",
            "How do hidden layers help neural networks?",
            "What is multi-head attention?",
        ]
        print(f"\nRunning {len(questions)} RAG chat questions...")
        for i, q in enumerate(questions, 1):
            result = engine.chat(q, session_id=session_id)
            snippet = result["answer"][:120].replace("\n", " ")
            print(f"  Q{i:02d} ({result['latency_s']}s): {snippet}...")

        # --- 3. Generate 5-question quiz ------------------------------------
        ctx_chunks = ing.query_context("machine learning neural networks transformers", n=5)
        ctx_text = " ".join(c["text"] for c in ctx_chunks)
        print("\nGenerating 5-question quiz...")
        mcqs = quiz_gen.generate_mcq("AI foundations", ctx_text, n=3)
        shorts = quiz_gen.generate_short_answer("AI foundations", ctx_text, n=2)
        all_qs = mcqs + shorts
        print(f"  Generated: {len(mcqs)} MCQs + {len(shorts)} short-answer = {len(all_qs)} total")

        # --- 4. Grade answers with LLM judge --------------------------------
        quiz_sess = QuizSession()
        sample_answers = [
            "A learning algorithm trained on labelled data",
            "By computing the gradient of the loss",
            "A",          # MCQ placeholder answer
            "Non-linear activation used in hidden layers",
            "Weights importance of each token relative to others",
        ]
        print("\nGrading answers with LLM judge...")
        for i, (q, ans) in enumerate(zip(all_qs, sample_answers), 1):
            score, feedback = quiz_gen.grade_answer(q, ans)
            quiz_sess.questions.append(q)
            quiz_sess.scores.append(score)
            quiz_sess.feedbacks.append(feedback)
            print(f"  Q{i}: score={score:.2f} | {feedback[:80]}")

        # --- 5. Print session report ----------------------------------------
        print("\n" + "=" * 60)
        print("SESSION REPORT")
        print("=" * 60)
        print(f"Documents ingested : {len(sample_docs)}")
        print(f"Chat questions     : {len(questions)}")
        print(f"Quiz questions     : {len(all_qs)}")
        print(f"Quiz score         : {quiz_sess.total_score:.1f} / {quiz_sess.max_score:.0f}  ({quiz_sess.pct}%)")
        assert len(quiz_sess.scores) == len(all_qs), "Score count mismatch!"
        assert quiz_sess.max_score > 0, "No quiz scores recorded!"
        print("\nAll assertions passed.")
        print("=" * 60)


# Run the integration test
t_start = time.time()
run_integration_test()
print(f"\nTotal integration test time: {round(time.time() - t_start, 1)}s")

# %% [markdown]
# ## Key Takeaways
# - RAG grounds LLM answers in your own documents, dramatically reducing hallucinations
#   and providing citable sources that students can verify.
# - Adaptive depth control (ELI5 / intermediate / technical) personalises explanations
#   without maintaining a complex user model — a simple classifier prompt is enough.
# - ChromaDB lets you store and query embeddings locally with zero infrastructure,
#   making it ideal for course projects and prototypes before scaling to managed services.
# - An LLM-as-judge pattern converts free-text quiz answers into numeric scores with
#   natural-language feedback, eliminating the need for brittle regex or keyword matching.
# - Structuring a capstone around dataclasses (IngestionReport, QuizQuestion, QuizSession)
#   keeps state explicit and testable, which is essential when components are composed
#   in a multi-step AI pipeline.
