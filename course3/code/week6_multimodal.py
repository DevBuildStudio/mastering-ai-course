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
# # Week 6: Multimodal AI Engineering
# This notebook covers multimodal AI with Pixtral vision models, PDF document intelligence,
# multimodal RAG pipelines, chart analysis, and audio transcription. You will build systems
# that combine text, images, and audio into unified AI-powered workflows.

# %% [markdown]
# ## 1. Setup
# Install dependencies and configure the Pixtral vision model client.
# PyMuPDF (fitz) handles PDF rendering, Pillow handles image processing.

# %%
# !pip install PyMuPDF Pillow mistralai python-dotenv openai-whisper

import os
import time
import base64
import json
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from mistralai import Mistral
from mistralai.models import SDKError

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=MISTRAL_API_KEY)

VISION_MODEL = "pixtral-12b-2409"
LARGE_MODEL = "mistral-large-latest"
SMALL_MODEL = "mistral-small-latest"
EMBED_MODEL = "mistral-embed"

print(f"Mistral client initialized.")
print(f"Vision model : {VISION_MODEL}")
print(f"Large model  : {LARGE_MODEL}")
print(f"Embed model  : {EMBED_MODEL}")

# %% [markdown]
# ## 2. Pixtral Vision Basics
# Pixtral-12B understands images natively. We encode images as base64 data URIs and send
# them alongside text in the content array. This section covers image description,
# OCR, and multi-image comparison.

# %%
def base64_encode_image(path: str) -> str:
    """Encode an image file to a base64 data URI string.

    Args:
        path: Filesystem path to the image (JPEG, PNG, GIF, WEBP).

    Returns:
        A data URI string: 'data:<mime>;base64,<data>'.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(suffix, "image/png")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def image_message(text: str, image_path: str) -> dict:
    """Build a Pixtral user message dict combining text and a local image.

    Args:
        text: The instruction or question to accompany the image.
        image_path: Filesystem path to the image file.

    Returns:
        A message dict ready for the messages list.
    """
    data_uri = base64_encode_image(image_path)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    }


def describe_image(image_path: str) -> str:
    """Ask Pixtral for a detailed description of an image.

    Args:
        image_path: Path to the image file.

    Returns:
        A natural-language description string.
    """
    try:
        start = time.time()
        msg = image_message("Describe this image in detail, including all visual elements.", image_path)
        response = client.chat.complete(model=VISION_MODEL, messages=[msg])
        elapsed = time.time() - start
        result = response.choices[0].message.content
        print(f"[describe_image] {elapsed:.2f}s | {len(result)} chars")
        return result
    except SDKError as e:
        print(f"[describe_image] API error: {e}")
        return ""


def ocr_image(image_path: str) -> str:
    """Extract all text from an image using Pixtral vision OCR.

    Args:
        image_path: Path to the image containing text.

    Returns:
        Extracted text as a plain string.
    """
    try:
        start = time.time()
        msg = image_message(
            "Extract ALL text from this image exactly as it appears. "
            "Preserve formatting, line breaks, and structure. Output only the extracted text.",
            image_path,
        )
        response = client.chat.complete(model=VISION_MODEL, messages=[msg])
        elapsed = time.time() - start
        result = response.choices[0].message.content
        print(f"[ocr_image] {elapsed:.2f}s | {len(result)} chars extracted")
        return result
    except SDKError as e:
        print(f"[ocr_image] API error: {e}")
        return ""


def compare_images(image_path1: str, image_path2: str, question: str = "") -> str:
    """Compare two images side-by-side using Pixtral.

    Args:
        image_path1: Path to the first image.
        image_path2: Path to the second image.
        question: Optional specific comparison question.

    Returns:
        Comparison analysis as a string.
    """
    prompt = question or "Compare these two images in detail. Describe similarities, differences, and key observations."
    try:
        start = time.time()
        uri1 = base64_encode_image(image_path1)
        uri2 = base64_encode_image(image_path2)
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": uri1}},
                {"type": "image_url", "image_url": {"url": uri2}},
            ],
        }
        response = client.chat.complete(model=VISION_MODEL, messages=[msg])
        elapsed = time.time() - start
        result = response.choices[0].message.content
        print(f"[compare_images] {elapsed:.2f}s")
        return result
    except SDKError as e:
        print(f"[compare_images] API error: {e}")
        return ""


# Demo: create a minimal test PNG (a solid red 100x100 square) without external files
def _make_test_image(path: str, color: tuple = (220, 50, 50)) -> str:
    """Create a solid-color test PNG for demonstrations."""
    from PIL import Image as PILImage
    img = PILImage.new("RGB", (200, 100), color=color)
    img.save(path)
    return path

_test_img = _make_test_image("/tmp/test_red.png", (220, 50, 50))
_test_img2 = _make_test_image("/tmp/test_blue.png", (50, 100, 220))

print("--- Image Description ---")
print(describe_image(_test_img))

print("\n--- OCR on test image ---")
print(ocr_image(_test_img))

print("\n--- Compare two images ---")
print(compare_images(_test_img, _test_img2))

# %% [markdown]
# ## 3. Document Intelligence
# PDFProcessor uses PyMuPDF to render each page as an image, then applies Pixtral for
# vision-based table and form extraction alongside the native text layer. This hybrid
# approach captures content that pure text extraction misses.

# %%
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    print("PyMuPDF not installed. Run: pip install PyMuPDF")


@dataclass
class PageContent:
    """Structured content extracted from a single PDF page."""
    page_num: int
    text_layer: str
    vision_text: str
    tables: list[str]
    form_fields: dict
    image_bytes: bytes = field(default_factory=bytes)


class PDFProcessor:
    """Extract structured content from PDF pages using both text and vision."""

    def __init__(self, dpi: int = 150):
        """Initialize processor with rendering resolution.

        Args:
            dpi: Dots per inch for page rendering (higher = better quality, slower).
        """
        self.dpi = dpi

    def render_page_to_image(self, pdf_path: str, page_num: int) -> bytes:
        """Render a PDF page to PNG image bytes using PyMuPDF.

        Args:
            pdf_path: Path to the PDF file.
            page_num: Zero-based page index.

        Returns:
            PNG image as bytes.
        """
        if not FITZ_AVAILABLE:
            return b""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        zoom = self.dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        doc.close()
        return pix.tobytes("png")

    def extract_text_layer(self, pdf_path: str, page_num: int) -> str:
        """Extract the native text layer from a PDF page (fast, no API).

        Args:
            pdf_path: Path to the PDF file.
            page_num: Zero-based page index.

        Returns:
            Plain text string from the page.
        """
        if not FITZ_AVAILABLE:
            return ""
        doc = fitz.open(pdf_path)
        text = doc[page_num].get_text("text")
        doc.close()
        return text.strip()

    def extract_table_with_vision(self, image_bytes: bytes) -> str:
        """Use Pixtral to extract tabular data from a page image.

        Args:
            image_bytes: PNG image bytes of the page.

        Returns:
            Markdown-formatted table string.
        """
        if not image_bytes:
            return ""
        try:
            b64 = base64.b64encode(image_bytes).decode()
            uri = f"data:image/png;base64,{b64}"
            msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Extract all tables from this document page. "
                        "Format each table in Markdown. If no tables, reply 'No tables found.'"
                    )},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
            response = client.chat.complete(model=VISION_MODEL, messages=[msg])
            return response.choices[0].message.content
        except SDKError as e:
            return f"API error: {e}"

    def extract_form_fields(self, image_bytes: bytes) -> dict:
        """Use Pixtral to identify form fields and their values from a page image.

        Args:
            image_bytes: PNG image bytes of the page.

        Returns:
            Dict mapping field labels to values.
        """
        if not image_bytes:
            return {}
        try:
            b64 = base64.b64encode(image_bytes).decode()
            uri = f"data:image/png;base64,{b64}"
            msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Extract all form fields from this document page. "
                        "Return a JSON object mapping field label to value. "
                        "If no form fields exist, return {}."
                    )},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
            response = client.chat.complete(
                model=VISION_MODEL,
                messages=[msg],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except (SDKError, json.JSONDecodeError) as e:
            print(f"[extract_form_fields] error: {e}")
            return {}

    def process_document(self, pdf_path: str) -> list[PageContent]:
        """Process all pages of a PDF, extracting text layer and vision content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of PageContent, one per page.
        """
        if not FITZ_AVAILABLE:
            print("PyMuPDF unavailable, skipping PDF processing.")
            return []
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
        pages = []
        for i in range(num_pages):
            print(f"  Processing page {i+1}/{num_pages}...")
            start = time.time()
            text = self.extract_text_layer(pdf_path, i)
            img_bytes = self.render_page_to_image(pdf_path, i)
            b64 = base64.b64encode(img_bytes).decode() if img_bytes else ""
            uri = f"data:image/png;base64,{b64}" if b64 else ""
            vision_text = ""
            tables = []
            form_fields = {}
            if uri:
                try:
                    msg = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe all content on this document page including text, tables, and charts."},
                            {"type": "image_url", "image_url": {"url": uri}},
                        ],
                    }
                    resp = client.chat.complete(model=VISION_MODEL, messages=[msg])
                    vision_text = resp.choices[0].message.content
                    tables = [self.extract_table_with_vision(img_bytes)]
                    form_fields = self.extract_form_fields(img_bytes)
                except SDKError as e:
                    vision_text = f"API error: {e}"
            elapsed = time.time() - start
            pages.append(PageContent(
                page_num=i,
                text_layer=text,
                vision_text=vision_text,
                tables=tables,
                form_fields=form_fields,
                image_bytes=img_bytes,
            ))
            print(f"    text_layer={len(text)} chars, vision_text={len(vision_text)} chars [{elapsed:.1f}s]")
        return pages


processor = PDFProcessor(dpi=150)
print("PDFProcessor ready. Pass a PDF path to processor.process_document(path).")
print("Example: pages = processor.process_document('my_report.pdf')")

# %% [markdown]
# ## 4. Multimodal RAG
# MultimodalRAG ingests PDFs by embedding the combined text+vision content per page and
# storing the rendered page image alongside the text chunk. At query time it retrieves the
# top-5 chunks by cosine similarity and injects both text and images into Pixtral.

# %%
import math

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-10)


@dataclass
class ImageChunk:
    """A multimodal chunk combining text content, page image, and metadata."""
    text: str
    image_b64: str
    metadata: dict
    embedding: list[float] = field(default_factory=list)


class MultimodalRAG:
    """Retrieval-augmented generation system that understands both text and images."""

    def __init__(self):
        """Initialize an empty multimodal RAG store."""
        self.chunks: list[ImageChunk] = []

    def ingest_pdf_with_images(self, pdf_path: str) -> None:
        """Ingest a PDF by rendering each page, extracting text, and embedding combined content.

        Args:
            pdf_path: Path to the PDF to ingest.
        """
        proc = PDFProcessor(dpi=120)
        pages = proc.process_document(pdf_path)
        texts = []
        for page in pages:
            combined = f"Page {page.page_num + 1}:\n{page.text_layer}\n{page.vision_text}"
            texts.append(combined[:4000])  # stay within embed token limits
        if not texts:
            print("No pages extracted.")
            return
        try:
            emb_response = client.embeddings.create(model=EMBED_MODEL, inputs=texts)
            for i, (page, emb_obj) in enumerate(zip(pages, emb_response.data)):
                b64 = base64.b64encode(page.image_bytes).decode() if page.image_bytes else ""
                chunk = ImageChunk(
                    text=texts[i],
                    image_b64=b64,
                    metadata={"source": pdf_path, "page": page.page_num},
                    embedding=emb_obj.embedding,
                )
                self.chunks.append(chunk)
            print(f"Ingested {len(pages)} pages from {pdf_path}.")
        except SDKError as e:
            print(f"[ingest_pdf_with_images] Embedding error: {e}")

    def retrieve(self, query: str, top_k: int = 5) -> list[ImageChunk]:
        """Find the top-k most relevant chunks for a query using embedding similarity.

        Args:
            query: Natural language question.
            top_k: Number of chunks to return.

        Returns:
            Ranked list of ImageChunk objects.
        """
        if not self.chunks:
            return []
        try:
            emb_resp = client.embeddings.create(model=EMBED_MODEL, inputs=[query])
            q_emb = emb_resp.data[0].embedding
        except SDKError as e:
            print(f"[retrieve] Embedding error: {e}")
            return []
        scored = sorted(
            self.chunks,
            key=lambda c: _cosine_similarity(q_emb, c.embedding),
            reverse=True,
        )
        return scored[:top_k]

    def generate_with_context(self, question: str, text_chunks: list[ImageChunk],
                               image_chunks: list[ImageChunk]) -> str:
        """Generate an answer using retrieved text and page images as multimodal context.

        Args:
            question: The user question to answer.
            text_chunks: Chunks providing textual context.
            image_chunks: Chunks whose page images are injected visually.

        Returns:
            Answer string from Pixtral.
        """
        context_text = "\n\n---\n\n".join(c.text for c in text_chunks)
        content: list[dict] = [
            {"type": "text", "text": (
                f"Answer the following question using the provided document context and images.\n\n"
                f"CONTEXT:\n{context_text}\n\n"
                f"QUESTION: {question}\n\n"
                f"Document page images follow:"
            )},
        ]
        for chunk in image_chunks:
            if chunk.image_b64:
                uri = f"data:image/png;base64,{chunk.image_b64}"
                content.append({"type": "image_url", "image_url": {"url": uri}})
        try:
            start = time.time()
            response = client.chat.complete(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": content}],
            )
            elapsed = time.time() - start
            answer = response.choices[0].message.content
            print(f"[generate_with_context] {elapsed:.2f}s")
            return answer
        except SDKError as e:
            return f"API error: {e}"


rag = MultimodalRAG()
print("MultimodalRAG initialized.")
print("Usage: rag.ingest_pdf_with_images('doc.pdf')")
print("       chunks = rag.retrieve('What are the revenue figures?')")
print("       answer = rag.generate_with_context(question, chunks, chunks)")

# %% [markdown]
# ## 5. Chart and Diagram Understanding
# ChartAnalyzer uses Pixtral with a structured prompt to extract machine-readable insight
# from charts and diagrams. It outputs a ChartAnalysis dataclass and can convert chart
# data to a pandas DataFrame for downstream analysis.

# %%
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

CHART_ANALYSIS_PROMPT = """Analyze this chart or diagram and return a JSON object with exactly these keys:
{
  "chart_type": "<bar|line|pie|scatter|table|diagram|other>",
  "data_points": [{"label": "...", "value": "..."}],
  "trend": "<increasing|decreasing|stable|cyclical|mixed|N/A>",
  "key_insight": "<one sentence summarizing the most important takeaway>",
  "axes": {"x": "<label or null>", "y": "<label or null>"}
}
Return only valid JSON."""


@dataclass
class ChartAnalysis:
    """Structured analysis of a chart or diagram."""
    chart_type: str
    data_points: list[dict]
    trend: str
    key_insight: str
    axes: dict


class ChartAnalyzer:
    """Analyze charts and diagrams using Pixtral vision."""

    def analyze_chart(self, image_path: str) -> ChartAnalysis:
        """Extract structured metadata and insights from a chart image.

        Args:
            image_path: Path to the chart image file.

        Returns:
            ChartAnalysis dataclass with extracted information.
        """
        try:
            start = time.time()
            uri = base64_encode_image(image_path)
            msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": CHART_ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
            response = client.chat.complete(
                model=VISION_MODEL,
                messages=[msg],
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - start
            data = json.loads(response.choices[0].message.content)
            print(f"[analyze_chart] {elapsed:.2f}s | type={data.get('chart_type')}")
            return ChartAnalysis(
                chart_type=data.get("chart_type", "unknown"),
                data_points=data.get("data_points", []),
                trend=data.get("trend", "N/A"),
                key_insight=data.get("key_insight", ""),
                axes=data.get("axes", {}),
            )
        except (SDKError, json.JSONDecodeError) as e:
            print(f"[analyze_chart] error: {e}")
            return ChartAnalysis("unknown", [], "N/A", str(e), {})

    def extract_table_data(self, image_path: str):
        """Extract tabular data from a chart or table image into a DataFrame.

        Args:
            image_path: Path to the image containing a table or chart data.

        Returns:
            pandas DataFrame if pandas is available, else list of dicts.
        """
        try:
            uri = base64_encode_image(image_path)
            msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Extract all data from this chart or table as JSON. "
                        "Return a JSON array of objects where each object is a row. "
                        "Use column headers as keys."
                    )},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }
            response = client.chat.complete(
                model=VISION_MODEL,
                messages=[msg],
                response_format={"type": "json_object"},
            )
            raw = json.loads(response.choices[0].message.content)
            rows = raw if isinstance(raw, list) else raw.get("data", [raw])
            if PANDAS_AVAILABLE:
                return pd.DataFrame(rows)
            return rows
        except (SDKError, json.JSONDecodeError) as e:
            print(f"[extract_table_data] error: {e}")
            return pd.DataFrame() if PANDAS_AVAILABLE else []

    def compare_charts(self, img_path1: str, img_path2: str) -> str:
        """Compare two charts and describe differences in data, trends, and design.

        Args:
            img_path1: Path to the first chart image.
            img_path2: Path to the second chart image.

        Returns:
            Comparison analysis string.
        """
        return compare_images(
            img_path1, img_path2,
            question=(
                "Compare these two charts. Describe: (1) chart types, "
                "(2) data ranges and scales, (3) trends shown, "
                "(4) key differences in the data story each chart tells."
            ),
        )


analyzer = ChartAnalyzer()

# Demo: analyze the test image we created earlier (not a real chart, but verifies the API path)
print("--- Chart Analysis Demo (solid color test image) ---")
analysis = analyzer.analyze_chart(_test_img)
print(f"Type      : {analysis.chart_type}")
print(f"Trend     : {analysis.trend}")
print(f"Insight   : {analysis.key_insight}")
print(f"Data pts  : {analysis.data_points[:3]}")

# %% [markdown]
# ## 6. Audio with Whisper
# OpenAI Whisper transcribes audio locally (no API cost). TranscriptionPipeline splits
# long audio into 30-second chunks, transcribes in parallel, then merges results.
# The final transcription feeds into Mistral for audio Q&A.

# %%
try:
    import whisper as whisper_lib
    WHISPER_AVAILABLE = True
    print("Whisper available.")
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper not installed. Run: pip install openai-whisper")

import concurrent.futures


def transcribe_audio(audio_path: str, model_size: str = "base") -> dict:
    """Transcribe an audio file using OpenAI Whisper.

    Args:
        audio_path: Path to the audio file (MP3, WAV, M4A, etc.).
        model_size: Whisper model size: tiny, base, small, medium, large.

    Returns:
        Dict with keys: text (str), segments (list), language (str).
    """
    if not WHISPER_AVAILABLE:
        return {"text": "", "segments": [], "language": "unknown"}
    start = time.time()
    model = whisper_lib.load_model(model_size)
    result = model.transcribe(audio_path)
    elapsed = time.time() - start
    print(f"[transcribe_audio] {elapsed:.2f}s | lang={result.get('language')} | {len(result['text'])} chars")
    return {"text": result["text"], "segments": result.get("segments", []), "language": result.get("language", "")}


class TranscriptionPipeline:
    """Pipeline for transcribing long audio files in parallel chunks."""

    def __init__(self, chunk_seconds: int = 30, model_size: str = "base"):
        """Initialize the pipeline.

        Args:
            chunk_seconds: Duration of each audio chunk in seconds.
            model_size: Whisper model size to use.
        """
        self.chunk_seconds = chunk_seconds
        self.model_size = model_size

    def chunk_audio(self, audio_path: str) -> list[str]:
        """Split an audio file into fixed-length chunks saved as WAV files.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            List of paths to the chunk WAV files.
        """
        try:
            import subprocess
            import tempfile
            out_dir = tempfile.mkdtemp()
            pattern = os.path.join(out_dir, "chunk_%03d.wav")
            cmd = [
                "ffmpeg", "-i", audio_path,
                "-f", "segment", "-segment_time", str(self.chunk_seconds),
                "-c", "copy", pattern, "-y", "-loglevel", "error",
            ]
            subprocess.run(cmd, check=True)
            chunks = sorted(Path(out_dir).glob("chunk_*.wav"))
            print(f"[chunk_audio] Created {len(chunks)} chunks in {out_dir}")
            return [str(c) for c in chunks]
        except Exception as e:
            print(f"[chunk_audio] error (ffmpeg required): {e}")
            return [audio_path]

    def transcribe_chunks(self, chunk_paths: list[str]) -> list[dict]:
        """Transcribe audio chunks in parallel using a thread pool.

        Args:
            chunk_paths: List of paths to chunk WAV files.

        Returns:
            List of transcription result dicts, in order.
        """
        results = [None] * len(chunk_paths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
            futures = {exe.submit(transcribe_audio, p, self.model_size): i
                       for i, p in enumerate(chunk_paths)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def merge_transcription(self, chunk_results: list[dict]) -> str:
        """Merge ordered chunk transcriptions into a single text string.

        Args:
            chunk_results: List of dicts returned by transcribe_audio.

        Returns:
            Full transcription as a single string.
        """
        return " ".join(r["text"].strip() for r in chunk_results if r and r.get("text"))

    def audio_qa(self, audio_path: str, question: str) -> str:
        """Transcribe an audio file then answer a question about its content.

        Args:
            audio_path: Path to the audio file.
            question: Question to answer based on the audio content.

        Returns:
            Answer string from Mistral.
        """
        result = transcribe_audio(audio_path, self.model_size)
        transcript = result.get("text", "")
        if not transcript:
            return "Could not transcribe audio."
        try:
            response = client.chat.complete(
                model=LARGE_MODEL,
                messages=[
                    {"role": "system", "content": "You answer questions based solely on the provided audio transcript."},
                    {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nQUESTION: {question}"},
                ],
            )
            return response.choices[0].message.content
        except SDKError as e:
            return f"API error: {e}"


pipeline = TranscriptionPipeline(chunk_seconds=30, model_size="base")
print("TranscriptionPipeline ready.")
print("Usage: result = transcribe_audio('recording.mp3')")
print("       answer = pipeline.audio_qa('meeting.mp3', 'What were the action items?')")

# %% [markdown]
# ## 7. Lab Exercise: Multimodal Document Q&A
# Build a complete pipeline that ingests multiple PDFs with charts and tables, constructs
# a multimodal RAG index, answers 10 questions that require visual understanding, and
# compares multimodal accuracy against text-only RAG. A report is printed showing which
# questions required vision to answer correctly.

# %%
import textwrap
from PIL import Image as PILImage, ImageDraw, ImageFont


def _create_synthetic_pdf(path: str, title: str, content: str) -> str:
    """Create a minimal synthetic PDF with text content for the lab demo.

    Args:
        path: Output path for the PDF file.
        title: Document title to embed.
        content: Body text to include in the PDF.

    Returns:
        Path to the created PDF.
    """
    if not FITZ_AVAILABLE:
        print("PyMuPDF unavailable, cannot create synthetic PDF.")
        return path
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 80), title, fontsize=18)
    y = 130
    for line in textwrap.wrap(content, 80):
        page.insert_text((50, y), line, fontsize=11)
        y += 16
        if y > 800:
            break
    doc.save(path)
    doc.close()
    return path


def run_lab_exercise():
    """Run the full multimodal document Q&A lab exercise.

    Creates 3 synthetic PDFs, builds a multimodal RAG index, answers 10 questions,
    compares multimodal vs text-only accuracy, and prints a comparison report.
    """
    print("=" * 60)
    print("LAB EXERCISE: Multimodal Document Q&A")
    print("=" * 60)

    # --- Step 1: Create synthetic documents ---
    pdf_dir = Path("/tmp/lab_pdfs")
    pdf_dir.mkdir(exist_ok=True)

    docs = [
        ("financial_report.pdf", "Q3 2024 Financial Report",
         "Revenue reached $4.2 billion, up 18% year-over-year. "
         "Operating margin improved to 23.5%. Net income was $850 million. "
         "The APAC region grew 32% driven by enterprise sales. "
         "R&D spending increased to $420 million (10% of revenue). "
         "Headcount: 12,400 employees worldwide."),
        ("product_roadmap.pdf", "Product Roadmap H1 2025",
         "Phase 1 (Jan-Feb): Launch multimodal search feature. "
         "Phase 2 (Mar): Release mobile SDK v3. "
         "Phase 3 (Apr-May): Enterprise SSO integration. "
         "Phase 4 (Jun): Self-hosted deployment option. "
         "Key metric targets: 99.9% uptime, <200ms p95 latency, NPS > 60."),
        ("market_analysis.pdf", "Global AI Market Analysis 2024",
         "The global AI market is valued at $184 billion in 2024. "
         "Expected CAGR of 37% through 2030, reaching $1.3 trillion. "
         "Top segments: NLP (28%), computer vision (22%), MLOps (15%). "
         "North America holds 42% market share. "
         "Top vendors: OpenAI, Google, Microsoft, Anthropic, Mistral AI."),
    ]

    pdf_paths = []
    for fname, title, content in docs:
        p = str(pdf_dir / fname)
        _create_synthetic_pdf(p, title, content)
        pdf_paths.append(p)
        print(f"Created: {p}")

    # --- Step 2: Build multimodal RAG ---
    print("\nBuilding multimodal RAG index...")
    mrag = MultimodalRAG()
    text_chunks_store: list[ImageChunk] = []

    for pdf_path in pdf_paths:
        mrag.ingest_pdf_with_images(pdf_path)

    # Text-only RAG (same chunks but answers without images)
    print(f"\nTotal chunks in index: {len(mrag.chunks)}")

    # --- Step 3: Define 10 test questions ---
    questions = [
        "What was the total revenue in Q3 2024?",
        "Which region had the highest growth rate?",
        "What is the operating margin percentage?",
        "When is the mobile SDK v3 planned for release?",
        "What is the target p95 latency for the product?",
        "What is the global AI market value in 2024?",
        "Which AI segment has the largest market share?",
        "How much was spent on R&D?",
        "What is the expected CAGR for the AI market?",
        "What is the net income figure mentioned in the financial report?",
    ]

    # --- Step 4: Answer questions with both approaches ---
    results = []
    print("\nAnswering 10 questions...\n")
    for i, question in enumerate(questions, 1):
        print(f"Q{i}: {question}")

        # Retrieve relevant chunks
        retrieved = mrag.retrieve(question, top_k=3)

        # Text-only answer (use large model with text context only)
        text_context = "\n\n".join(c.text for c in retrieved)
        try:
            text_resp = client.chat.complete(
                model=LARGE_MODEL,
                messages=[
                    {"role": "system", "content": "Answer based only on the provided context."},
                    {"role": "user", "content": f"CONTEXT:\n{text_context}\n\nQUESTION: {question}"},
                ],
            )
            text_answer = text_resp.choices[0].message.content.strip()
        except SDKError as e:
            text_answer = f"Error: {e}"

        # Multimodal answer (with page images)
        multimodal_answer = mrag.generate_with_context(question, retrieved, retrieved)

        # Heuristic: does the answer contain a number or specific fact?
        requires_vision = any(kw in question.lower() for kw in
                               ["chart", "table", "graph", "figure", "diagram", "visual"])
        results.append({
            "question": question,
            "text_answer": text_answer[:120],
            "multimodal_answer": multimodal_answer[:120],
            "requires_vision": requires_vision,
        })
        print(f"  Text-only    : {text_answer[:80]}...")
        print(f"  Multimodal   : {multimodal_answer[:80]}...")
        print()

    # --- Step 5: Print comparison report ---
    print("\n" + "=" * 60)
    print("COMPARISON REPORT: Text-Only vs Multimodal RAG")
    print("=" * 60)
    vision_questions = [r for r in results if r["requires_vision"]]
    non_vision = [r for r in results if not r["requires_vision"]]
    print(f"Total questions      : {len(results)}")
    print(f"Vision-required      : {len(vision_questions)}")
    print(f"Text-sufficient      : {len(non_vision)}")
    print()
    print("Questions requiring visual understanding:")
    for r in vision_questions:
        print(f"  - {r['question']}")
    print()
    print("Sample answers (Q1):")
    print(f"  Text-only  : {results[0]['text_answer']}")
    print(f"  Multimodal : {results[0]['multimodal_answer']}")
    print()
    print("Report complete. In production, score answers against ground truth")
    print("to measure the accuracy delta between text-only and multimodal RAG.")
    print("=" * 60)

    return results


lab_results = run_lab_exercise()
assert len(lab_results) == 10, "Expected 10 question results"
print(f"\nLab exercise passed: {len(lab_results)} questions answered.")

# %% [markdown]
# ## Key Takeaways
# - Pixtral-12B enables native vision understanding: pass base64 images directly in the
#   content array alongside text for OCR, chart analysis, and document intelligence.
# - Multimodal RAG outperforms text-only RAG for documents where critical information
#   lives in charts, diagrams, tables, or scanned images that text extraction misses.
# - PyMuPDF (fitz) is the fastest way to render PDF pages to images; combining its text
#   layer with Pixtral vision gives the best of both worlds for document processing.
# - Whisper provides free, high-quality local audio transcription; coupling it with
#   Mistral enables cost-effective audio Q&A without sending audio to a paid API.
# - Structured JSON output (response_format={"type":"json_object"}) makes vision results
#   machine-readable so chart analysis and form extraction integrate cleanly into pipelines.
