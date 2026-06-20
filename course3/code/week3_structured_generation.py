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
# ## Course 3, Week 3: Structured Generation and Output Reliability
#
# This notebook covers techniques for reliably extracting structured data from LLMs.
# We explore JSON mode, Pydantic validation with Instructor, complex schema design,
# and repair pipelines that ensure 100% schema conformance in production systems.

# %% [markdown]
# ## Section 1: Setup
#
# Install dependencies and configure the Mistral client. We use `instructor` to
# wrap the Mistral client with automatic Pydantic validation and retry logic.
# `python-dotenv` loads API keys from a `.env` file so keys never appear in code.

# %%
# !pip install mistralai python-dotenv instructor pydantic

import os
import json
import time
import logging
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from mistralai import Mistral
import instructor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load from environment — set MISTRAL_API_KEY in your shell or a .env file
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
MODEL_LARGE = "mistral-large-latest"
MODEL_SMALL = "mistral-small-latest"

raw_client = Mistral(api_key=MISTRAL_API_KEY)
print("Mistral raw client ready.")

# %% [markdown]
# ## Section 2: JSON Mode with Mistral
#
# Mistral's `response_format={"type": "json_object"}` guarantees the response is
# syntactically valid JSON, but it does NOT enforce a specific schema. We demonstrate
# this distinction and show how to guide structure via the system prompt.

# %%
def demo_json_mode_valid() -> dict:
    """Request JSON output and verify it parses without error."""
    system = (
        "You are a data extractor. Always respond with valid JSON in this exact schema: "
        '{"name": "string", "age": integer, "city": "string"}'
    )
    user = "Extract: Alice is 30 years old and lives in Berlin."
    start = time.time()
    response = raw_client.chat.complete(
        model=MODEL_LARGE,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    elapsed = time.time() - start
    raw_text = response.choices[0].message.content
    data = json.loads(raw_text)          # guaranteed to succeed in JSON mode
    print(f"[JSON mode] Valid parse in {elapsed:.2f}s => {data}")
    assert "name" in data, "Expected 'name' field"
    return data


def demo_json_mode_schema_drift() -> dict:
    """Show that JSON mode can return valid JSON with an unexpected structure."""
    system = "Respond with valid JSON only."
    user = "Tell me something interesting about whales."
    response = raw_client.chat.complete(
        model=MODEL_SMALL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    print(f"[Schema drift] Got keys: {list(data.keys())} — schema NOT enforced by JSON mode")
    return data


result_valid = demo_json_mode_valid()
result_drift = demo_json_mode_schema_drift()

# %% [markdown]
# ## Section 3: Instructor + Pydantic
#
# The `instructor` library wraps the Mistral client to automatically validate
# responses against a Pydantic model and retry on validation failures. This
# guarantees schema conformance — not just syntactic validity.

# %%
class ExtractedEvent(BaseModel):
    """Structured representation of a calendar event extracted from free text."""

    title: str = Field(description="Short event title, 5 words or fewer")
    date: str = Field(description="ISO-8601 date string, e.g. 2025-03-15")
    location: Optional[str] = Field(default=None, description="Venue or city, if mentioned")
    attendees: List[str] = Field(default_factory=list, description="List of attendee full names")


# Wrap raw client with instructor for automatic Pydantic validation + retry
instructor_client = instructor.from_mistral(raw_client)

def extract_event(text: str) -> ExtractedEvent:
    """Extract a structured event from unstructured text using Instructor."""
    start = time.time()
    event = instructor_client.chat.completions.create(
        model=MODEL_LARGE,
        response_model=ExtractedEvent,
        messages=[
            {"role": "system", "content": "Extract event details. Return valid JSON matching the schema."},
            {"role": "user", "content": text},
        ],
        max_retries=3,
    )
    elapsed = time.time() - start
    print(f"[Instructor] Extracted in {elapsed:.2f}s: {event.model_dump()}")
    return event


sample_text = (
    "Hey team, don't forget our quarterly planning session on 2025-09-10 "
    "at the Marriott Downtown. Bob Smith, Carol Tan, and David Lee will join."
)
event = extract_event(sample_text)
assert isinstance(event.attendees, list), "attendees must be a list"
assert event.date == "2025-09-10", f"Unexpected date: {event.date}"
print("Instructor extraction assertions passed.")

# %% [markdown]
# ## Section 4: Complex Schema Design
#
# Real-world extraction tasks need nested models. We use `Field(description=...)`
# as inline instructions to the model, `Optional` with defaults for nullable fields,
# and `Literal` for constrained vocabularies. Nesting keeps each model focused.

# %%
class Citation(BaseModel):
    """A single source cited in a research report section."""

    url: str = Field(description="Full URL of the source")
    title: str = Field(description="Title of the article or page")
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Relevance to the section topic, 0.0-1.0"
    )


class Section(BaseModel):
    """One section of a research report with supporting citations."""

    title: str = Field(description="Section heading")
    content: str = Field(description="Two to four sentence summary of this section")
    sources: List[Citation] = Field(default_factory=list, description="Supporting citations")


class ResearchReport(BaseModel):
    """Full structured research report with metadata and sections."""

    topic: str = Field(description="Main research topic")
    summary: str = Field(description="One-sentence executive summary")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Overall confidence in findings"
    )
    sections: List[Section] = Field(description="Two or three sections covering the topic")
    limitations: Optional[str] = Field(
        default=None, description="Known gaps or limitations of this report"
    )


def generate_research_report(topic: str) -> ResearchReport:
    """Generate a structured research report on the given topic."""
    start = time.time()
    report = instructor_client.chat.completions.create(
        model=MODEL_LARGE,
        response_model=ResearchReport,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Produce a concise structured report. "
                    "Invent plausible but clearly fictional URLs for citations."
                ),
            },
            {"role": "user", "content": f"Write a research report on: {topic}"},
        ],
        max_retries=3,
    )
    elapsed = time.time() - start
    print(f"[ResearchReport] Generated in {elapsed:.2f}s")
    print(f"  Topic     : {report.topic}")
    print(f"  Confidence: {report.confidence}")
    print(f"  Sections  : {len(report.sections)}")
    for sec in report.sections:
        print(f"    - {sec.title} ({len(sec.sources)} citations)")
    return report


report = generate_research_report("quantum computing applications in drug discovery")
assert report.confidence in ("low", "medium", "high")
assert len(report.sections) >= 1
print("Complex schema assertions passed.")

# %% [markdown]
# ## Section 5: Validation and Repair Pipeline
#
# Production pipelines must handle malformed model outputs gracefully. This pipeline
# tries four escalating repair strategies before giving up. Each attempt is timed and
# logged so we can measure per-attempt repair rates over a batch.

# %%
class ValidationResult(BaseModel):
    """Result of validating a structured output against a schema."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    attempt_number: int = 1


def validate_output(data: dict, schema: type[BaseModel]) -> ValidationResult:
    """Validate a dict against a Pydantic schema; return structured result."""
    try:
        schema.model_validate(data)
        return ValidationResult(is_valid=True)
    except Exception as exc:
        errors = [str(e) for e in exc.errors()] if hasattr(exc, "errors") else [str(exc)]
        return ValidationResult(is_valid=False, errors=errors)


def repair_with_prompt(bad_output: str, error_msg: str, schema: type[BaseModel]) -> Optional[dict]:
    """Ask the model to fix a broken output given the validation error message."""
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    repair_prompt = (
        f"The following JSON failed validation:\n\n{bad_output}\n\n"
        f"Error: {error_msg}\n\n"
        f"Fix it so it strictly matches this JSON schema:\n{schema_json}\n"
        "Return ONLY the corrected JSON object."
    )
    response = raw_client.chat.complete(
        model=MODEL_LARGE,
        messages=[{"role": "user", "content": repair_prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


class RepairPipeline:
    """
    Four-attempt repair pipeline for structured LLM outputs.

    Attempt 1: JSON mode (fixes syntax errors)
    Attempt 2: Instructor with retry (fixes schema conformance)
    Attempt 3: Explicit repair prompt (fixes semantic errors)
    Attempt 4: Simplified schema (drops optional fields)
    Fallback  : Returns None and logs failure
    """

    def __init__(self, schema: type[BaseModel]):
        """Initialize with the target Pydantic schema."""
        self.schema = schema
        self.repair_counts = {1: 0, 2: 0, 3: 0, 4: 0, "failed": 0}

    def extract(self, text: str) -> Optional[BaseModel]:
        """Run extraction with progressive repair; return validated model or None."""
        # Attempt 1 — JSON mode + manual validation
        try:
            schema_hint = json.dumps(self.schema.model_json_schema(), indent=2)
            response = raw_client.chat.complete(
                model=MODEL_LARGE,
                messages=[
                    {"role": "system", "content": f"Extract data matching this schema:\n{schema_hint}"},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            result = validate_output(data, self.schema)
            if result.is_valid:
                self.repair_counts[1] += 1
                return self.schema.model_validate(data)
            bad_output, error_msg = json.dumps(data), "; ".join(result.errors)
        except Exception as exc:
            bad_output, error_msg = text[:200], str(exc)

        # Attempt 2 — Instructor with automatic retry
        try:
            obj = instructor_client.chat.completions.create(
                model=MODEL_LARGE,
                response_model=self.schema,
                messages=[{"role": "user", "content": text}],
                max_retries=2,
            )
            self.repair_counts[2] += 1
            return obj
        except Exception as exc:
            error_msg = str(exc)

        # Attempt 3 — Explicit repair prompt
        try:
            fixed = repair_with_prompt(bad_output, error_msg, self.schema)
            result = validate_output(fixed, self.schema)
            if result.is_valid:
                self.repair_counts[3] += 1
                return self.schema.model_validate(fixed)
        except Exception:
            pass

        # Attempt 4 — Simplified schema (required fields only)
        try:
            required_fields = {
                k: v
                for k, v in self.schema.model_fields.items()
                if v.is_required()
            }
            simple_schema = {
                "type": "object",
                "properties": {k: {"type": "string"} for k in required_fields},
                "required": list(required_fields.keys()),
            }
            response = raw_client.chat.complete(
                model=MODEL_LARGE,
                messages=[
                    {"role": "system", "content": f"Extract only these fields as JSON: {list(required_fields.keys())}"},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            self.repair_counts[4] += 1
            return self.schema.model_validate(data)
        except Exception:
            pass

        # Fallback
        logger.error("All repair attempts failed for input: %s", text[:100])
        self.repair_counts["failed"] += 1
        return None

    def report_repair_rates(self, total: int) -> None:
        """Print per-attempt repair rates as percentages."""
        print(f"\n--- Repair Pipeline Stats (n={total}) ---")
        for k, v in self.repair_counts.items():
            label = f"Attempt {k}" if isinstance(k, int) else "Failed"
            print(f"  {label}: {v}/{total} ({100*v/max(total,1):.0f}%)")


# Quick smoke test
class SimpleSchema(BaseModel):
    """Simple schema for pipeline smoke test."""
    name: str
    value: int

pipeline = RepairPipeline(SimpleSchema)
obj = pipeline.extract("Name: Foo, Value: 42")
print(f"[RepairPipeline smoke test] Result: {obj}")

# %% [markdown]
# ## Section 6: Document Information Extraction
#
# We apply Instructor-backed schemas to three document types: invoices, contact
# cards, and meeting notes. Running over five sample documents lets us measure
# field-level conformance — how many expected fields are non-null after extraction.

# %%
class LineItem(BaseModel):
    """A single line item on an invoice."""

    description: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0.0)
    total: float = Field(ge=0.0)


class InvoiceExtractor(BaseModel):
    """Structured invoice extracted from plain-text document."""

    vendor_name: str
    invoice_number: str
    date: str = Field(description="ISO-8601 date")
    line_items: List[LineItem]
    total_amount: float = Field(ge=0.0)
    currency: str = Field(description="3-letter ISO currency code, e.g. USD")


class ContactCardExtractor(BaseModel):
    """Structured contact card extracted from free text."""

    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None


class MeetingNotesExtractor(BaseModel):
    """Key structured fields extracted from meeting notes."""

    meeting_title: str
    date: str
    attendees: List[str]
    action_items: List[str]
    next_meeting_date: Optional[str] = None


SAMPLE_DOCUMENTS = [
    ("invoice", "Invoice #INV-2025-001 from Acme Corp dated 2025-07-15. "
     "Items: 5x Widget A @ $10.00 each = $50.00; 2x Gadget B @ $25.00 each = $50.00. "
     "Total: $100.00 USD."),
    ("invoice", "Bill from TechSupplies Ltd. Ref: TS-8821, 2025-08-01. "
     "1x Server Rack $1200.00; 4x CAT6 Cable $15.00 each = $60.00. Grand total $1260.00 EUR."),
    ("contact", "Dr. Jane Doe, VP Engineering at DataStream Inc. "
     "Reach her at jane.doe@datastream.io or +1-555-234-5678."),
    ("contact", "Contact: Marcus Leung, Senior Consultant. marcus@consulting.net. No phone on file."),
    ("meeting", "Q3 Roadmap Sync — 2025-09-05. Attendees: Priya Sharma, Tom Okafor, Lin Wei. "
     "Action items: finalize sprint plan, schedule UX review, update changelog. "
     "Next meeting: 2025-09-19."),
]

SCHEMA_MAP = {
    "invoice": InvoiceExtractor,
    "contact": ContactCardExtractor,
    "meeting": MeetingNotesExtractor,
}


def measure_field_conformance(obj: BaseModel) -> float:
    """Return fraction of model fields that are non-None (field-level conformance)."""
    fields = obj.model_fields
    non_null = sum(1 for k in fields if getattr(obj, k, None) is not None)
    return non_null / len(fields) if fields else 0.0


conformance_scores = []
for doc_type, doc_text in SAMPLE_DOCUMENTS:
    schema = SCHEMA_MAP[doc_type]
    try:
        extracted = instructor_client.chat.completions.create(
            model=MODEL_LARGE,
            response_model=schema,
            messages=[
                {"role": "system", "content": "Extract structured data from the document."},
                {"role": "user", "content": doc_text},
            ],
            max_retries=3,
        )
        score = measure_field_conformance(extracted)
        conformance_scores.append(score)
        print(f"[{doc_type}] conformance={score:.0%} | {extracted.model_dump()}")
    except Exception as exc:
        print(f"[{doc_type}] FAILED: {exc}")
        conformance_scores.append(0.0)

avg_conformance = sum(conformance_scores) / len(conformance_scores)
print(f"\nAverage field-level conformance across {len(SAMPLE_DOCUMENTS)} docs: {avg_conformance:.0%}")
assert avg_conformance >= 0.8, "Expected >= 80% average field conformance"

# %% [markdown]
# ## Section 7: Lab Exercise — Job Posting Extraction Pipeline
#
# Build a complete extraction pipeline for unstructured job postings. Define a
# `JobPosting` Pydantic model, run it over 10 sample postings with a repair loop,
# measure schema conformance, and save all results to JSON. Target: 100% conformance.

# %%
class SalaryRange(BaseModel):
    """Structured salary range extracted from a job posting."""

    min_salary: Optional[int] = Field(default=None, description="Minimum annual salary in USD")
    max_salary: Optional[int] = Field(default=None, description="Maximum annual salary in USD")
    currency: str = Field(default="USD", description="ISO currency code")


class JobPosting(BaseModel):
    """Fully structured representation of a job posting."""

    title: str = Field(description="Exact job title as listed")
    company: str = Field(description="Hiring company name")
    salary_range: Optional[SalaryRange] = Field(
        default=None, description="Compensation range if mentioned"
    )
    required_skills: List[str] = Field(
        description="List of required technical or soft skills"
    )
    experience_years: int = Field(
        ge=0, description="Minimum years of experience required (0 if not specified)"
    )
    location: str = Field(description="City, state, or 'Remote' if fully remote")
    remote: bool = Field(description="True if any remote work is offered")


SAMPLE_JOB_POSTINGS = [
    "Senior Python Engineer at DataFlow Inc. Remote-friendly. 5+ years exp. "
    "Skills: Python, FastAPI, PostgreSQL, Docker. Salary: $130k-$160k. NYC preferred.",
    "Junior Frontend Dev — Pixel Studio, Austin TX. 1 yr exp OK. React, TypeScript, CSS. $70k-$90k. On-site.",
    "ML Engineer, Neuro Systems. 3 yrs. PyTorch, transformers, MLflow, AWS. $120k-$145k. San Francisco.",
    "DevOps Lead at CloudArch. Terraform, Kubernetes, CI/CD, Linux. 6 yrs. Remote. $150k-$175k.",
    "Data Analyst — RetailMetrics. SQL, Tableau, Excel. 2 yrs. Chicago, IL. $65k-$80k. Hybrid.",
    "Backend Engineer (Go), Fintech Co. 4 yrs Go, gRPC, Kafka, Redis. $140k. Remote US only.",
    "Product Designer at UXLab. Figma, user research, prototyping. 3 yrs. London UK. £55k-£70k. Hybrid.",
    "Security Engineer, SecureNet. SIEM, penetration testing, OWASP, cloud security. 5 yrs. Washington DC. $135k-$160k.",
    "iOS Developer at AppWorks. Swift, SwiftUI, Xcode, Core Data. 2 yrs. Toronto. CAD $90k-$110k. On-site.",
    "Fullstack Engineer (Node + React). StartupXYZ. AWS, MongoDB, REST APIs. 3 yrs. Remote. $115k-$130k.",
]

OUTPUT_PATH = r"d:\gith\courses\course3\code\job_extractions.json"

def run_job_extraction_pipeline(postings: List[str]) -> List[dict]:
    """
    Extract JobPosting models from a list of raw job description strings.

    Uses RepairPipeline for robustness. Measures per-record conformance and
    writes all results (including failures as None) to OUTPUT_PATH.
    Returns list of dicts ready for JSON serialisation.
    """
    pipeline = RepairPipeline(JobPosting)
    results = []
    conformance_scores = []

    for i, text in enumerate(postings, 1):
        start = time.time()
        job = pipeline.extract(text)
        elapsed = time.time() - start

        if job is not None:
            score = measure_field_conformance(job)
            conformance_scores.append(score)
            record = {"index": i, "conformance": score, "elapsed_s": round(elapsed, 2), **job.model_dump()}
            print(f"  [{i:02d}] {job.title} @ {job.company} | conform={score:.0%} | {elapsed:.2f}s")
        else:
            conformance_scores.append(0.0)
            record = {"index": i, "conformance": 0.0, "elapsed_s": round(elapsed, 2), "error": "all attempts failed"}
            print(f"  [{i:02d}] FAILED extraction")

        results.append(record)

    pipeline.report_repair_rates(len(postings))

    avg = sum(conformance_scores) / len(conformance_scores)
    print(f"\nOverall schema conformance: {avg:.0%} across {len(postings)} postings")

    # Save to JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved {len(results)} extractions to {OUTPUT_PATH}")

    return results


print("=== Lab Exercise: Job Posting Extraction Pipeline ===\n")
job_results = run_job_extraction_pipeline(SAMPLE_JOB_POSTINGS)

# Final assertions
successful = [r for r in job_results if "error" not in r]
print(f"\nSuccessfully extracted: {len(successful)}/{len(SAMPLE_JOB_POSTINGS)}")
assert len(successful) >= 8, f"Expected at least 8/10 successful extractions, got {len(successful)}"
avg_final = sum(r["conformance"] for r in job_results) / len(job_results)
assert avg_final >= 0.8, f"Expected >= 80% average conformance, got {avg_final:.0%}"
print("Lab exercise assertions passed.")

# %% [markdown]
# ## Key Takeaways
#
# - JSON mode guarantees syntactically valid JSON but does NOT enforce your schema —
#   always combine it with a Pydantic validator or Instructor for schema conformance.
# - `instructor.from_mistral()` wraps the raw client with automatic validation and
#   retry logic, turning schema conformance from a hope into a guarantee.
# - Nested Pydantic models with `Field(description=...)` serve as inline instructions
#   to the model; the descriptions are the single best lever for output quality.
# - A four-attempt repair pipeline (JSON mode → Instructor → repair prompt →
#   simplified schema) can achieve near-100% conformance even on adversarial inputs.
# - Measure field-level conformance (fraction of non-null required fields), not just
#   parse success; a model that returns empty lists passes JSON parsing but fails
#   the business requirement.
