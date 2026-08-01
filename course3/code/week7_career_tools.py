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
# # Course 3, Week 7: The AI Engineering Career and Field
#
# This notebook equips you with practical tools for navigating the AI engineering
# career landscape. We build portfolio analyzers, blog post generators, learning
# path advisors, job description analyzers, and open source contribution finders —
# all powered by Mistral AI.

# %% [markdown]
# ## Setup
# Install dependencies and configure the Mistral client. We also import `requests`
# for GitHub API calls and standard library modules for data handling.

# %%
# !pip install mistralai python-dotenv requests

import os
import time
import json
import re
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "your-key-here")
client = Mistral(api_key=MISTRAL_API_KEY)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")  # optional, raises rate limit

def github_headers() -> dict:
    """Return headers for GitHub API requests, including auth token if available."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

print("Setup complete. Mistral client initialized.")

# %% [markdown]
# ## Portfolio Analyzer
# A recruiter's first stop is your GitHub profile. The `GitHubPortfolioAnalyzer`
# fetches your repositories, scores each README for quality signals, and calls
# Mistral to generate concrete improvement suggestions.

# %%
import requests

@dataclass
class ReadmeScore:
    """Quality assessment of a single README file."""
    has_demo: bool
    has_architecture_diagram: bool
    has_metrics: bool
    score: float  # 0.0 – 1.0


@dataclass
class PortfolioReport:
    """Aggregated quality report across all repositories."""
    username: str
    total_repos: int
    ai_repos: int
    avg_readme_score: float
    top_repos: list
    weakest_repo: Optional[str]
    improvement_suggestions: list = field(default_factory=list)


AI_LANGUAGES = {"Python", "Jupyter Notebook"}
AI_TOPICS = {"machine-learning", "deep-learning", "nlp", "llm", "ai", "neural-network",
             "transformers", "pytorch", "tensorflow", "reinforcement-learning"}


class GitHubPortfolioAnalyzer:
    """Analyze a GitHub user's public repositories for AI engineering portfolio quality."""

    def fetch_repos(self, username: str) -> list:
        """Fetch public repos for a GitHub user via the GitHub REST API."""
        url = f"https://api.github.com/users/{username}/repos"
        params = {"per_page": 50, "sort": "updated"}
        try:
            resp = requests.get(url, headers=github_headers(), params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            print(f"GitHub API error: {exc}")
            return []

    def analyze_readme(self, readme_text: str) -> ReadmeScore:
        """Score a README based on presence of demo, diagrams, and performance metrics."""
        text_lower = readme_text.lower()
        has_demo = any(kw in text_lower for kw in ["demo", "live", "deploy", "try it", "colab"])
        has_arch = any(kw in text_lower for kw in ["architecture", "diagram", "mermaid", "drawio", "!["])
        has_metrics = any(kw in text_lower for kw in ["accuracy", "f1", "bleu", "rouge", "benchmark", "%"])
        score = (has_demo * 0.35) + (has_arch * 0.35) + (has_metrics * 0.30)
        return ReadmeScore(has_demo=has_demo, has_architecture_diagram=has_arch,
                           has_metrics=has_metrics, score=score)

    def _is_ai_repo(self, repo: dict) -> bool:
        """Return True if the repo appears to be AI/ML related by language or topics."""
        lang = repo.get("language") or ""
        topics = set(repo.get("topics") or [])
        return lang in AI_LANGUAGES or bool(topics & AI_TOPICS)

    def score_portfolio(self, repos: list) -> PortfolioReport:
        """Compute aggregate portfolio quality metrics across all repos."""
        if not repos:
            return PortfolioReport(username="unknown", total_repos=0, ai_repos=0,
                                   avg_readme_score=0.0, top_repos=[], weakest_repo=None)
        ai_repos = [r for r in repos if self._is_ai_repo(r)]
        scores = []
        for repo in repos:
            readme_url = f"https://raw.githubusercontent.com/{repo['full_name']}/HEAD/README.md"
            try:
                r = requests.get(readme_url, timeout=6)
                text = r.text if r.status_code == 200 else ""
            except requests.RequestException:
                text = ""
            rs = self.analyze_readme(text)
            scores.append((repo["name"], rs.score))

        scores.sort(key=lambda x: x[1], reverse=True)
        avg = sum(s for _, s in scores) / len(scores)
        top = [name for name, _ in scores[:3]]
        weakest = scores[-1][0] if scores else None

        return PortfolioReport(
            username=repos[0]["owner"]["login"],
            total_repos=len(repos),
            ai_repos=len(ai_repos),
            avg_readme_score=round(avg, 3),
            top_repos=top,
            weakest_repo=weakest,
        )

    def generate_improvement_suggestions(self, report: PortfolioReport) -> list:
        """Call Mistral to produce 5 actionable portfolio improvement suggestions."""
        prompt = (
            f"GitHub portfolio analysis for @{report.username}:\n"
            f"- {report.total_repos} total repos, {report.ai_repos} AI-related\n"
            f"- Average README score: {report.avg_readme_score:.0%}\n"
            f"- Top repos: {', '.join(report.top_repos)}\n"
            f"- Weakest repo: {report.weakest_repo}\n\n"
            "List exactly 5 concise, actionable improvements an AI engineer should make "
            "to strengthen this portfolio. Return JSON: {\"suggestions\": [\"...\", ...]}"
        )
        try:
            start = time.time()
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - start
            data = json.loads(response.choices[0].message.content)
            suggestions = data.get("suggestions", [])
            print(f"Improvement suggestions generated in {elapsed:.2f}s")
            return suggestions
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return ["Unable to generate suggestions — check your API key."]


# Demo with a real (or mocked) username
analyzer = GitHubPortfolioAnalyzer()
sample_repos = [
    {"name": "llm-rag-demo", "full_name": "octocat/llm-rag-demo", "language": "Python",
     "topics": ["llm", "rag"], "owner": {"login": "octocat"}, "stargazers_count": 42},
    {"name": "hello-world", "full_name": "octocat/hello-world", "language": "JavaScript",
     "topics": [], "owner": {"login": "octocat"}, "stargazers_count": 1},
]
report = analyzer.score_portfolio(sample_repos)
print(f"Portfolio report: {report}")
suggestions = analyzer.generate_improvement_suggestions(report)
print("Suggestions:")
for i, s in enumerate(suggestions, 1):
    print(f"  {i}. {s}")
assert len(suggestions) > 0, "Expected at least one suggestion"

# %% [markdown]
# ## Technical Blog Post Generator
# Consistent technical writing builds domain authority. `BlogPostGenerator` uses
# Mistral to outline, expand, and assemble a ~1200-word post, and generates
# Mermaid diagrams to illustrate architecture decisions.

# %%
@dataclass
class BlogPost:
    """Structured representation of a generated technical blog post."""
    title: str
    outline: list
    sections: dict
    mermaid_diagram: str
    word_count: int
    full_text: str


class BlogPostGenerator:
    """Generate structured technical blog posts for AI engineering topics."""

    def generate_outline(self, topic: str, project_description: str) -> list:
        """Use Mistral to create a 5-section blog post outline."""
        prompt = (
            f"Create a 5-section outline for a technical blog post titled '{topic}'.\n"
            f"Project context: {project_description}\n"
            "Each section should have a title and 3 key points.\n"
            'Return JSON: {"sections": [{"title": "...", "key_points": ["...", "...", "..."]}]}'
        )
        try:
            start = time.time()
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - start
            data = json.loads(response.choices[0].message.content)
            print(f"Outline generated in {elapsed:.2f}s")
            return data.get("sections", [])
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return []

    def expand_section(self, section_title: str, key_points: list) -> str:
        """Expand a single outline section into ~200 words of prose."""
        prompt = (
            f"Write a ~200-word technical blog section titled '{section_title}'.\n"
            f"Cover these points: {'; '.join(key_points)}\n"
            "Write for a senior software engineer audience. Be specific and concise."
        )
        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return f"[Section '{section_title}' — generation failed]"

    def add_code_snippets(self, section: str, language: str) -> str:
        """Append a relevant code snippet to a section using Codestral."""
        prompt = (
            f"Given this blog section:\n\n{section[:600]}\n\n"
            f"Write a short, illustrative {language} code example (10-20 lines) "
            "that demonstrates the main concept. Return only the code block."
        )
        try:
            response = client.chat.complete(
                model="codestral-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            snippet = response.choices[0].message.content.strip()
            return f"{section}\n\n```{language}\n{snippet}\n```"
        except Exception as exc:
            print(f"Codestral error: {exc}")
            return section

    def generate_mermaid_diagram(self, description: str) -> str:
        """Generate a Mermaid flowchart diagram from an architecture description."""
        prompt = (
            f"Create a Mermaid flowchart diagram for: {description}\n"
            "Return only valid Mermaid syntax starting with 'flowchart LR' or 'flowchart TD'."
        )
        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content.strip()
            # Strip fences if present
            content = re.sub(r"```(?:mermaid)?", "", content).strip("` \n")
            return content
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return "flowchart LR\n    A[Input] --> B[LLM] --> C[Output]"

    def assemble_post(self, title: str, sections: list, mermaid: str) -> BlogPost:
        """Assemble expanded sections into a complete blog post, targeting 1200 words."""
        parts = [f"# {title}\n"]
        full_sections: dict = {}
        for sec in sections:
            expanded = self.expand_section(sec["title"], sec.get("key_points", []))
            parts.append(f"\n## {sec['title']}\n\n{expanded}")
            full_sections[sec["title"]] = expanded

        diagram_block = f"\n\n## Architecture\n\n```mermaid\n{mermaid}\n```"
        parts.append(diagram_block)

        full_text = "\n".join(parts)
        wc = len(full_text.split())
        return BlogPost(title=title, outline=sections, sections=full_sections,
                        mermaid_diagram=mermaid, word_count=wc, full_text=full_text)


gen = BlogPostGenerator()
topic = "How I Built a RAG Pipeline with Mistral and Vector Search"
desc = "A production retrieval-augmented generation system using Mistral embeddings, FAISS, and a FastAPI backend."
outline = gen.generate_outline(topic, desc)
print(f"Outline sections: {[s['title'] for s in outline]}")
assert len(outline) == 5, f"Expected 5 sections, got {len(outline)}"

mermaid = gen.generate_mermaid_diagram("User query → embedding → FAISS search → Mistral LLM → answer")
print(f"Mermaid diagram (first line): {mermaid.splitlines()[0]}")

# Expand only the first section to save API calls in demo
if outline:
    first = outline[0]
    section_text = gen.expand_section(first["title"], first.get("key_points", []))
    print(f"Section '{first['title']}' word count: {len(section_text.split())}")

# %% [markdown]
# ## Learning Path Advisor
# Career growth requires a personalized, prioritized study plan. `LearningPathAdvisor`
# assesses your skill matrix, identifies gaps against a target role, and generates
# a concrete 30/60/90-day plan using Mistral.

# %%
@dataclass
class SkillMatrix:
    """Snapshot of a learner's current skill levels."""
    skills: dict  # skill_name -> level (beginner/intermediate/advanced)
    target_role: str
    gaps: list = field(default_factory=list)


@dataclass
class PersonalizedPath:
    """A prioritized, time-estimated learning plan."""
    prioritized_topics: list
    recommended_projects: list
    estimated_hours: int
    plan_30_60_90: dict


ROLE_SKILLS = {
    "ai-engineer": ["LLMs", "RAG", "embeddings", "fine-tuning", "MLOps",
                    "Python", "vector databases", "prompt engineering", "API design", "evaluation"],
    "ml-engineer": ["PyTorch", "TensorFlow", "distributed training", "feature engineering",
                    "model serving", "MLOps", "statistics", "Python", "SQL", "Spark"],
}

RESOURCES = {
    "LLMs": ["Andrej Karpathy's makemore series", "Hugging Face NLP course", "fast.ai Practical Deep Learning"],
    "RAG": ["LangChain RAG tutorial", "LlamaIndex docs", "Building RAG with Mistral (official)"],
    "embeddings": ["Sentence Transformers docs", "OpenAI embedding cookbook", "Pinecone learning center"],
    "MLOps": ["Made With ML", "Full Stack Deep Learning", "Chip Huyen's FSDL course"],
    "prompt engineering": ["Anthropic prompt engineering guide", "Mistral prompting best practices",
                           "Learn Prompting (learnprompting.org)"],
    "vector databases": ["Pinecone docs", "Weaviate academy", "Qdrant quick-start"],
}


class LearningPathAdvisor:
    """Assess skills, identify gaps, and generate personalized AI engineering learning paths."""

    def assess_current_skills(self, user_profile: dict) -> SkillMatrix:
        """Build a SkillMatrix from a user-supplied profile dictionary."""
        skills = user_profile.get("skills", {})
        target_role = user_profile.get("target_role", "ai-engineer")
        return SkillMatrix(skills=skills, target_role=target_role)

    def identify_gaps(self, skill_matrix: SkillMatrix) -> list:
        """Return skills required for the target role that the user lacks or is beginner-level in."""
        required = ROLE_SKILLS.get(skill_matrix.target_role, [])
        gaps = []
        for skill in required:
            level = skill_matrix.skills.get(skill, "none")
            if level in ("none", "beginner"):
                gaps.append(skill)
        skill_matrix.gaps = gaps
        return gaps

    def resources_for_topic(self, topic: str) -> list:
        """Return a curated list of learning resources for a given topic."""
        return RESOURCES.get(topic, [f"Search 'learn {topic} AI engineering' on YouTube / Coursera"])

    def generate_30_60_90_plan(self, goals: str, gaps: list) -> dict:
        """Use Mistral to generate a structured 30/60/90-day learning plan."""
        prompt = (
            f"Create a detailed 30/60/90-day learning plan for an AI engineer.\n"
            f"Career goals: {goals}\n"
            f"Skill gaps to address: {', '.join(gaps[:8])}\n"
            "Return JSON: {\"day_30\": [\"...\"], \"day_60\": [\"...\"], \"day_90\": [\"...\"]}\n"
            "Each period should have 3-4 concrete, actionable milestones."
        )
        try:
            start = time.time()
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - start
            plan = json.loads(response.choices[0].message.content)
            print(f"30/60/90 plan generated in {elapsed:.2f}s")
            return plan
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return {"day_30": [], "day_60": [], "day_90": []}

    def build_path(self, user_profile: dict, goals: str) -> PersonalizedPath:
        """Assemble a full PersonalizedPath from profile and career goals."""
        matrix = self.assess_current_skills(user_profile)
        gaps = self.identify_gaps(matrix)
        plan = self.generate_30_60_90_plan(goals, gaps)
        projects = [f"Build a {g}-focused project" for g in gaps[:3]]
        hours = len(gaps) * 20
        return PersonalizedPath(
            prioritized_topics=gaps,
            recommended_projects=projects,
            estimated_hours=hours,
            plan_30_60_90=plan,
        )


advisor = LearningPathAdvisor()
profile = {
    "target_role": "ai-engineer",
    "skills": {"Python": "advanced", "prompt engineering": "intermediate", "SQL": "intermediate"},
}
matrix = advisor.assess_current_skills(profile)
gaps = advisor.identify_gaps(matrix)
print(f"Skill gaps identified: {gaps}")
assert len(gaps) > 0, "Expected gaps for a partial skill profile"

resources = advisor.resources_for_topic("RAG")
print(f"RAG resources: {resources}")

path = advisor.build_path(profile, goals="Transition from backend engineer to AI engineer in 6 months")
print(f"Estimated study hours: {path.estimated_hours}")
print(f"Day-30 milestones: {path.plan_30_60_90.get('day_30', [])}")

# %% [markdown]
# ## Job Description Analyzer
# Tailoring your application to each job posting dramatically improves response
# rates. `JobAnalyzer` extracts required skills, scores your resume fit, and
# rewrites bullets to align with what the hiring manager is looking for.

# %%
@dataclass
class MatchReport:
    """Resume-to-job-description alignment assessment."""
    match_score: float  # 0.0 – 1.0
    gaps: list
    strengths: list
    tailored_bullets: list = field(default_factory=list)


class JobAnalyzer:
    """Extract requirements from job descriptions and align resumes to them."""

    def extract_skills(self, job_description: str) -> dict:
        """Parse required and nice-to-have skills from a job description using Mistral."""
        prompt = (
            f"Extract skills from this job description:\n\n{job_description[:2000]}\n\n"
            "Return JSON: {\"required\": [\"...\"], \"nice_to_have\": [\"...\"]}"
        )
        try:
            start = time.time()
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - start
            data = json.loads(response.choices[0].message.content)
            print(f"Skills extracted in {elapsed:.2f}s — required: {len(data.get('required', []))}")
            return data
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return {"required": [], "nice_to_have": []}

    def match_resume(self, job_desc: str, resume_text: str) -> MatchReport:
        """Score how well a resume matches a job description and identify gaps."""
        prompt = (
            f"Job description (first 1000 chars):\n{job_desc[:1000]}\n\n"
            f"Resume (first 1000 chars):\n{resume_text[:1000]}\n\n"
            "Assess the match. Return JSON:\n"
            "{\"match_score\": 0.0-1.0, \"gaps\": [\"...\"], \"strengths\": [\"...\"]}\n"
            "gaps and strengths should each have 3 items."
        )
        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return MatchReport(
                match_score=float(data.get("match_score", 0.5)),
                gaps=data.get("gaps", []),
                strengths=data.get("strengths", []),
            )
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return MatchReport(match_score=0.0, gaps=[], strengths=[])

    def tailor_resume_bullet(self, bullet: str, job_desc: str) -> str:
        """Rewrite a resume bullet point to better match the job description."""
        prompt = (
            f"Rewrite this resume bullet to align with the job description.\n"
            f"Original: {bullet}\n"
            f"Job context (200 chars): {job_desc[:200]}\n"
            "Rules: start with a strong action verb, include a quantified result, "
            "mirror keywords from the job description. Return only the rewritten bullet."
        )
        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return bullet

    def generate_cover_letter_points(self, job: str, resume: str) -> list:
        """Return 4 key points to include in a cover letter."""
        prompt = (
            f"Given this job (200 chars): {job[:200]}\n"
            f"And this resume (200 chars): {resume[:200]}\n"
            "List 4 compelling cover letter talking points as JSON: {\"points\": [\"...\"]}"
        )
        try:
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("points", [])
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return []


SAMPLE_JOB = """
AI Engineer — Acme Corp
Required: Python, LLMs, RAG, vector databases, API design, MLOps, prompt engineering.
Nice to have: fine-tuning, Kubernetes, TypeScript, distributed systems.
You will build and deploy production LLM applications serving 1M+ users.
"""

SAMPLE_RESUME = """
Senior Software Engineer with 5 years Python experience.
Built REST APIs serving 500k daily users. Familiar with PyTorch and NLP basics.
Deployed ML models via FastAPI. No LLM production experience yet.
"""

job_analyzer = JobAnalyzer()
skills = job_analyzer.extract_skills(SAMPLE_JOB)
print(f"Required skills: {skills.get('required', [])[:4]}")

match_report = job_analyzer.match_resume(SAMPLE_JOB, SAMPLE_RESUME)
print(f"Match score: {match_report.match_score:.0%}")
print(f"Gaps: {match_report.gaps}")
assert 0.0 <= match_report.match_score <= 1.0

bullet = "Worked on backend services and helped with some ML experiments"
tailored = job_analyzer.tailor_resume_bullet(bullet, SAMPLE_JOB)
print(f"Original: {bullet}")
print(f"Tailored: {tailored}")

# %% [markdown]
# ## Open Source Contribution Finder
# Contributing to AI open source projects builds reputation and real-world experience.
# `OSS_Finder` searches GitHub for approachable issues, analyzes complexity, and
# drafts a PR description so you can hit the ground running.

# %%
@dataclass
class IssueAnalysis:
    """Complexity and value assessment of a GitHub issue."""
    repo: str
    issue_number: int
    title: str
    url: str
    estimated_hours: int
    required_skills: list
    value_to_maintainer: str  # low / medium / high


class OSS_Finder:
    """Find and analyze good-first-issue opportunities in AI repositories."""

    DEFAULT_REPOS = ["mistralai/mistral-inference", "langchain-ai/langchain"]

    def search_good_first_issues(self, repos: list = None, label: str = "good first issue") -> list:
        """Fetch open issues with the specified label from the given repositories."""
        repos = repos or self.DEFAULT_REPOS
        issues = []
        for repo in repos:
            url = f"https://api.github.com/repos/{repo}/issues"
            params = {"labels": label, "state": "open", "per_page": 5}
            try:
                resp = requests.get(url, headers=github_headers(), params=params, timeout=10)
                resp.raise_for_status()
                for issue in resp.json():
                    if "pull_request" not in issue:  # exclude PRs from issues endpoint
                        issues.append({"repo": repo, "issue": issue})
            except requests.RequestException as exc:
                print(f"GitHub error for {repo}: {exc}")
        print(f"Found {len(issues)} good-first issues across {len(repos)} repos")
        return issues

    def analyze_issue_complexity(self, repo: str, issue: dict) -> IssueAnalysis:
        """Use Mistral to estimate hours, required skills, and maintainer value for an issue."""
        body = (issue.get("body") or "")[:500]
        prompt = (
            f"Analyze this GitHub issue for contribution feasibility:\n"
            f"Repo: {repo}\nTitle: {issue['title']}\nBody: {body}\n\n"
            "Return JSON: {\"estimated_hours\": int, \"required_skills\": [\"...\"], "
            "\"value_to_maintainer\": \"low|medium|high\"}"
        )
        try:
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return IssueAnalysis(
                repo=repo, issue_number=issue["number"], title=issue["title"],
                url=issue["html_url"],
                estimated_hours=int(data.get("estimated_hours", 4)),
                required_skills=data.get("required_skills", []),
                value_to_maintainer=data.get("value_to_maintainer", "medium"),
            )
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return IssueAnalysis(repo=repo, issue_number=issue["number"],
                                 title=issue["title"], url=issue["html_url"],
                                 estimated_hours=4, required_skills=["Python"],
                                 value_to_maintainer="medium")

    def suggest_contribution(self, skill_level: str, issues: list) -> list:
        """Return the top 3 issues best suited to the given skill level (beginner/intermediate/advanced)."""
        hour_cap = {"beginner": 8, "intermediate": 20, "advanced": 9999}
        cap = hour_cap.get(skill_level, 20)
        analyzed = []
        for item in issues[:10]:
            analysis = self.analyze_issue_complexity(item["repo"], item["issue"])
            if analysis.estimated_hours <= cap:
                analyzed.append(analysis)
        analyzed.sort(key=lambda a: a.estimated_hours)
        top3 = analyzed[:3]
        for a in top3:
            print(f"  [{a.estimated_hours}h] {a.repo}#{a.issue_number}: {a.title[:60]}")
        return top3

    def generate_pr_description(self, issue: IssueAnalysis, solution_sketch: str) -> str:
        """Draft a PR description from an issue analysis and a solution sketch."""
        prompt = (
            f"Write a GitHub PR description for this contribution:\n"
            f"Issue: {issue.title} ({issue.url})\n"
            f"Solution approach: {solution_sketch}\n"
            "Include: motivation, what changed, how to test. Keep it under 200 words."
        )
        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"Mistral error: {exc}")
            return f"Fixes #{issue.issue_number}: {issue.title}\n\n{solution_sketch}"


oss = OSS_Finder()
live_issues = oss.search_good_first_issues(repos=["langchain-ai/langchain"])
if live_issues:
    print("Top contributions for an intermediate developer:")
    top3 = oss.suggest_contribution("intermediate", live_issues)
    if top3:
        sketch = "Refactor the relevant function, add a unit test, update docstring."
        pr_desc = oss.generate_pr_description(top3[0], sketch)
        print(f"\nSample PR description:\n{pr_desc[:300]}...")
else:
    print("No live issues fetched — add a GITHUB_TOKEN to increase rate limits.")

# %% [markdown]
# ## Lab Exercise: Personal AI Engineering Audit
# This self-contained exercise walks you through a complete career audit.
# Replace the placeholder values with your own GitHub username and resume text
# to produce a personalized `PersonalDevelopmentPlan`.

# %%
def run_personal_audit(
    github_username: str = "octocat",
    your_resume: str = SAMPLE_RESUME,
    your_goals: str = "Become a senior AI engineer focused on LLM applications",
    your_skills: dict = None,
) -> str:
    """
    Run a complete personal AI engineering audit and return a markdown development plan.

    Steps:
      1. Analyze GitHub portfolio
      2. Generate improvement suggestions for the weakest repo
      3. Generate a blog post outline for 'How I Built X'
      4. Compare profile against a sample AI engineer job description
      5. Assemble a PersonalDevelopmentPlan as markdown
    """
    if your_skills is None:
        your_skills = {"Python": "intermediate", "prompt engineering": "beginner"}

    print("=" * 60)
    print("PERSONAL AI ENGINEERING AUDIT")
    print("=" * 60)

    # Step 1: Portfolio analysis
    print("\n[1/4] Analyzing GitHub portfolio...")
    start = time.time()
    gh_analyzer = GitHubPortfolioAnalyzer()
    repos = gh_analyzer.fetch_repos(github_username)
    if repos:
        port_report = gh_analyzer.score_portfolio(repos)
    else:
        # Use sample data if GitHub API unavailable
        port_report = analyzer.score_portfolio(sample_repos)
    print(f"    {port_report.total_repos} repos | {port_report.ai_repos} AI repos | "
          f"avg README score {port_report.avg_readme_score:.0%}")

    # Step 2: Improvement suggestions
    print("\n[2/4] Generating improvement suggestions...")
    suggestions = gh_analyzer.generate_improvement_suggestions(port_report)

    # Step 3: Blog post outline
    print("\n[3/4] Generating blog post outline...")
    blog_gen = BlogPostGenerator()
    blog_outline = blog_gen.generate_outline(
        topic=f"How I Built a Production LLM App",
        project_description="A RAG-based assistant using Mistral AI",
    )

    # Step 4: Job match
    print("\n[4/4] Comparing profile to AI engineer job description...")
    jd_analyzer = JobAnalyzer()
    match = jd_analyzer.match_resume(SAMPLE_JOB, your_resume)

    # Step 5: Learning path
    lp_advisor = LearningPathAdvisor()
    profile = {"target_role": "ai-engineer", "skills": your_skills}
    matrix = lp_advisor.assess_current_skills(profile)
    gaps = lp_advisor.identify_gaps(matrix)
    plan = lp_advisor.generate_30_60_90_plan(your_goals, gaps)

    elapsed = time.time() - start

    # Assemble markdown report
    lines = [
        "# Personal AI Engineering Development Plan",
        "",
        f"**GitHub:** @{github_username}  ",
        f"**Target role:** AI Engineer  ",
        f"**Audit duration:** {elapsed:.1f}s",
        "",
        "## Portfolio Health",
        f"- Total repos: {port_report.total_repos}",
        f"- AI-related repos: {port_report.ai_repos}",
        f"- Average README score: {port_report.avg_readme_score:.0%}",
        f"- Top repos: {', '.join(port_report.top_repos)}",
        f"- Weakest repo: {port_report.weakest_repo}",
        "",
        "## Portfolio Improvements",
    ] + [f"- {s}" for s in suggestions] + [
        "",
        "## Blog Post Outline: How I Built a Production LLM App",
    ] + [f"- {sec.get('title', '')}" for sec in blog_outline] + [
        "",
        "## Job Match Report",
        f"- Match score: {match.match_score:.0%}",
        f"- Strengths: {', '.join(match.strengths)}",
        f"- Gaps: {', '.join(match.gaps)}",
        "",
        "## Skill Gaps to Address",
    ] + [f"- {g}" for g in gaps] + [
        "",
        "## 30/60/90-Day Plan",
        "### First 30 Days",
    ] + [f"- {m}" for m in plan.get("day_30", [])] + [
        "### Days 31-60",
    ] + [f"- {m}" for m in plan.get("day_60", [])] + [
        "### Days 61-90",
    ] + [f"- {m}" for m in plan.get("day_90", [])] + [
        "",
        "---",
        "*Generated by the AI Engineering Career Tools notebook.*",
    ]

    md_plan = "\n".join(lines)
    print("\n" + "=" * 60)
    print("PERSONAL DEVELOPMENT PLAN")
    print("=" * 60)
    print(md_plan[:1200], "...")  # preview
    return md_plan


# Run the audit — swap in your own username/resume to personalize
dev_plan = run_personal_audit(
    github_username="octocat",
    your_resume=SAMPLE_RESUME,
    your_goals="Land a senior AI engineer role at a product company within 6 months",
)
assert "Personal AI Engineering Development Plan" in dev_plan
print(f"\nTotal plan length: {len(dev_plan.split())} words")

# %% [markdown]
# ## Key Takeaways
# - Your GitHub portfolio is a living resume: score it objectively with automated
#   README analysis and act on the improvement suggestions before your next job search.
# - Technical blog posts compound over time; generating an LLM-assisted outline
#   removes the blank-page barrier while keeping the expertise authentically yours.
# - Skill gap analysis against role-specific requirements turns vague career goals
#   into a prioritized, time-boxed 30/60/90-day study plan.
# - Job description parsing and resume tailoring increase interview callback rates
#   by mirroring the exact language hiring managers scan for.
# - Open source contributions to AI repos (even small documentation or test fixes)
#   create public proof-of-work that differentiates you from candidates with only
#   private work experience.
