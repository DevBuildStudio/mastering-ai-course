# Week 0: Setup & Orientation

> **Theme: Get your environment working before you learn a single concept.** This short, pre-course week exists so that Week 1 can focus entirely on ideas, not installation problems. By the end of Week 0 you will have Python, your editor, and a working Mistral API key all verified with a one-line test call.

---

## 0.1 How This Course Works

This is Course 1 of a three-course AI Engineering curriculum. Each week pairs a markdown chapter (concepts, diagrams, checkpoints) with a runnable Python file and a matching Jupyter notebook:

| Artifact | Purpose |
|---|---|
| `weekN-topic.md` | The chapter — read this first. Explains concepts, includes diagrams and checkpoint questions. |
| `code/weekN_topic.py` | A standalone script version of the week's exercises — good for running from the command line. |
| `notebooks/weekN_topic.ipynb` | An interactive notebook version of the same material — good for experimenting cell-by-cell. |

You do not need to choose between the script and the notebook — most learners read the chapter, then work through the notebook, and keep the script as a reference.

---

## 0.2 Prerequisites

- **Python basics** — functions, classes, `pip`, virtual environments
- **REST APIs** — making HTTP requests, understanding JSON responses
- **Command line usage** — navigating directories, running scripts

You do **not** need prior experience with machine learning, PyTorch/TensorFlow, or GPUs. Everything in Course 1 runs on a laptop CPU using hosted APIs.

---

## 0.3 Install What You Need

You'll need four things installed on your machine:

- **Python 3.11+**
- **Node.js 20+** (only needed later, for Course 2's MCP servers — safe to install now)
- **Git**
- **VS Code** with the Jupyter extension

> [!NOTE]
> **In plain English:** Think of this like setting up a kitchen before cooking. Python is your stove, VS Code is your countertop and tools, and the API key (next section) is your ingredient delivery service. None of the recipes (Week 1 onward) will work until the kitchen is ready.

---

## 0.4 Get a Mistral API Key

This course uses [Mistral AI](https://console.mistral.ai) for all hosted-model exercises.

1. Go to [console.mistral.ai](https://console.mistral.ai)
2. Sign up or log in
3. Navigate to **API Keys** in the left sidebar
4. Click **Create new key** and give it a name (e.g. `ai-engineering-course`)
5. Copy the key immediately — it will not be shown again
6. Add billing info if prompted (a free tier is available for development)

> **Key Insight:** Keep your API key secret. Never commit it to git. Always load it from an environment variable, never hard-code it into a script or notebook.

---

## 0.5 Set Up Your Environment

```bash
# Navigate to the course directory
cd mastering-ai-course

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
.\venv\Scripts\activate    # Windows

# Install Course 1 dependencies
pip install -r course1/requirements.txt

# Copy the environment template and add your key
cp course1/.env.example course1/.env
# Edit course1/.env and set MISTRAL_API_KEY=your_key_here
```

Your `course1/.env` file should contain at minimum:

```
MISTRAL_API_KEY=your_key_here
```

---

## 0.6 Verify Your Setup

Run this quick check — either as a script or in a fresh notebook cell — before moving on to Week 1:

```python
import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

api_key = os.environ.get("MISTRAL_API_KEY")
assert api_key, "MISTRAL_API_KEY not found — check your .env file"

client = Mistral(api_key=api_key)
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Reply with exactly: setup ok"}],
)
print(response.choices[0].message.content)
```

If this prints `setup ok` (or something close to it), your environment is ready for Week 1.

---

## 0.7 Running Notebooks

```bash
pip install jupyterlab
jupyter lab
```

JupyterLab opens in your browser — navigate to `course1/notebooks/` to find the `.ipynb` files. You can also open notebooks directly in VS Code using the Jupyter extension, which is the recommended workflow for this course.

---

## 0.8 Troubleshooting

**API key not found (`AuthenticationError` or `KeyError: 'MISTRAL_API_KEY'`)**
- Confirm `course1/.env` exists and contains `MISTRAL_API_KEY=...` with no extra spaces or quotes.
- Make sure `load_dotenv()` is called before any API calls — in a notebook, do this in the first cell.
- On Windows, confirm the file wasn't saved as `.env.txt` — enable file extensions in File Explorer to check.

**Rate limit errors (`429 Too Many Requests`)**
- Free-tier accounts have lower rate limits. Add a short `time.sleep(1)` between requests when batching calls.
- Check usage and quota at [console.mistral.ai](https://console.mistral.ai) under **Usage**.

**Import errors (`ModuleNotFoundError`)**
- Ensure your virtual environment is activated before running scripts or launching JupyterLab.
- Re-run `pip install -r course1/requirements.txt` inside the activated environment.

---

## Chapter 0 Checkpoint

Before moving on to Week 1, make sure you can answer:

1. Where should your `MISTRAL_API_KEY` live, and where should it never live?
2. What's the difference between the `.md` chapter, the `code/*.py` script, and the `notebooks/*.ipynb` notebook for a given week?
3. What output confirms your environment is correctly set up?

**Next:** [Week 1: How LLMs Actually Work](week1-llm-foundations.md)
