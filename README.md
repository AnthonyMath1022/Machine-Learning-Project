# Job Market Project

A Python workspace for collecting job listings, extracting skills, and comparing
resumes with job descriptions. The current folder (`Machine Learning Project`)
is the project root; it corresponds to `job-market-project/` in the proposed layout.

## Folder structure

```text
job-market-project/
|-- src/
|   |-- __init__.py
|   |-- main.py
|   |-- crawler/
|   |   |-- __init__.py
|   |   |-- jobstreet.py
|   |   `-- parser.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- job.py
|   |   `-- model.py
|   `-- nlp/
|       |-- __init__.py
|       |-- skill_extractor.py
|       |-- embeddings.py
|       `-- matcher.py
|-- notebooks/
|   `-- resume_match.ipynb
|-- data/
|   `-- .gitkeep
|-- requirements.txt
|-- .env
|-- .env.example
|-- .gitignore
`-- README.md
```

## Where to put code and documents

| Path | Responsibility |
| --- | --- |
| `src/main.py` | Existing desktop resume-selection interface; future pipeline entry point. |
| `src/crawler/jobstreet.py` | JobStreet Malaysia search, individual ad fetching, and CSV export. |
| `src/crawler/parser.py` | Extract job fields from recognized HTML containers or JSON-LD JobPosting metadata. |
| `src/models/job.py` | Pydantic `Job` schema for structured listing data. |
| `src/models/model.py` | Preserved, optional language-model experiment from the original `model.py`. |
| `src/nlp/skill_extractor.py` | Keyword-based skill extraction; extend `SKILLS` here. |
| `src/nlp/embeddings.py` | Existing SentenceTransformer text embeddings. |
| `src/nlp/matcher.py` | Starter TF-IDF and vector cosine similarity functions. |
| `notebooks/resume_match.ipynb` | Existing notebook, including its saved outputs, for experiments. |
| `data/` | Local resumes, saved HTML, datasets, and matching results. Contents are ignored by Git. |
| `requirements.txt` | Starter dependency list; versions are unpinned and installation has not been verified. |
| `.env` | Local credentials, with blank placeholders. Ignored by Git. |

The original files were moved rather than discarded: `webcrawlling.py` became
`src/crawler/jobstreet.py`, `embedding.py` became `src/nlp/embeddings.py`, and
`requirements-resume.txt` became `requirements.txt`. Package `__init__.py` files
allow imports such as `from src.models.job import Job`.

## Setup

Run these PowerShell commands from the project root using Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `python` is unavailable, use the full path to your installed `python.exe`.
The virtual environment commands above do not require PowerShell activation.
Keep optional experimental dependencies commented out unless you use them.

The existing UI entry point is:

```powershell
.\.venv\Scripts\python.exe -m src.main
```

This launches a prototype; resume parsing and the complete matching pipeline
are still unfinished. The notebook-only `%gui qt6` line was removed from the
Python script and its launch code now has a `__main__` guard.

Open `notebooks/resume_match.ipynb` in VS Code or Jupyter. Its dependency-install
cell now resolves `requirements.txt` from either the project root or the
`notebooks/` directory. Saved output may still mention the old dependency name.
For imports from `src` in a notebook started inside `notebooks/`, run:

```python
import sys
from pathlib import Path

project_root = Path.cwd()
if not (project_root / "src").is_dir():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from src.nlp.skill_extractor import extract_skills
from src.nlp.matcher import calculate_similarity

skills = extract_skills("Python, SQL, and Power BI experience")
score = calculate_similarity("Python SQL", "Seeking Python and SQL skills")
```

## Configuration and remaining work

Fill in `.env` only when credentials are needed. Existing scripts do not
automatically load it; an API-backed workflow should explicitly call
`dotenv.load_dotenv` with the project's `.env` path.

The UI remains a prototype. It still needs a corrected
confirmation-dialog callback and implementations of `parse_resume` and
`clear_selection`. The crawler now uses the shared skill extractor and matching
helpers, but it has not been connected to the UI.

Importing the existing embeddings module loads a model, and importing the
optional `src/models/model.py` experiment loads a larger model and performs
generation. Keep those imports out of lightweight utilities until needed.
Connect the modules through `src/main.py` and validate the complete pipeline
before relying on matching results.

## JobStreet crawler

Use the crawler from the project root. Search returns summaries from one page;
fetch each selected ad separately when you need its full description:

```python
import requests

from src.crawler.jobstreet import JobDescriptionSearch, export_to_csv

with JobDescriptionSearch(timeout=20) as crawler:
    try:
        listings = crawler.search_job("Data Scientist", "Kuala Lumpur", page=1)
        if listings:
            job = crawler.fetch_job(listings[0].url)
            print(job.title, job.company, job.skills)
            print(crawler.calculate_similarity("Python SQL Power BI", job.description))
            export_to_csv([job], "data/jobs.csv")
    except (requests.RequestException, ValueError) as error:
        print(f"Could not retrieve a job: {error}")
```

The search route follows the [public JobStreet search page](https://my.jobstreet.com/data-scientist-jobs/in-Kuala-Lumpur).
Each request uses a timeout and checks HTTP status. Search deduplicates job IDs
and removes tracking parameters. A summary's description is `None` until the
full ad is fetched. Salary fields stay `None`; salary ranges and periods are
not yet parsed. Requirements extraction uses recognized headings and may return
`None` for unstructured prose; the full description is still retained.

JobStreet returned HTTP 403 during the live request check on 9 October 2026.
Successful live crawling and the current HTML selectors could not be verified.
This client propagates blocked requests and cannot render JavaScript. Missing
ad bodies and unsupported search layouts raise `ValueError`; a recognized
no-results message returns an empty list. Offline tests cover representative
HTML and JSON-LD fixtures, without contacting JobStreet.

The unfinished AI agent, duplicate URL request, interactive prompts, and
import-time NLTK download were removed. No API key or tokenization download is
needed. `extract_skills` and `cosine_similarity` remain importable from
`jobstreet.py` and reuse `src/nlp`. `Jobs_keywords_calculation` remains an alias
for `jobs_keywords_calculation`; it returns word tokens, not inferred skills.
Extend the skills catalog in `src/nlp/skill_extractor.py` for your target roles.
NLTK and DeepAgents are optional notebook dependencies in `requirements.txt`.

Run the offline regressions from the project root:

```powershell
py -3 -m unittest discover -s tests -v
```

If you use the project's virtual environment, substitute
`.\.venv\Scripts\python.exe` for `py -3`.
