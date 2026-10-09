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
|   |-- resume/
|   |   |-- __init__.py
|   |   |-- parser.py
|   |   `-- ocr.py
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
|-- requirements-desktop.txt
|-- requirements-model.txt
|-- requirements-resume.txt
|-- requirements-ocr.txt
|-- .env
|-- .env.example
|-- .gitignore
`-- README.md
```

## Where to put code and documents

| Path | Responsibility |
| --- | --- |
| `src/main.py` | PySide6 desktop resume selection, background extraction, text review and export. |
| `src/crawler/jobstreet.py` | JobStreet Malaysia search, individual ad fetching, and CSV export. |
| `src/crawler/parser.py` | Extract job fields from recognized HTML containers or JSON-LD JobPosting metadata. |
| `src/models/job.py` | Pydantic `Job` schema for structured listing data. |
| `src/models/model.py` | Lazy local Gemma agent for job summaries and resume fit comparisons. |
| `src/resume/parser.py` | Local PDF/DOCX/TXT resume extraction with clear extraction errors. |
| `src/resume/ocr.py` | Optional Baidu Unlimited-OCR fallback in an isolated CUDA worker. |
| `src/nlp/skill_extractor.py` | Keyword-based skill extraction; extend `SKILLS` here. |
| `src/nlp/embeddings.py` | Existing SentenceTransformer text embeddings. |
| `src/nlp/matcher.py` | Starter TF-IDF and vector cosine similarity functions. |
| `notebooks/resume_match.ipynb` | Existing notebook, including its saved outputs, for experiments. |
| `data/` | Local resumes, saved HTML, datasets, and matching results. Contents are ignored by Git. |
| `requirements.txt` | Starter dependency list; versions are unpinned and installation has not been verified. |
| `requirements-desktop.txt` | Lightweight PySide6 UI and PDF/DOCX readers without the AI stack. |
| `requirements-model.txt` | Optional local Gemma inference dependencies. |
| `requirements-resume.txt` | Lightweight PDF/DOCX parsing dependencies without model/UI packages. |
| `requirements-ocr.txt` | Baidu's tested Transformers 4.57.1 OCR stack, installed separately from Gemma. |
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

For just the desktop resume workspace, install the smaller dependency set:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
```

Launch the PySide6 desktop interface:

```powershell
.\.venv\Scripts\python.exe -m src.main
```

The workspace supports PDF, DOCX and UTF-8 TXT resumes. Browse for a document,
then click **Extract resume**. Extraction runs in a background thread so the
window stays responsive. Review the plain text, word/character counts and
keyword-based skill matches; use **Copy text** or **Save text** to export it.
**Open original document** uses the desktop's default application. The original
resume cannot be overwritten by the text export. Ctrl+O opens a resume, Ctrl+S
saves extracted text, and Ctrl+Q exits.

Encrypted PDFs prompt for a password. For image-only PDF pages, opt into
**Use OCR for scanned PDF pages** after completing the separate OCR setup below.
OCR is off by default; its first use may download model weights. While extraction
is running, file changes and closing are disabled until the worker finishes.
The interface also supports `python src/main.py`. Starting the desktop does not
load Gemma or embedding models. Job crawling and AI job matching are not yet
connected to this interface.

Run the desktop workflow tests with the lightweight dependencies installed:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_main.py -v
```

The tests use Qt's offscreen platform, real local PDF/DOCX/TXT fixtures and a
mocked OCR backend; they do not download or run AI models.

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

The desktop UI uses the shared resume extractor and skill catalog. The crawler
uses the shared skill extractor and matching helpers, but crawling and AI
matching still need to be connected to the UI.

Importing the existing embeddings module still loads a model. The Gemma agent
in `src/models/model.py` is safe to import: it loads weights only on the first
analysis call. Connect the modules through `src/main.py` and validate the
complete pipeline before relying on matching results.

## Local Gemma job and resume agent

The agent keeps `google/gemma-4-31B-it` as its default model and follows the
[official model loading API](https://huggingface.co/google/gemma-4-31B-it).
Install its optional dependencies inside the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
```

The first analysis downloads the model to the Hugging Face cache unless already
cached. The 31B model is large: approximately 61 GB for 16-bit parameter storage
alone, before inference overhead. The detected laptop GPU has about 8 GB VRAM;
the full model cannot fit on it. `device_map="auto"` permits CPU offloading when
host memory is sufficient; it does not quantize the model or guarantee it fits.
No full-weight download or live Gemma inference has been performed for this
implementation. Offline tests use a fake generation backend.

Use extracted plain text, not a filename, PDF bytes, raw HTML, or a placeholder
such as `<soup.prettify()>`:

```python
from src.models.model import JobMatchAgent

agent = JobMatchAgent()
job_text = "Data Analyst. Requirements: Python and SQL experience."
resume_text = "Built reporting pipelines using Python and SQL."

summary = agent.summarize_job(job_text)
print(summary.model_dump_json(indent=2))

# This method summarizes the original job again, then assesses the resume.
report = agent.assess_resume(job_text, resume_text)
print(report.fit)
print(report.model_dump_json(indent=2))
```

For crawler results, pass `job.description` from `fetch_job()` to the agent.
For saved HTML, obtain text with `extract_job_description_from_html()` first.
Use `extract_resume_text()` for PDF/DOCX/TXT resumes, as described below.
Desktop integration of the Gemma agent remains unfinished. Its command line accepts a UTF-8
job `.txt` file and a resume in any of those three formats:

```powershell
.\.venv\Scripts\python.exe -m src.models.model --job data/job.txt --resume data/resume.txt
.\.venv\Scripts\python.exe -m src.models.model --job data/job.txt --resume data/resume.pdf
.\.venv\Scripts\python.exe -m src.models.model --job data/job.txt --resume data/resume.docx
# Summary only:
.\.venv\Scripts\python.exe -m src.models.model --job data/job.txt
```

Each report includes a job summary, an advisory fit label (`strong_fit`,
`partial_fit`, `low_fit`, or `insufficient_information`), an assessment of each
extracted requirement, resume evidence, and follow-up questions. Requirement
indices refer to the zero-based `job_summary.requirements` list. A missing
resume mention is `not_evidenced`, not proof of missing ability. The prompt
restricts the analysis to job-related criteria and asks it to ignore protected
characteristics and instructions embedded in documents.

`low_fit` requires explicit conflicting resume evidence for a core requirement;
missing mentions alone cannot produce that conclusion.
Generated JSON is validated against typed schemas. Requirement and resume
quotes must occur in the original texts, and each extracted requirement must
be assessed once. Invalid results raise `AgentOutputError`; blank inputs and
oversized prompts raise `ValueError`. These checks verify structure and quote
presence, not whether the model's interpretation is correct or every job
requirement was extracted. A person must review the evidence and conclusions.

## Resume document extraction

Native text extraction runs locally without model weights, API keys, or Qt. The dependencies
are already included in `requirements.txt`; to install only the document readers:

```powershell
python -m pip install -r requirements-resume.txt
```

Use the shared extractor before sending a resume to the agent:

```python
from src.resume.parser import extract_resume_text
from src.models.model import JobMatchAgent

resume_text = extract_resume_text("data/resume.pdf")  # .docx and UTF-8 .txt also work
report = JobMatchAgent().assess_resume(job_description, resume_text)
print(report.model_dump_json(indent=2))

# For a password-protected PDF, pass the password supplied by its owner:
# resume_text = extract_resume_text("data/resume.pdf", password=pdf_password)
```

Inspect extracted text without starting the model:

```powershell
python -m src.resume.parser data/resume.pdf
python -m src.resume.parser data/resume.docx
```

PDF extraction reads all pages in order, skipping genuinely blank pages.
DOCX extraction walks paragraphs and tables in document order, including nested
tables and active header/footer variants. Linked header/footer parts and merged
table cells are included once. The original document is never modified.

`ResumeExtractionError` reports unsupported formats, corrupt files, or invalid
text encoding. `NoResumeTextError` reports documents with no readable text.
`EncryptedPDFError` requests a correct password. Missing or unreadable paths
raise standard `OSError` subclasses. Without OCR enabled, image-only PDF pages
raise `OCRRequiredError`, including in a mixed text/scanned document. With
`ocr=True` or `--ocr`, those pages use Baidu Unlimited-OCR and are merged with
native text in page order. Install `pypdf[crypto]` if PDF encryption
requires its optional cryptographic backend.

A PDF page
with both readable text and images may still contain unextracted image text;
review the extract. Complex PDF columns can also affect reading order (see the
[pypdf extraction limitations](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)).
DOCX floating text boxes, images, and tracked-change contents are not extracted.
Legacy `.doc` files must first be converted to `.docx`.

## Baidu Unlimited-OCR for scanned PDFs

This project uses [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)
for image-only PDF pages. Baidu's documented environment uses Transformers
4.57.1; the Gemma agent needs Transformers 5.7 or newer, so use separate virtual
environments. The OCR adapter starts a subprocess and waits for it to exit before
Gemma loads, releasing the OCR model's GPU memory. No API key is required.

Set up the OCR environment on Windows with Python 3.12 and CUDA PyTorch:

```powershell
py -3.12 -m venv .venv-ocr
.\.venv-ocr\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-ocr\Scripts\python.exe -m pip install -r requirements-ocr.txt
```

The CUDA 12.8 wheel selection follows the [official PyTorch installation table](https://pytorch.org/get-started/previous-versions/).
Use a compatible NVIDIA driver and a GPU supporting bfloat16. The worker reports
a clear error for CPU-only PyTorch, incompatible Transformers, or failed OCR.
The first OCR call downloads about 6.7 GB of model weights to
`data/model-cache/unlimited-ocr/`, which is ignored by Git.

```powershell
# Extraction only; run using the main environment that has requirements-resume.txt:
python -m src.resume.parser data/scanned-resume.pdf --ocr
# Extract first, then compare using the Gemma environment:
.\.venv\Scripts\python.exe -m src.models.model --job data/job.txt --resume data/scanned-resume.pdf --ocr
```

```python
from src.resume.parser import extract_resume_text

resume_text = extract_resume_text("data/scanned-resume.pdf", ocr=True)
# To use another OCR environment:
# resume_text = extract_resume_text(path, ocr=True, ocr_python="C:/envs/ocr/python.exe")
```

The default OCR interpreter is `.venv-ocr/Scripts/python.exe` on Windows or
`.venv-ocr/bin/python` elsewhere. Override it with `--ocr-python`, `ocr_python=`,
or `UNLIMITED_OCR_PYTHON`. Password-protected PDFs accept the same `password=`
argument as native extraction; the password is passed in a temporary request
file, not on the command line. Request files, rendered pages, and recognition
artifacts are deleted when the worker finishes, including after failures.

The model and its custom inference code are pinned to revision
`07dea832e22aefee32ad281d4b80551282e1c168`. Loading uses `trust_remote_code=True`
as required by the official implementation. Each requested page is rendered at
300 DPI and recognized separately at the documented 1024-pixel base size to
reduce VRAM pressure. The adapter uses the returned text with `save_results=False`
and removes layout markers; it does not run the model's visualization output.
No partial text is returned on worker failure, empty OCR results, missing page
results, or generation truncation. OCR accuracy still needs human review.
OCR is optional and off by default; native PDF/DOCX/TXT reads never load it.

Verification: all 55 offline tests passed. A real image-only synthetic resume
(`data/ocr-smoke-resume.pdf`) was recognized on the RTX 5050 Laptop GPU using
the pinned model and this CUDA environment. The synthetic text included Python,
SQL, Power BI, work experience and education. This confirms the local OCR smoke
test; it does not establish accuracy for every real resume or layout.

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
