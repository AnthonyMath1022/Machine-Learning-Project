"""Fetch public JobStreet Malaysia pages and return structured job listings.

Run from the project root and import through ``src.crawler.jobstreet``.
This requests-based client cannot render JavaScript or bypass blocked requests.
"""

import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from src.crawler.parser import (
    extract_company,
    extract_description,
    extract_requirements,
    extract_title,
)
from src.models.job import Job
from src.nlp.matcher import calculate_similarity, cosine_similarity
from src.nlp.skill_extractor import SKILLS, extract_skills


BASE_URL = "https://my.jobstreet.com"


def _job_url(url: str) -> tuple[str, str]:
    """Validate a Malaysia job link and discard tracking parameters."""
    parsed = urlsplit(urljoin(BASE_URL, url))
    match = re.fullmatch(r"/job/(\d+)/?", parsed.path)
    if parsed.scheme != "https" or parsed.netloc != "my.jobstreet.com" or not match:
        raise ValueError("Expected a JobStreet Malaysia URL such as https://my.jobstreet.com/job/12345678.")
    job_id = match.group(1)
    return job_id, f"{BASE_URL}/job/{job_id}"


class JobDescriptionSearch:
    """Search one results page at a time; fetch full descriptions separately.

    Requests exceptions propagate to the caller. Unsupported or incomplete HTML
    raises ValueError so a blocked page is not treated as a job description.
    An injected session remains the caller's responsibility to close.
    """

    SKILLS = SKILLS

    def __init__(self, timeout: float = 20.0, session: requests.Session | None = None):
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number of seconds.")
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self._owns_session = session is None

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def build_search_url(job_title: str, location: str = "", page: int = 1) -> str:
        """Build the current public search route with escaped path segments."""
        if not job_title.strip():
            raise ValueError("job_title must not be blank.")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError("page must be a positive integer.")
        title = quote(re.sub(r"\s+", "-", job_title.strip()), safe="")
        url = f"{BASE_URL}/{title}-jobs"
        if location.strip():
            place = quote(re.sub(r"\s+", "-", location.strip()), safe="")
            url += f"/in-{place}"
        if page > 1:
            url += f"?page={page}"
        return url

    def _fetch_html(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        try:
            response.raise_for_status()
            return response.text
        finally:
            response.close()

    def search_job(self, job_title: str, location: str = "", page: int = 1) -> list[Job]:
        """Return unique job summaries from one page, without fetching every ad.

        Summary descriptions remain None until fetch_job() loads the full ad.
        Empty results are returned only for an explicit no-results message.
        """
        html = self._fetch_html(self.build_search_url(job_title, location, page))
        soup = BeautifulSoup(html, "html.parser")
        jobs = []
        seen = set()
        for link in soup.select('a[data-automation="jobTitle"][href], h2 a[href], h3 a[href]'):
            try:
                job_id, url = _job_url(link["href"])
            except ValueError:
                continue
            title = link.get_text(" ", strip=True)
            if job_id in seen or not title:
                continue
            card = link.find_parent("article")
            company = extract_company(card) if card is not None else None
            place = card.select_one('[data-automation="jobLocation"]') if card is not None else None
            jobs.append(Job(
                job_id=job_id, title=title, company=company,
                location=place.get_text(" ", strip=True) if place else None,
                url=url,
            ))
            seen.add(job_id)
        if not jobs:
            text = soup.get_text(" ", strip=True).lower().replace("\u2019", "'")
            if not any(message in text for message in (
                "no matching jobs", "no jobs found", "we couldn't find any jobs",
            )):
                raise ValueError("No job cards found. The page may require JavaScript, be blocked, or use an unsupported layout.")
        return jobs

    def fetch_job(self, url: str) -> Job:
        """Fetch a single full job ad for resume matching and skill extraction."""
        job_id, canonical_url = _job_url(url)
        soup = BeautifulSoup(self._fetch_html(canonical_url), "html.parser")
        title = extract_title(soup)
        description = extract_description(soup)
        if not title or not description:
            raise ValueError("Job title or description is missing. The ad may have expired, require JavaScript, or use an unsupported layout.")
        place = soup.select_one('[data-automation="job-detail-location"], [data-automation="jobLocation"]')
        return Job(
            job_id=job_id, title=title, company=extract_company(soup),
            location=place.get_text(" ", strip=True) if place else None,
            description=description, requirements=extract_requirements(soup),
            skills=extract_skills(description), url=canonical_url,
        )

    @staticmethod
    def extract_job_description_from_html(html_content: str) -> str:
        """Extract the ad body, excluding navigation, scripts, and other ads."""
        description = extract_description(BeautifulSoup(html_content, "html.parser"))
        if not description:
            raise ValueError("No job description found in the HTML.")
        return description

    # Keep the original helper entry points while sharing their implementations.
    calculate_similarity = staticmethod(calculate_similarity)

    @staticmethod
    def jobs_keywords_calculation(job_description: str) -> list[str]:
        """Return lowercase word tokens; use extract_skills() for known skills."""
        return re.findall(r"\b\w+\b", job_description.lower())

    Jobs_keywords_calculation = jobs_keywords_calculation


def export_to_csv(data: Iterable[Job | Mapping], filename: str | Path) -> Path:
    """Export Job objects or row dictionaries, returning the written path."""
    import pandas as pd

    rows = [row.model_dump() if isinstance(row, Job) else dict(row) for row in data]
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=None if rows else list(Job.model_fields)).to_csv(
        path, index=False, encoding="utf-8-sig",
    )
    return path

