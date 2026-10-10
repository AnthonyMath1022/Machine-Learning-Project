"""Offline regressions: no JobStreet access, API keys, or model downloads."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from bs4 import BeautifulSoup

from src.crawler.jobstreet import (
    JobDescriptionSearch, JobStreetAccessError, cosine_similarity, export_to_csv, extract_skills,
)
from src.crawler.parser import (
    extract_company, extract_description, extract_requirements, extract_title,
)


SEARCH_HTML = """
<html><body><article>
  <h2><a data-automation="jobTitle" href="/job/12345678?origin=cardTitle">Data Scientist</a></h2>
  <a data-automation="jobCompany">Example Sdn Bhd</a>
  <a data-automation="jobLocation">Kuala Lumpur</a>
  <a data-automation="jobTitle" href="/job/12345678?ref=duplicate">Data Scientist</a>
  <p data-automation="jobShortDescription">A summary, not the full ad.</p>
</article><article>
  <h3><a href="https://my.jobstreet.com/job/87654321">Data Analyst</a></h3>
</article>
<h2><a href="https://elsewhere.example/job/12345678">External ad</a></h2>
<h2><a href="/companies/example">Unrelated link</a></h2></body></html>
"""

DETAIL_HTML = """
<html><body><nav>Sign in and search other Python jobs</nav>
<h1 data-automation="job-detail-title">Data Scientist</h1>
<a data-automation="advertiser-name">Example Sdn Bhd</a>
<a data-automation="job-detail-location">Kuala Lumpur</a>
<div data-automation="jobAdDetails">
  <h2>Responsibilities</h2><p>Build dashboards.</p>
  <h2>Requirements:</h2><ul><li>Python and SQL experience.</li>
  <li>Power <b>BI</b> skills.</li></ul>
  <h2>Benefits</h2><p>Paid leave.</p>
  <script>fake skill: AWS</script><style>Azure</style>
</div><footer>TensorFlow roles elsewhere</footer></body></html>
"""


def response_for(html):
    response = Mock()
    response.text = html
    return response


class JobStreetTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock(spec=requests.Session)
        self.client = JobDescriptionSearch(session=self.session)

    def test_search_url_escapes_user_input_and_supports_pages(self):
        self.assertEqual(
            self.client.build_search_url(" Data Scientist ", "Kuala Lumpur", 2),
            "https://my.jobstreet.com/Data-Scientist-jobs/in-Kuala-Lumpur?page=2",
        )
        self.assertEqual(
            self.client.build_search_url("C++ / R&D", ""),
            "https://my.jobstreet.com/C%2B%2B-%2F-R%26D-jobs",
        )

    def test_invalid_inputs_do_not_make_requests(self):
        with self.assertRaises(ValueError):
            self.client.search_job(" ")
        for page in (0, -1, True, 1.5):
            with self.subTest(page=page), self.assertRaises(ValueError):
                self.client.search_job("Python", page=page)
        for timeout in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                JobDescriptionSearch(timeout=timeout, session=self.session)
        self.session.get.assert_not_called()

    def test_search_returns_unique_summaries_with_one_request(self):
        response = response_for(SEARCH_HTML)
        self.session.get.return_value = response
        with patch("builtins.input", side_effect=AssertionError("Unexpected prompt")):
            jobs = self.client.search_job("Data Scientist", "Kuala Lumpur")
        self.assertEqual([job.job_id for job in jobs], ["12345678", "87654321"])
        self.assertEqual(jobs[0].title, "Data Scientist")
        self.assertEqual(jobs[0].company, "Example Sdn Bhd")
        self.assertEqual(jobs[0].location, "Kuala Lumpur")
        self.assertIsNone(jobs[0].description)
        self.assertEqual(jobs[0].url, "https://my.jobstreet.com/job/12345678")
        self.assertIsNone(jobs[1].company)
        self.session.get.assert_called_once_with(
            self.client.build_search_url("Data Scientist", "Kuala Lumpur"), timeout=20.0,
        )
        response.raise_for_status.assert_called_once()
        response.close.assert_called_once()

    def test_explicit_no_results_is_empty(self):
        self.session.get.return_value = response_for("<h1>No matching jobs</h1>")
        self.assertEqual(self.client.search_job("Missing role"), [])

    def test_unknown_or_challenge_page_is_not_silently_empty(self):
        for html in ("", "<h1>Verify you are human</h1>", "<div id='app'></div>"):
            with self.subTest(html=html), self.assertRaises(ValueError):
                self.session.get.return_value = response_for(html)
                self.client.search_job("Python")

    def test_http_errors_propagate_and_close_response(self):
        response = response_for("<h1>Access denied</h1>")
        response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        self.session.get.return_value = response
        with self.assertRaises(requests.HTTPError):
            self.client.search_job("Python")
        response.close.assert_called_once()

    def test_timeout_propagates(self):
        self.session.get.side_effect = requests.Timeout("Timed out")
        with self.assertRaises(requests.Timeout):
            self.client.search_job("Python")

    def test_blocked_requests_have_actionable_errors_and_close_responses(self):
        for status in (403, 429):
            with self.subTest(status=status):
                response = response_for("<title>Just a moment...</title>")
                response.status_code = status
                response.raise_for_status.side_effect = requests.HTTPError(response=response)
                self.session.get.return_value = response
                with self.assertRaises(JobStreetAccessError) as caught:
                    self.client.search_job("Python")
                self.assertIs(caught.exception.response, response)
                self.assertIn(str(status), str(caught.exception))
                self.assertIn("browser", str(caught.exception))
                response.close.assert_called_once()

    def test_fetch_job_extracts_full_ad_without_navigation_or_scripts(self):
        self.session.get.return_value = response_for(DETAIL_HTML)
        job = self.client.fetch_job("/job/12345678?origin=cardTitle#details")
        self.assertEqual(job.title, "Data Scientist")
        self.assertEqual(job.company, "Example Sdn Bhd")
        self.assertEqual(job.location, "Kuala Lumpur")
        self.assertEqual(job.skills, ["Python", "SQL", "Power BI"])
        self.assertEqual(job.requirements, "Python and SQL experience.\nPower BI skills.")
        self.assertIn("Paid leave.", job.description)
        self.assertNotIn("fake skill", job.description)
        self.assertNotIn("Sign in", job.description)
        self.assertNotIn("TensorFlow", job.description)
        self.session.get.assert_called_once_with("https://my.jobstreet.com/job/12345678", timeout=20.0)

    def test_fetch_rejects_non_job_and_external_links_before_request(self):
        for url in (
            "https://elsewhere.example/job/12345678",
            "https://my.jobstreet.com.evil.example/job/12345678",
            "https://my.jobstreet.com/companies/example",
            "http://my.jobstreet.com/job/12345678",
            "https://my.jobstreet.com/job/not-a-number",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.client.fetch_job(url)
        self.session.get.assert_not_called()

    def test_fetch_missing_title_or_description_raises(self):
        for html in ("<h1>Expired ad</h1>", '<div data-automation="jobAdDetails">SQL</div>'):
            with self.subTest(html=html), self.assertRaises(ValueError):
                self.session.get.return_value = response_for(html)
                self.client.fetch_job("/job/12345678")

    def test_static_helpers_and_legacy_keyword_name(self):
        description = self.client.extract_job_description_from_html(DETAIL_HTML)
        self.assertIn("Power BI", description)
        self.assertEqual(JobDescriptionSearch.Jobs_keywords_calculation("Python, SQL!"), ["python", "sql"])
        self.assertEqual(self.client.jobs_keywords_calculation("Python, SQL!"), ["python", "sql"])
        with self.assertRaises(ValueError):
            self.client.extract_job_description_from_html("<nav>Jobs</nav>")

    def test_skill_terms_and_similarity_regressions(self):
        self.assertEqual(extract_skills("Work in research with spreadsheets"), [])
        self.assertEqual(extract_skills("R, SQL, and python"), ["Python", "R", "SQL"])
        self.assertEqual(self.client.calculate_similarity("", "Python SQL"), 0.0)
        self.assertEqual(self.client.calculate_similarity("!!!", "???"), 0.0)
        self.assertAlmostEqual(self.client.calculate_similarity("Python SQL", "Python SQL"), 1.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 2]), 0.0)
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])

    def test_session_ownership(self):
        with self.client:
            pass
        self.session.close.assert_not_called()
        owned = Mock(spec=requests.Session)
        with patch("src.crawler.jobstreet.requests.Session", return_value=owned):
            with JobDescriptionSearch():
                pass
        owned.close.assert_called_once()

    def test_csv_export_accepts_jobs_and_dictionaries_and_empty_results(self):
        self.session.get.return_value = response_for(DETAIL_HTML)
        job = self.client.fetch_job("/job/12345678")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output" / "jobs.csv"
            self.assertEqual(export_to_csv([job], path), path)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["company"], "Example Sdn Bhd")
            self.assertEqual(rows[0]["description"], job.description)
            export_to_csv([{"title": "Analyst", "company": "Caf\u00e9"}], path)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["company"], "Caf\u00e9")
            export_to_csv([], path)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                self.assertIn("job_id", next(csv.reader(stream)))


class ParserTests(unittest.TestCase):
    def test_jsonld_object_array_and_graph(self):
        posting = {
            "@type": ["JobPosting"], "title": "Analyst",
            "hiringOrganization": {"name": "Example"},
            "description": "<h2>Qualifications</h2><p>SQL required.</p><h2>Benefits</h2><p>Leave</p>",
        }
        for value in (posting, [posting], {"@graph": [{"@type": "Organization"}, posting]}):
            with self.subTest(value=value):
                soup = BeautifulSoup('<script type="application/ld+json">' + json.dumps(value) + '</script>', "html.parser")
                self.assertEqual(extract_title(soup), "Analyst")
                self.assertEqual(extract_company(soup), "Example")
                self.assertIn("SQL required.", extract_description(soup))
                self.assertEqual(extract_requirements(soup), "SQL required.")

    def test_invalid_jsonld_does_not_hide_valid_posting(self):
        soup = BeautifulSoup('''
            <script type="application/ld+json">invalid</script>
            <script type="application/ld+json">{"@type": "JobPosting", "title": "Analyst", "description": "SQL"}</script>
        ''', "html.parser")
        self.assertEqual(extract_title(soup), "Analyst")
        self.assertEqual(extract_description(soup), "SQL")

    def test_missing_fields_return_none(self):
        soup = BeautifulSoup("<nav>Sign in</nav>", "html.parser")
        for extractor in (extract_title, extract_company, extract_description, extract_requirements):
            self.assertIsNone(extractor(soup))

    def test_requirements_heading_variants_and_repeated_sections(self):
        soup = BeautifulSoup('''<div data-automation="jobAdDetails">
            <p>What We\u2019re Looking For:</p><p>Python</p>
            <h2>Responsibilities</h2><p>Build models</p>
            <h2>Preferred Qualifications</h2><p>SQL</p>
            <h2>Additional Information</h2><p>Apply now</p>
        </div>''', "html.parser")
        self.assertEqual(extract_requirements(soup), "Python\nSQL")

    def test_parser_preserves_original_soup(self):
        soup = BeautifulSoup(DETAIL_HTML, "html.parser")
        original = str(soup)
        extract_description(soup)
        extract_requirements(soup)
        self.assertEqual(str(soup), original)


if __name__ == "__main__":
    unittest.main()
