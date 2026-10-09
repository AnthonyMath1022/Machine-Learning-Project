"""Extract job fields from JobStreet HTML or schema.org JobPosting metadata."""

import json

from bs4 import BeautifulSoup, Tag


REQUIREMENT_HEADINGS = {
    "requirements", "requirement", "qualifications", "preferred qualifications",
    "skills required", "what we're looking for", "what we are looking for",
    "candidate profile", "job requirements", "your qualifications",
}
_SECTION_END_HEADINGS = {
    "responsibilities", "key responsibilities", "job responsibilities",
    "job description", "job scope", "what you'll do", "your role", "benefits",
    "what we offer", "about us", "company description", "additional information",
}


def _posting(soup: BeautifulSoup | Tag) -> dict:
    """Find a JobPosting in a JSON-LD object, array, or @graph."""
    def find(value):
        if isinstance(value, dict):
            types = value.get("@type", [])
            if types == "JobPosting" or isinstance(types, list) and "JobPosting" in types:
                return value
            return find(value.get("@graph", []))
        if isinstance(value, list):
            for item in value:
                posting = find(item)
                if posting:
                    return posting
        return {}

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            posting = find(json.loads(script.get_text()))
        except (ValueError, TypeError):
            continue
        if posting:
            return posting
    return {}


def _text(node: Tag | BeautifulSoup) -> str:
    # Work on a copy so callers can reuse their soup for other fields.
    clean = BeautifulSoup(str(node), "html.parser")
    for element in clean.select("script, style, noscript, nav, footer, form, svg"):
        element.decompose()
    # Separate blocks, but keep inline markup together (e.g. Power <b>BI</b>).
    for element in clean.find_all("br"):
        element.replace_with("\n")
    for element in clean.find_all(("p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6")):
        element.insert_before("\n")
        element.insert_after("\n")
    return "\n".join(
        " ".join(line.split()) for line in clean.get_text(" ").splitlines()
        if line.strip()
    )


def _description_node(soup: BeautifulSoup | Tag) -> Tag | BeautifulSoup | None:
    node = soup.select_one('[data-automation="jobAdDetails"], [itemprop="description"]')
    if node is not None:
        return node
    description = _posting(soup).get("description")
    if isinstance(description, str) and description.strip():
        return BeautifulSoup(description, "html.parser")
    return None


def extract_title(soup: BeautifulSoup | Tag) -> str | None:
    node = soup.select_one('[data-automation="job-detail-title"], [data-automation="jobTitle"], h1')
    if node is not None:
        text = node.get_text(" ", strip=True)
        if text:
            return text
    title = _posting(soup).get("title")
    return title.strip() if isinstance(title, str) and title.strip() else None


def extract_company(soup: BeautifulSoup | Tag) -> str | None:
    node = soup.select_one('[data-automation="advertiser-name"], [data-automation="jobCompany"], [data-automation="companyName"]')
    if node is not None:
        text = node.get_text(" ", strip=True)
        if text:
            return text
    organization = _posting(soup).get("hiringOrganization", {})
    name = organization.get("name") if isinstance(organization, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else None


def extract_description(soup: BeautifulSoup | Tag) -> str | None:
    """Return only an identified ad body; never fall back to the entire page."""
    node = _description_node(soup)
    return (_text(node) or None) if node is not None else None


def extract_requirements(soup: BeautifulSoup | Tag) -> str | None:
    """Collect recognized requirements sections, or None if not identifiable.

    This is a heading heuristic, not a complete interpretation of prose.
    The full description remains available when headings cannot be identified.
    """
    description = extract_description(soup)
    if not description:
        return None
    lines = []
    collecting = False
    for line in description.splitlines():
        heading = line.casefold().replace("\u2019", "'").strip().rstrip(":")
        if heading in REQUIREMENT_HEADINGS:
            collecting = True
        elif heading in _SECTION_END_HEADINGS:
            collecting = False
        elif collecting:
            lines.append(line)
    return "\n".join(lines) or None
