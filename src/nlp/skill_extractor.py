"""Extract known skills from resume or job description text."""

import re
from collections.abc import Iterable


SKILLS = (
    "Python", "R", "SQL", "Power BI", "Tableau", "Excel", "TensorFlow",
    "PyTorch", "AWS", "Azure", "Spark", "Hadoop", "SAS", "MATLAB",
)


def extract_skills(text: str, skills: Iterable[str] = SKILLS) -> list[str]:
    """Return catalog skills found as whole terms, ignoring case.

    This starter uses keywords; extend the catalog for your target roles.
    """
    return [
        skill for skill in skills
        if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", text, re.IGNORECASE)
    ]
