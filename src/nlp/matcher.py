"""Basic similarity scoring for resume and job description text."""

from collections.abc import Sequence
from math import sqrt


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compare equal-length vectors; return zero if either vector has zero norm."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same number of dimensions.")
    magnitude1 = sqrt(sum(value ** 2 for value in vec1))
    magnitude2 = sqrt(sum(value ** 2 for value in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return sum(a * b for a, b in zip(vec1, vec2)) / (magnitude1 * magnitude2)


def calculate_similarity(resume_text: str, job_description: str) -> float:
    """Return TF-IDF cosine similarity, a text score rather than a hiring probability."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not resume_text.strip() or not job_description.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    if not vectorizer.build_analyzer()(resume_text + " " + job_description):
        return 0.0
    matrix = vectorizer.fit_transform([resume_text, job_description])
    return float((matrix @ matrix.T).toarray()[0, 1])
