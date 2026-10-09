from pydantic import BaseModel


class Job(BaseModel):
    job_id: str
    title: str
    company: str | None = None
    location: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None

    description: str | None = None
    requirements: str | None = None

    skills: list[str] = []

    url: str