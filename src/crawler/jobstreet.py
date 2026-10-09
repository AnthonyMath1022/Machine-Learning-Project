from matplotlib.pylab import matrix
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from model import MODEL_ID, device, processor, model
import os
import requests
import urllib.request

import nltk
from nltk.tokenize import word_tokenize
from src.nlp.matcher import cosine_similarity

nltk.download('punkt')

class JobDescriptionSearch():
    def search_job(self, job_title, location):
        # Fetch search results and collect job-detail links here.
        user_input = input("Enter the job title you want to search for: ")
        location_input = input("Enter the location you want to search in: ")
        url = f"https://my.jobstreet.com/en/job-search/job-vacancy.php?ojs=10&key={job_title}&location={location}"
        request = requests.get(url)
        content = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(request.content, 'html.parser')

    @staticmethod
    def extract_job_description_from_html(html_content, selector):
        soup = BeautifulSoup(html_content, "html.parser")
        description = soup.select_one(selector)

        if description is None:
            raise ValueError(
                "Description element not found; check the HTML and selector."
            )

        for element in description.select("script, style, noscript"):
            element.decompose()

        text = description.get_text(separator="\n", strip=True)

        if not text:
            raise ValueError("The description element is empty.")

        return text

    def calculate_similarity(resume_text, job_description):
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([resume_text, job_description])
        return float(cosine_similarity(matrix[0], matrix[1])[0, 0])

    def Jobs_keywords_calculation(job_description):
        tokens = word_tokenize(job_description)
        keywords = [word.lower() for word in tokens if word.isalnum()]
        return keywords

    SKILLS = [
    "Python",
    "R",
    "SQL",
    "Power BI",
    "Tableau",
    "Excel",
    "TensorFlow",
    "PyTorch",
    "AWS",
    "Azure",
    "Spark",
    "Hadoop",
    "SAS",
    "MATLAB"
]

REQUIREMENT_HEADINGS = {
    "requirements",
    "requirement",
    "qualifications",
    "preferred qualifications",
    "skills required",
    "what we're looking for",
    "candidate profile"
}

RESPONSIBILITY_HEADINGS = {
    "responsibilities",
    "key responsibilities",
    "job responsibilities",
    "job description",
    "job scope",
    "what you'll do",
    "your role"
}
def extract_skills(text):

    text_lower = text.lower()

    return [
        skill
        for skill in SKILLS
        if skill.lower() in text_lower
    ]

def export_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data exported to {filename}")

