import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from deepagents import create_deep_agent
import os
import requests
import urllib.request

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class JobDescriptionSearch():
    def __init__(self):
        self
    def search_job(job_title,location):
        user_input = input("Enter the job title you want to search for: ")
        location_input = input("Enter the location you want to search in: ")
        url = f"https://my.jobstreet.com/en/job-search/job-vacancy.php?ojs=10&key={job_title}&location={location}"
        request = requests.get(url)
        content = urllib.request.urlopen(url).read()
        soup = BeautifulSoup(request.content, 'html.parser')


        prompt = f"Please extract the job description from the following HTML content:\n\n{soup.prettify()}"
        
        agent = create_deep_agent(
            model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        )

    def extract_job_description_from_html(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        job_description = soup.get_text(separator="\n")
        return job_description.strip()

    def calculate_similarity(resume_text, job_description):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        similarity_score = (tfidf_matrix * tfidf_matrix.T).A[0, 1]
        return similarity_score

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

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)
