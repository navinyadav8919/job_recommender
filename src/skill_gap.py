import spacy
nlp = spacy.load("en_core_web_sm")

# Predefined skill sets per job role
job_skills = {
    "Software Engineer": ["python", "java", "spring", "sql"],
    "Data Scientist": ["python", "pandas", "ml", "numpy"],
    "HR": ["communication", "recruitment", "training"],
    "Web Developer": ["html", "css", "javascript", "react"],
}

def extract_keywords(text):
    doc = nlp(text.lower())
    return list(set([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN']]))

def skill_gap(predicted_role, resume_keywords):
    expected = set(job_skills.get(predicted_role, []))
    actual = set(resume_keywords)
    missing = expected - actual
    return list(missing)
