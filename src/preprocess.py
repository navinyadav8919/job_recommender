import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

# ✅ Clean and normalize resume text
def clean_text(text):
    text = text.encode('ascii', 'ignore').decode()  # Remove non-ASCII garbage
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# ✅ Load CSV and preprocess
def load_and_preprocess(filepath):
    # Proper encoding to avoid weird characters
    df = pd.read_csv(filepath, encoding='utf-8', encoding_errors='ignore')  # ✅ for pandas >= 1.3.0

    df = df[['Resume', 'Category']].dropna()
    df.columns = ['resume_text', 'job_category']
    df['cleaned_resume'] = df['resume_text'].apply(clean_text)
    return df

# ✅ TF-IDF vectorizer
def tfidf_vectorize(text_series):
    vectorizer = TfidfVectorizer(max_features=500)
    vectors = vectorizer.fit_transform(text_series).toarray()
    return vectors, vectorizer
