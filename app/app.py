import streamlit as st
import pandas as pd
import pickle
import os
import sys
# from src.preprocess import clean_text


# Ensure src module is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.skill_gap import extract_keywords, skill_gap
from src.preprocess import clean_text

# === Load Models ===
with open(r"D:\Projects\iscale_projects\job_recommender\models\tfidf_vectorizer.pkl", 'rb') as f:
    tfidf_model = pickle.load(f)

with open(r"D:\Projects\iscale_projects\job_recommender\models\kmeans_model.pkl", 'rb') as f:
    kmeans_model = pickle.load(f)

with open(r"D:\Projects\iscale_projects\job_recommender\models\classifier.pkl", 'rb') as f:
    clf = pickle.load(f)

# === Load Cleaned Data (Optional, for category checking or stats) ===
df = pd.read_csv(r"D:\Projects\iscale_projects\job_recommender\Data\cleaned_resumes.csv")

# === Streamlit UI ===
st.set_page_config(page_title="AI Job Recommender", layout="wide")
st.title("🧠 AI-Powered Job Recommender for Students")

st.header("📋 Paste Your Resume Below")
user_resume = st.text_area("Resume Text", height=250)

if st.button("🔍 Recommend Job"):
    if not user_resume.strip():
        st.error("⚠️ Please paste your resume text to get a recommendation.")
    else:
        # Preprocess user input
        cleaned_user_resume = clean_text(user_resume)
        user_vec = tfidf_model.transform([cleaned_user_resume]).toarray()
        cluster_label = kmeans_model.predict(user_vec)[0]

        input_df = pd.DataFrame(user_vec)
        input_df['cluster'] = cluster_label
        input_df.columns = input_df.columns.astype(str)  # To match training format

        prediction = clf.predict(input_df)[0]
        keywords = extract_keywords(user_resume)
        gap = skill_gap(prediction, keywords)

        st.success(f"✅ Recommended Job Role: **{prediction.upper()}**")

        if gap:
            st.warning("🛠️ You may want to improve on the following skills:")
            st.code(", ".join(gap))
        else:
            st.info("🎯 Your resume already matches the key skills for this role!")
