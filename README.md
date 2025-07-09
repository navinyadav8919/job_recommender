
# 💼 AI Job Recommender System

An intelligent, AI-powered job recommender system that analyzes resume text and suggests the most suitable job role. It combines NLP, unsupervised clustering, and supervised machine learning to provide accurate career predictions and skill gap analysis.

## 🚀 Features

- 📄 **Resume Parsing & Cleaning** with spaCy
- 🔤 **TF-IDF Vectorization** of resumes
- 🧠 **KMeans Clustering** to group similar resumes
- 🎯 **RandomForest Classifier** for job role prediction
- ⚡ **Skill Gap Analyzer** comparing user skills with job role expectations
- 🌐 **Streamlit Web App Interface**


## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Language | Python |
| NLP | spaCy, TfidfVectorizer |
| ML Models | KMeans, RandomForestClassifier |
| Frontend | Streamlit |
| DevOps | Git, GitHub |
| Others | pandas, scikit-learn, joblib |


## 📁 Folder Structure


job\_recommender/
│
├── app/                   # Streamlit app
│   └── app.py
│
├── src/                   # Preprocessing, clustering, skill-gap logic
│   ├── preprocess.py
│   ├── clustering.py
│   ├── model.py
│   └── skill\_gap.py
│
├── notebooks/             # Training notebooks
│   └── train\_model.ipynb
│
├── models/                # Saved ML models (pkl files)
│   ├── tfidf\_vectorizer.pkl
│   ├── kmeans\_model.pkl
│   └── classifier.pkl
│
├── Data/                  # Dataset and cleaned CSVs
│   └── UpdatedResumeDataSet.csv
│
├── requirements.txt
└── README.md


## 🧪 How It Works

1. Resume is cleaned and preprocessed using **spaCy** (lemmatization, stopwords removed, non-ASCII cleaned)
2. Vectorized using **TF-IDF** (max 3000 features)
3. Clustered using **KMeans** to find resume patterns
4. A **RandomForest classifier** is trained using the cluster + TF-IDF features
5. User enters resume in the app → prediction + skill-gap

## 📦 Installation

1. **Clone the repository**

git clone https://github.com/navinyadav8919/job_recommender.git
cd job_recommender

2. **Create virtual environment**

python -m venv env
env\Scripts\activate  # Windows


3. **Install requirements**

pip install -r requirements.txt
python -m spacy download en_core_web_sm


4. **Run the app**

cd app
streamlit run app.py


## 🔍 Example Resume Input

I am experienced in Python, pandas, scikit-learn, and machine learning. I’ve built models using supervised and unsupervised learning and worked on data cleaning and visualization.

🧠 Output:

* ✅ Recommended Job Role: `data science`
* ⚠️ Skill Gap: `sql, deep learning`


## 📈 Future Improvements

* Use sentence embeddings (e.g., BERT)
* Add user authentication
* Integrate job posting APIs
* Support PDF resume uploads


## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.



## 📄 License

MIT License

## 📬 Contact

* **Author**: Naveen Yadav
* **GitHub**: [@navinyadav8919](https://github.com/navinyadav8919)

