# backend/main.py
import re
import os
import string
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy.sparse import hstack, csr_matrix

# ============================================================
# LOAD MODEL & VECTORIZER (sekali saat server start)
# ============================================================
# Define base directory (tempat main.py berada)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Loading model...")
MODEL = joblib.load(os.path.join(BASE_DIR, "trained_model.pkl"))
TFIDF = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
print("Model loaded successfully!")

# Hardcode engineered features (sama persis dengan notebook cell 21)
# URUTAN INI TIDAK BOLEH DIUBAH — match dengan training
ENGINEERED_FEATURES = [
    "uppercase_count",
    "exclamation_count",
    "question_count",
    "word_count",
    "avg_word_length"
]

# ============================================================
# PREPROCESSING (sama persis dengan notebook cell 10, 12, 13)
# ============================================================
CUSTOM_STOPWORDS = {
    "experience", "work", "working", "team", "job", "jobs", "skills", "skill",
    "candidate", "company", "looking", "ability", "new", "year", "years",
    "business", "employee", "employees", "position", "role", "opportunity",
    "great", "good", "time", "need", "provide", "including", "required",
    "will", "must"
}
ALL_STOPWORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS)

def clean_text(text: str) -> str:
    """Sama persis dengan notebook cell 10."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text: str) -> str:
    """Sama persis dengan notebook cell 13."""
    return " ".join(w for w in text.split() if w not in ALL_STOPWORDS)

def extract_engineered_features(raw_text: str, clean_txt: str) -> dict:
    """
    Sama persis dengan notebook cell 21.
    PENTING:
    - uppercase_count, exclamation_count, question_count → dihitung dari full_text (raw)
    - word_count, avg_word_length → dihitung dari clean_text (sudah di-cleaning)
    """
    words = clean_txt.split()
    return {
        "uppercase_count":   sum(c.isupper() for c in raw_text),
        "exclamation_count": raw_text.count("!"),
        "question_count":    raw_text.count("?"),
        "word_count":        len(words),
        "avg_word_length":   float(np.mean([len(w) for w in words])) if words else 0.0,
    }

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Fake Job Detector API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # nanti pas deploy, ganti ke URL Vercel-mu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobInput(BaseModel):
    text: str = Field(..., min_length=50, description="Full job posting text")

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    flags: list[str]

@app.get("/")
def root():
    return {"status": "ok", "message": "Fake Job Detector API is running"}

@app.post("/predict", response_model=PredictionOutput)
@app.post("/predict", response_model=PredictionOutput)
def predict(job: JobInput):
    try:
        # 1. Clean text
        cleaned = clean_text(job.text)
        cleaned = remove_stopwords(cleaned)

        # 2. TF-IDF transform (ini SAJA — tanpa engineered features)
        X = TFIDF.transform([cleaned])

        # 3. Predict
        proba = MODEL.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        # 4. Hitung engineered features (HANYA untuk flag/UI, bukan untuk prediksi)
        eng = extract_engineered_features(job.text, cleaned)

        # 5. Build human-readable flags
        flags = []
        text_lower = job.text.lower()
        if pred_class == 1:  # FAKE
            if any(w in text_lower for w in ["earn", "income", "from home", "easy money", "no experience"]):
                flags.append("Suspicious compensation or work-from-home language")
            if eng["exclamation_count"] > 3:
                flags.append(f"Excessive exclamation marks ({eng['exclamation_count']} found)")
            if eng["word_count"] < 80:
                flags.append("Unusually short job description")
            if eng["uppercase_count"] > 50:
                flags.append("Heavy use of UPPERCASE text")
            if not flags:
                flags.append("Pattern matches known fraudulent postings")
        else:  # REAL
            if eng["word_count"] > 200:
                flags.append("Detailed and comprehensive description")
            if any(w in text_lower for w in ["bachelor", "degree", "years experience", "qualifications"]):
                flags.append("Specific qualification requirements listed")
            if any(w in text_lower for w in ["responsibilities", "requirements", "benefits"]):
                flags.append("Structured posting with clear sections")
            if not flags:
                flags.append("Language patterns consistent with legitimate postings")

        return PredictionOutput(
            prediction="FAKE" if pred_class == 1 else "REAL",
            confidence=confidence,
            probabilities={"real": float(proba[0]), "fake": float(proba[1])},
            flags=flags
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))