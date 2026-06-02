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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading model...")
MODEL = joblib.load(os.path.join(BASE_DIR, "trained_model.pkl"))
TFIDF = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
print("Model loaded successfully!")

# Hardcode engineered features — MATCH PERSIS dengan notebook
NUMERIC_FEATURES = [
    "uppercase_count", "exclamation_count",
    "question_count", "word_count", "avg_word_length",
    "raw_word_count"
]

BINARY_FEATURES = [
    "has_company_profile", "has_requirements", "has_benefits",
    "has_salary_range", "has_company_logo", "telecommuting", "has_questions"
]

ENGINEERED_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES  # total 13

# ============================================================
# DECISION THRESHOLD — KRUSIAL untuk fraud detection
# ============================================================
# Notebook pakai F2-optimal threshold (lebih ke recall) atau F1-optimal,
# yang biasanya di range 0.15-0.30, BUKAN 0.5 default.
# Tune nilai ini sesuai feel:
#   0.20 = sangat agresif flag fake (recall tinggi, banyak false positive)
#   0.30 = agresif (rekomendasi untuk fraud detection)
#   0.40 = sedang
#   0.50 = default sklearn (terlalu konservatif untuk fraud)
FAKE_THRESHOLD = 0.30

# ============================================================
# PREPROCESSING (sama persis dengan notebook cell 33 & 37)
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
    """SAMA PERSIS dengan notebook cell 33."""
    text = str(text).lower()
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in ALL_STOPWORDS)

def extract_features(raw_text: str, cleaned: str) -> dict:
    """
    MATCH PERSIS dengan notebook cell 129 (predict_job_posting):
    - Numeric features dihitung dari raw_text & cleaned
    - Binary features SEMUA DI-SET 0 (sesuai notebook inference)
    """
    raw_words = raw_text.split()
    clean_words = cleaned.split()

    return {
        "uppercase_count":   sum(c.isupper() for c in raw_text),
        "exclamation_count": raw_text.count("!"),
        "question_count":    raw_text.count("?"),
        "word_count":        len(clean_words),
        "avg_word_length":   float(np.mean([len(w) for w in clean_words])) if clean_words else 0.0,
        "raw_word_count":    len(raw_words),
        "has_company_profile": 0,
        "has_requirements":    0,
        "has_benefits":        0,
        "has_salary_range":    0,
        "has_company_logo":    0,
        "telecommuting":       0,
        "has_questions":       0,
    }

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Fake Job Detector API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return {
        "status": "ok",
        "message": "Fake Job Detector API is running",
        "threshold": FAKE_THRESHOLD
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(job: JobInput):
    try:
        cleaned = clean_text(job.text)
        cleaned = remove_stopwords(cleaned)

        tfidf_vec = TFIDF.transform([cleaned])

        features = extract_features(job.text, cleaned)
        eng_vec = np.array(
            [[features[f] for f in ENGINEERED_FEATURES]],
            dtype=np.float64
        )

        X = hstack([tfidf_vec, csr_matrix(eng_vec)])

        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)[0]
            fake_prob = float(proba[1])
            real_prob = float(proba[0])
        elif hasattr(MODEL, "decision_function"):
            score = MODEL.decision_function(X)[0]
            fake_prob = float(1 / (1 + np.exp(-score)))
            real_prob = 1.0 - fake_prob
        else:
            pred = int(MODEL.predict(X)[0])
            fake_prob = 1.0 if pred == 1 else 0.0
            real_prob = 1.0 - fake_prob

        if fake_prob >= FAKE_THRESHOLD:
            pred_class = 1
            confidence = fake_prob
        else:
            pred_class = 0
            confidence = real_prob

        flags = []
        text_lower = job.text.lower()
        if pred_class == 1:
            if any(w in text_lower for w in ["earn", "income", "from home", "easy money", "no experience"]):
                flags.append("Suspicious compensation or work-from-home language")
            if features["exclamation_count"] > 3:
                flags.append(f"Excessive exclamation marks ({features['exclamation_count']} found)")
            if features["word_count"] < 80:
                flags.append("Unusually short job description")
            if features["uppercase_count"] > 50:
                flags.append("Heavy use of UPPERCASE text")
            if any(w in text_lower for w in ["bank", "wire transfer", "send money", "western union"]):
                flags.append("Requests financial information")
            if not flags:
                flags.append("Pattern matches known fraudulent postings")
        else:
            if features["word_count"] > 200:
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
            probabilities={"real": real_prob, "fake": fake_prob},
            flags=flags
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))