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
# LOAD MODEL & VECTORIZER
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading model...")
MODEL = joblib.load(os.path.join(BASE_DIR, "trained_model.pkl"))
TFIDF = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
print("Model loaded successfully!")

NUMERIC_FEATURES = [
    "uppercase_count", "exclamation_count",
    "question_count", "word_count", "avg_word_length",
    "raw_word_count"
]

BINARY_FEATURES = [
    "has_company_profile", "has_requirements", "has_benefits",
    "has_salary_range", "has_company_logo", "telecommuting", "has_questions"
]

ENGINEERED_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

# ============================================================
# RISK THRESHOLDS — 4 tier system
# ============================================================
HIGH_RISK_THRESHOLD   = 0.70  # >=70% fake_prob = HIGH RISK
MEDIUM_RISK_THRESHOLD = 0.40  # 40-70% = MEDIUM RISK
LOW_RISK_THRESHOLD    = 0.20  # 20-40% = LOW RISK
                              # <20% = LIKELY LEGITIMATE

def get_risk_level(fake_prob: float) -> dict:
    """Klasifikasi 4 tier berdasarkan fake probability."""
    if fake_prob >= HIGH_RISK_THRESHOLD:
        return {
            "level": "HIGH_RISK",
            "label": "High Risk",
            "message": "Strong indicators of fraud detected"
        }
    elif fake_prob >= MEDIUM_RISK_THRESHOLD:
        return {
            "level": "MEDIUM_RISK",
            "label": "Medium Risk",
            "message": "Several warning signs identified — verify carefully before applying"
        }
    elif fake_prob >= LOW_RISK_THRESHOLD:
        return {
            "level": "LOW_RISK",
            "label": "Low Risk",
            "message": "Some unusual patterns detected — review with caution"
        }
    else:
        return {
            "level": "LIKELY_LEGITIMATE",
            "label": "Likely Legitimate",
            "message": "Pattern consistent with legitimate job postings"
        }

# ============================================================
# PREPROCESSING (match notebook)
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
    """Match notebook inference: numeric dari teks, binary = 0."""
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
app = FastAPI(title="Fake Job Detector API", version="4.0")

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
    risk_level: str           # HIGH_RISK / MEDIUM_RISK / LOW_RISK / LIKELY_LEGITIMATE
    risk_label: str           # "High Risk" / "Medium Risk" / ...
    risk_message: str         # pesan lengkap untuk user
    fake_probability: float   # 0.0 - 1.0
    real_probability: float   # 0.0 - 1.0
    flags: list[str]

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Fake Job Detector API is running",
        "thresholds": {
            "high_risk":   HIGH_RISK_THRESHOLD,
            "medium_risk": MEDIUM_RISK_THRESHOLD,
            "low_risk":    LOW_RISK_THRESHOLD,
        }
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

        # Risk classification
        risk = get_risk_level(fake_prob)

        # Build context-aware flags
        flags = []
        text_lower = job.text.lower()
        is_suspicious = risk["level"] in ("HIGH_RISK", "MEDIUM_RISK", "LOW_RISK")

        if is_suspicious:
            if any(w in text_lower for w in ["earn", "income", "from home", "easy money", "no experience"]):
                flags.append("Suspicious compensation or work-from-home language")
            if features["exclamation_count"] > 3:
                flags.append(f"Excessive exclamation marks ({features['exclamation_count']} found)")
            if features["word_count"] < 80:
                flags.append("Unusually short job description")
            if features["uppercase_count"] > 50:
                flags.append("Heavy use of UPPERCASE text")
            if any(w in text_lower for w in ["bank", "wire transfer", "send money", "western union", "bank account details", "bank details"]):
                flags.append("Requests financial or banking information")
            if any(w in text_lower for w in ["limited spots", "apply now", "urgent", "immediate start", "act fast"]):
                flags.append("Uses urgency tactics")
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
            risk_level=risk["level"],
            risk_label=risk["label"],
            risk_message=risk["message"],
            fake_probability=fake_prob,
            real_probability=real_prob,
            flags=flags
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))