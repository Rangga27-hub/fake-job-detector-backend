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

# Order MUST MATCH training (cell 51 di notebook)
NUMERIC_FEATURES = [
    "uppercase_count", "exclamation_count",
    "question_count", "word_count", "avg_word_length",
    "raw_word_count"
]

BINARY_FEATURES = [
    "has_company_profile", "has_requirements", "has_benefits",
    "has_salary_range", "has_company_logo", "telecommuting", "has_questions"
]

ENGINEERED_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES  # 13 total

# ============================================================
# RISK THRESHOLDS (4-tier)
# ============================================================
HIGH_RISK_THRESHOLD   = 0.70
MEDIUM_RISK_THRESHOLD = 0.40
LOW_RISK_THRESHOLD    = 0.20

def get_risk_level(fake_prob: float) -> dict:
    if fake_prob >= HIGH_RISK_THRESHOLD:
        return {"level": "HIGH_RISK", "label": "High Risk",
                "message": "Strong indicators of fraud detected"}
    elif fake_prob >= MEDIUM_RISK_THRESHOLD:
        return {"level": "MEDIUM_RISK", "label": "Medium Risk",
                "message": "Several warning signs identified — verify carefully before applying"}
    elif fake_prob >= LOW_RISK_THRESHOLD:
        return {"level": "LOW_RISK", "label": "Low Risk",
                "message": "Some unusual patterns detected — review with caution"}
    else:
        return {"level": "LIKELY_LEGITIMATE", "label": "Likely Legitimate",
                "message": "Pattern consistent with legitimate job postings"}

# ============================================================
# PREPROCESSING (match notebook cell 33 & 37)
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

# ============================================================
# INPUT SCHEMA (structured form, bukan free text)
# ============================================================
class JobInput(BaseModel):
    # Required text fields
    title: str = Field(..., min_length=3, max_length=200,
                       description="Job title")
    description: str = Field(..., min_length=50, max_length=10000,
                             description="Full job description")

    # Optional text fields (kosong = has_xxx = 0)
    company_profile: str = Field(default="", max_length=5000,
                                  description="Company background information")
    requirements: str = Field(default="", max_length=5000,
                              description="Job requirements & qualifications")
    benefits: str = Field(default="", max_length=2000,
                          description="Benefits and perks")
    salary_range: str = Field(default="", max_length=200,
                              description="Salary range (e.g. $50,000-$70,000)")

    # Required binary checkboxes
    has_company_logo: bool = Field(...,
                                    description="Does the posting display a company logo?")
    telecommuting: bool = Field(...,
                                 description="Is the position remote/work-from-home?")
    has_questions: bool = Field(...,
                                 description="Does the posting include screening questions?")

class PredictionOutput(BaseModel):
    risk_level: str
    risk_label: str
    risk_message: str
    fake_probability: float
    real_probability: float
    flags: list[str]
    features_summary: dict  # untuk transparency, biar user tau apa yang dihitung

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Fake Job Detector API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Fake Job Detector API is running",
        "version": "5.0 - Structured Form Input",
        "thresholds": {
            "high_risk": HIGH_RISK_THRESHOLD,
            "medium_risk": MEDIUM_RISK_THRESHOLD,
            "low_risk": LOW_RISK_THRESHOLD,
        }
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(job: JobInput):
    try:
        # ============================================================
        # 1. BUILD full_text (gabungan 5 kolom teks — sama persis dengan notebook cell 31)
        # ============================================================
        full_text = " ".join([
            job.title,
            job.company_profile,
            job.description,
            job.requirements,
            job.benefits,
        ]).strip()

        # ============================================================
        # 2. CLEANING (notebook cell 33)
        # ============================================================
        cleaned = clean_text(full_text)
        cleaned = remove_stopwords(cleaned)

        # ============================================================
        # 3. TF-IDF (5000 features)
        # ============================================================
        tfidf_vec = TFIDF.transform([cleaned])

        # ============================================================
        # 4. NUMERIC FEATURES (6) — dihitung dari full_text/clean_text
        # ============================================================
        raw_words = full_text.split()
        clean_words = cleaned.split()

        features = {
            "uppercase_count":   sum(c.isupper() for c in full_text),
            "exclamation_count": full_text.count("!"),
            "question_count":    full_text.count("?"),
            "word_count":        len(clean_words),
            "avg_word_length":   float(np.mean([len(w) for w in clean_words])) if clean_words else 0.0,
            "raw_word_count":    len(raw_words),
        }

        # ============================================================
        # 5. BINARY FEATURES (7) — AKURAT dari user input
        # ============================================================
        features["has_company_profile"] = 1 if job.company_profile.strip() else 0
        features["has_requirements"]    = 1 if job.requirements.strip() else 0
        features["has_benefits"]        = 1 if job.benefits.strip() else 0
        features["has_salary_range"]    = 1 if job.salary_range.strip() else 0
        features["has_company_logo"]    = 1 if job.has_company_logo else 0
        features["telecommuting"]       = 1 if job.telecommuting else 0
        features["has_questions"]       = 1 if job.has_questions else 0

        # ============================================================
        # 6. COMBINE TF-IDF + 13 ENGINEERED (5013 total)
        # ============================================================
        eng_vec = np.array(
            [[features[f] for f in ENGINEERED_FEATURES]],
            dtype=np.float64
        )
        X = hstack([tfidf_vec, csr_matrix(eng_vec)])

        # ============================================================
        # 7. PREDICT
        # ============================================================
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

        # ============================================================
        # 8. RISK CLASSIFICATION (4-tier)
        # ============================================================
        risk = get_risk_level(fake_prob)

        # ============================================================
        # 9. BUILD CONTEXTUAL FLAGS
        # ============================================================
        flags = []
        text_lower = full_text.lower()
        is_suspicious = risk["level"] in ("HIGH_RISK", "MEDIUM_RISK", "LOW_RISK")

        if is_suspicious:
            if any(w in text_lower for w in ["earn", "income", "from home", "easy money", "no experience"]):
                flags.append("Suspicious compensation or work-from-home language detected")
            if features["exclamation_count"] > 3:
                flags.append(f"Excessive exclamation marks ({features['exclamation_count']} found)")
            if features["word_count"] < 80:
                flags.append("Unusually short job description")
            if features["uppercase_count"] > 50:
                flags.append("Heavy use of UPPERCASE text")
            if any(w in text_lower for w in ["bank", "wire transfer", "send money", "bank account details", "bank details"]):
                flags.append("Requests financial or banking information")
            if any(w in text_lower for w in ["limited spots", "apply now", "urgent", "act fast"]):
                flags.append("Uses urgency tactics")
            if not features["has_company_profile"]:
                flags.append("Missing company background information")
            if not features["has_requirements"]:
                flags.append("No specific requirements listed")
            if not features["has_company_logo"]:
                flags.append("No company logo provided")
            if not flags:
                flags.append("Pattern matches known fraudulent postings")
        else:
            if features["word_count"] > 200:
                flags.append("Detailed and comprehensive description")
            if features["has_requirements"]:
                flags.append("Clear job requirements provided")
            if features["has_benefits"]:
                flags.append("Benefits and compensation stated")
            if features["has_company_logo"]:
                flags.append("Company logo present")
            if features["has_company_profile"]:
                flags.append("Company background information provided")
            if not flags:
                flags.append("Language patterns consistent with legitimate postings")

        # ============================================================
        # 10. RESPONSE
        # ============================================================
        return PredictionOutput(
            risk_level=risk["level"],
            risk_label=risk["label"],
            risk_message=risk["message"],
            fake_probability=fake_prob,
            real_probability=real_prob,
            flags=flags,
            features_summary={
                "total_words":         features["word_count"],
                "raw_word_count":      features["raw_word_count"],
                "uppercase_count":     features["uppercase_count"],
                "exclamation_marks":   features["exclamation_count"],
                "has_company_profile": bool(features["has_company_profile"]),
                "has_requirements":    bool(features["has_requirements"]),
                "has_benefits":        bool(features["has_benefits"]),
                "has_salary_range":    bool(features["has_salary_range"]),
                "has_company_logo":    bool(features["has_company_logo"]),
                "telecommuting":       bool(features["telecommuting"]),
                "has_questions":       bool(features["has_questions"]),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))