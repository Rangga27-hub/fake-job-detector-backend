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

# Hardcode engineered features (sama persis dengan notebook)
# URUTAN INI TIDAK BOLEH DIUBAH — match dengan training (total 13 features)
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
# PREPROCESSING (sama persis dengan notebook)
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
    """Sama persis dengan notebook cell preprocessing."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in ALL_STOPWORDS)

def extract_numeric_features(raw_text: str, clean_txt: str) -> dict:
    """
    Sama persis dengan notebook:
    - uppercase, exclamation, question → dari full_text (raw)
    - word_count, avg_word_length → dari clean_text
    - raw_word_count → dari full_text sebelum cleaning
    """
    raw_words = raw_text.split()
    clean_words = clean_txt.split()
    return {
        "uppercase_count":   sum(c.isupper() for c in raw_text),
        "exclamation_count": raw_text.count("!"),
        "question_count":    raw_text.count("?"),
        "word_count":        len(clean_words),
        "avg_word_length":   float(np.mean([len(w) for w in clean_words])) if clean_words else 0.0,
        "raw_word_count":    len(raw_words),
    }

def extract_binary_features(raw_text: str) -> dict:
    """
    7 binary features — di EMSCAD dataset asli adalah kolom terpisah.
    Di production, user cuma paste teks, jadi kita generate heuristic dari konten teks.
    """
    text_lower = raw_text.lower()

    # has_company_profile: punya deskripsi tentang perusahaan
    company_keywords = [
        "about us", "our company", "we are", "founded in", "headquartered",
        "mission", "vision", "our team", "company overview"
    ]
    has_company_profile = int(any(kw in text_lower for kw in company_keywords))

    # has_requirements: punya section requirements/qualifications
    req_keywords = [
        "requirement", "qualification", "must have", "required",
        "bachelor", "degree", "years of experience", "minimum",
        "skills needed", "you have", "you should"
    ]
    has_requirements = int(any(kw in text_lower for kw in req_keywords))

    # has_benefits: ada pembahasan benefits/perks
    benefit_keywords = [
        "benefit", "insurance", "health care", "healthcare", "401k",
        "pto", "paid time off", "vacation", "stock option", "equity",
        "bonus", "compensation", "perks"
    ]
    has_benefits = int(any(kw in text_lower for kw in benefit_keywords))

    # has_salary_range: ada angka mata uang / range gaji
    salary_pattern = re.compile(
        r"\$\s?\d+[,\d]*"            # $50,000, $5000
        r"|\bIDR\s?\d+"              # IDR 5000000
        r"|\bRp\s?\d+"               # Rp 5000000
        r"|\bUSD\s?\d+"              # USD 5000
        r"|\d+\s?k\s?(usd|/yr|per)"  # 50k usd, 50k/yr
        r"|\bsalary[:\s]+\$?\d"      # salary: $50000, salary 50000
        , re.IGNORECASE
    )
    has_salary_range = int(bool(salary_pattern.search(raw_text)))

    # has_company_logo: ga bisa dideteksi dari teks pure
    # Default 1 (asumsi posting "lengkap" punya logo di originalnya)
    has_company_logo = 1

    # telecommuting: posting mention remote/WFH
    telecommuting_keywords = [
        "remote", "work from home", "wfh", "telecommute", "telework",
        "work anywhere", "distributed team", "fully remote"
    ]
    telecommuting = int(any(kw in text_lower for kw in telecommuting_keywords))

    # has_questions: ada screening question / pertanyaan ke applicant
    question_keywords = [
        "why do you", "tell us about", "what makes you", "describe a time",
        "screening question", "answer the following"
    ]
    # Atau ada banyak tanda tanya (>2 kemungkinan ada list pertanyaan)
    has_questions = int(
        any(kw in text_lower for kw in question_keywords)
        or raw_text.count("?") > 2
    )

    return {
        "has_company_profile": has_company_profile,
        "has_requirements":    has_requirements,
        "has_benefits":        has_benefits,
        "has_salary_range":    has_salary_range,
        "has_company_logo":    has_company_logo,
        "telecommuting":       telecommuting,
        "has_questions":       has_questions,
    }

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Fake Job Detector API", version="2.0")

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
    return {"status": "ok", "message": "Fake Job Detector API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(job: JobInput):
    try:
        # 1. Clean text
        cleaned = clean_text(job.text)
        cleaned = remove_stopwords(cleaned)

        # 2. TF-IDF transform
        tfidf_vec = TFIDF.transform([cleaned])

        # 3. Engineered features
        numeric = extract_numeric_features(job.text, cleaned)
        binary = extract_binary_features(job.text)
        all_features = {**numeric, **binary}

        # Susun feature vector sesuai urutan ENGINEERED_FEATURES
        eng_vec = np.array(
            [[all_features[f] for f in ENGINEERED_FEATURES]],
            dtype=np.float64
        )

        # 4. Gabung TF-IDF + engineered (total: 5000 + 13 = 5013)
        X = hstack([tfidf_vec, csr_matrix(eng_vec)])

        # 5. Predict
        proba = MODEL.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        # 6. Build flags untuk UI
        flags = []
        text_lower = job.text.lower()
        if pred_class == 1:  # FAKE
            if any(w in text_lower for w in ["earn", "income", "from home", "easy money", "no experience"]):
                flags.append("Suspicious compensation or work-from-home language")
            if numeric["exclamation_count"] > 3:
                flags.append(f"Excessive exclamation marks ({numeric['exclamation_count']} found)")
            if numeric["word_count"] < 80:
                flags.append("Unusually short job description")
            if numeric["uppercase_count"] > 50:
                flags.append("Heavy use of UPPERCASE text")
            if not binary["has_requirements"]:
                flags.append("Missing clear job requirements")
            if not binary["has_company_profile"]:
                flags.append("No company background information")
            if not flags:
                flags.append("Pattern matches known fraudulent postings")
        else:  # REAL
            if numeric["word_count"] > 200:
                flags.append("Detailed and comprehensive description")
            if binary["has_requirements"]:
                flags.append("Specific qualification requirements listed")
            if binary["has_benefits"]:
                flags.append("Benefits and compensation clearly stated")
            if binary["has_company_profile"]:
                flags.append("Company background information provided")
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