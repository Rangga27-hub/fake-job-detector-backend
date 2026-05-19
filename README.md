---
title: Fake Job Detector API
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Fake Job Detector API

FastAPI backend untuk deteksi job posting fraud menggunakan TF-IDF + LinearSVC.

## Endpoints

- `GET /` — health check
- `POST /predict` — analyze job posting text
- `GET /docs` — Swagger UI

## Model

- Algorithm: LinearSVC dengan Platt scaling
- Features: 5,000 TF-IDF terms
- Training: 17K+ real/fake job postings