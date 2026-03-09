# 🦟 DengueScan Hybrid — Multi-Center Dengue Diagnostic System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dengue-scan-hybrid.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Research](https://img.shields.io/badge/Status-Research%20Prototype-purple.svg)]()

> A robust machine learning framework for early dengue fever screening, validated on multi-center clinical data from **Munshiganj** and **Jamalpur** districts, Bangladesh.

---

## 📌 Overview

DengueScan Hybrid is a two-stage, safety-critical clinical decision support system (CDSS) designed to assist frontline healthcare workers in dengue-endemic, low-resource settings. It combines a **symptom-based screening model** with a **hematological confirmation model** into a unified hybrid pipeline — prioritizing recall to minimize missed diagnoses.

This project is developed as part of an undergraduate research study on **cross-regional generalization of machine learning models for dengue fever diagnosis in low-resource clinical settings**.

---

## 🚀 Key Features

- **Hybrid Two-Stage Diagnosis** — Symptom screening (Stage 1) feeds into hematological confirmation (Stage 2)
- **Safety-Critical Threshold** — Optimized decision threshold (0.25) minimizes false negatives
- **Multi-Center Validation** — Trained and tested across two geographically distinct districts
- **Explainable AI** — SHAP values provide transparent, interpretable predictions
- **Lightweight Deployment** — Runs entirely on standard CBC lab values; no advanced imaging required

---

## 📊 Model Performance

| Stage | Model Type | Accuracy | Sensitivity (Recall) |
|-------|-----------|----------|----------------------|
| Stage 1 | Symptom Screening | 76.0% | **96.4%** |
| Stage 2 | Lab Confirmation (CBC) | 83.3% | **99.1%** |

> High recall is prioritized over precision — in a screening context, missing a dengue case is more costly than a false alarm.

---

## 🛠️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/CodeCraftRisan/DengueScan_Hybrid.git
cd DengueScan_Hybrid
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

> Make sure `dengue_model_final.pkl`, `model_features.pkl`, and `dengue_model_clinical.pkl` are in the same directory as `app.py`.
---
```

---

## 🔬 Research Context

This system was developed and evaluated using clinical data collected from primary healthcare facilities in **Munshiganj** and **Jamalpur**, Bangladesh. The cross-regional design tests whether a model trained in one district generalizes to another — a critical challenge for ML deployment in real-world health systems.

**Key research questions addressed:**
- Can symptom-only models achieve clinically acceptable recall for dengue screening?
- Does combining symptom and CBC features improve diagnostic confidence?
- How well does a model generalize across geographically distinct patient populations?

---

## ⚠️ Disclaimer

Developed as part of an undergraduate research study on cross-regional generalization of machine learning models for dengue fever diagnosis in low-resource clinical settings. **Not intended for clinical diagnosis or medical decisions.** All model outputs are probabilistic estimates and must be interpreted by a qualified healthcare professional.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

