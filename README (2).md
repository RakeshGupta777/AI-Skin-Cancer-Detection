# DermAI — Skin Cancer Detection App

A Streamlit web app for classifying 7 types of skin lesions using the HAM10000 dataset.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — Main Streamlit application (756 lines)
- `mlp_model.pkl` — Trained MLP neural network
- `rf_model.pkl` — Trained Random Forest (200 trees)
- `scaler.pkl` — StandardScaler for MLP preprocessing
- `HAM10000_metadata.csv` — 10,015 records with dx, age, sex, localization
- `model_results.json` — Accuracy, confusion matrix, per-class metrics
- `training_history.json` — Epoch-by-epoch training curves
- `sample_images/` — 3 real dermoscopic image collages

## Model Performance
- Ensemble (MLP + RF) Test Accuracy: **82.95%**
- MLP Test Accuracy: **80.68%**
- Random Forest Test Accuracy: **82.95%**

## 5 App Pages
1. **Overview** — Project summary, class descriptions, sample images
2. **Analyze Lesion** — Upload any image → real ML prediction + confidence
3. **Dataset & EDA** — Class distribution, age/sex/localization charts from real metadata
4. **Architecture** — Model pipeline diagram, 14-step methodology
5. **Results** — Training curves, confusion matrix, per-class metrics radar

## Dataset
HAM10000 ("Human Against Machine with 10000 training images")
- 10,015 dermoscopic images · 7 lesion classes · ISIC archive
- Available on Kaggle: skin-cancer-mnist-ham10000
