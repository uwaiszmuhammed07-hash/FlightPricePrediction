# ✈️ Flight Price Prediction — PRCP-1025

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> A machine learning project to predict flight ticket prices — built as part of the Data Science Capstone program (PRCP-1025).

---

## 🗂️ Project Structure

```
FlightPricePrediction/
├── data/
│   └── Flight_Fare.xlsx
├── outputs/
├── PRCP-1025-FlightPricePrediction.ipynb
├── app.py
├── save_model.py
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebook fully first, then save the model
python save_model.py

# 3. Launch the web app
streamlit run app.py
```

---

## 📊 Models Trained

| Model | Description |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 / feature selection |
| Decision Tree | Non-linear splits |
| Random Forest | Ensemble averaging |
| Gradient Boosting | Sequential correction |
| XGBoost | Optimized boosting |
| LightGBM | Fast leaf-wise boosting |

---

## 🏆 Key Findings

- **Airline** is the strongest predictor of price
- **Duration** and **Total Stops** positively correlate with price
- **Seasonal patterns** — prices spike in March–May
- Tree-based models far outperform linear models (R² > 0.82)

---

## 🌐 Deploy on Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → set main file as `app.py` → Deploy

---

**Author:** Uwais · PRCP-1025 · Data Science Capstone
