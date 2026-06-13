# ZA-STEM Insights 📊

**South African Matric Performance Analyser**

An end-to-end data science project built by the **SKYLearn-Innovation NPO** team — four high-school learners, a university tutor, and programme head Dingaan Mahlatse Machethe — covering the full stack from data cleaning to forecasting and explainable ML.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-189FDD?style=for-the-badge)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-7B16D9?style=for-the-badge)](https://shap.readthedocs.io)

---

## What We Built

An interactive **9-page Streamlit dashboard** + 6 Jupyter notebooks that:
- Analyse 10 years (2014–2023) of South African NSC matric results across 9 provinces and 10 subjects
- Compare STEM vs. non-STEM performance with **formal statistical hypothesis testing**
- Cluster provinces into performance tiers using **K-Means unsupervised learning**
- Predict at-risk status by **comparing 3 ML models** (Logistic Regression, Random Forest, XGBoost) with cross-validation and **SHAP explainability**
- **Forecast** future pass rates 3 years ahead using time series (Holt ETS)

### Skills & Technologies Showcased
`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Plotly` `scikit-learn` `XGBoost` `SHAP` `statsmodels` `SciPy` `K-Means` `PCA` `Cross-Validation` `ROC-AUC` `Hypothesis Testing` `Time Series Forecasting` `Streamlit`

---

## Team & Contributions (3 skill tiers)

The SKYLearn-Innovation team spans high-school learners, a university tutor, and the programme head — each contributing at their level so the project covers the **full data science stack**.

### Tier 1 — High School Learners (beginner–intermediate)
| Contributor | Module | Skills |
|-------------|--------|--------|
| Learner 1 | `src/preprocess.py` | Python, Pandas, data cleaning & validation |
| Learner 2 | `notebooks/02_visualisations.ipynb` | Matplotlib, Seaborn, data storytelling |
| Learner 3 | `src/analyse.py`, `notebooks/03` | Pandas aggregation, trend/EDA analysis |
| Learner 4 | `src/model.py` (base RandomForest) | scikit-learn, train/test split, ROC-AUC |

### Tier 2 — Tutor (university undergraduate, intermediate–advanced)
| Module | Skills |
|--------|--------|
| `src/cluster.py` | K-Means, silhouette analysis, PCA, StandardScaler |
| `src/stats_tests.py` | t-test, Mann-Whitney, ANOVA, correlation, regression |
| `notebooks/06_clustering_statistics.ipynb` | Unsupervised ML + statistical inference |

### Tier 3 — Programme Head: Dingaan Machethe (MSc Data Science, UEL — mastery)
| Module | Skills |
|--------|--------|
| `src/model.py` (multi-model) | XGBoost, K-fold CV, model comparison, SHAP explainability |
| `src/forecast.py` | Time series forecasting (Holt ETS), confidence intervals |
| `notebooks/04`, `notebooks/05` | Explainable AI, forecasting |
| `app.py` | 9-page Streamlit architecture, deployment |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data (mirrors real DBE NSC structure)
python data/generate_sample_data.py

# 3. Launch the dashboard
streamlit run app.py
```

> **Real Data:** The DBE publishes official NSC results at [education.gov.za](https://www.education.gov.za). Download the CSV files and place them in `data/raw/` with the column structure matching `data/generate_sample_data.py`.

---

## Project Structure

```
za-stem-insights/
├── .streamlit/config.toml         # Dark theme config
├── data/
│   ├── raw/                       # Raw CSV (DBE or generated)
│   ├── processed/                 # Clean data + saved models
│   └── generate_sample_data.py    # Synthetic SA matric data generator
├── src/
│   ├── preprocess.py              # T1 — data pipeline
│   ├── analyse.py                 # T1 — statistical analysis
│   ├── model.py                   # T1+T3 — RF base + multi-model/CV/SHAP
│   ├── cluster.py                 # T2 — K-Means + PCA clustering
│   ├── stats_tests.py             # T2 — hypothesis testing
│   └── forecast.py                # T3 — time series forecasting
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_visualisations.ipynb
│   ├── 03_eda_trend_analysis.ipynb
│   ├── 04_ml_training_evaluation.ipynb
│   ├── 05_forecasting.ipynb
│   └── 06_clustering_statistics.ipynb
├── app.py                         # 9-page Streamlit dashboard
└── requirements.txt
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, national trend, STEM gap, at-risk table |
| 📈 Trends | Year-by-year pass rate, enrolment growth, STEM gap bar chart |
| 🗺️ Provinces | Province ranking, multi-province trend comparison |
| 📚 Subjects | Subject ranking, deep-dive by subject, province breakdown |
| ⚠️ At-Risk Predictor | ML prediction form + risk gauge + scatter plot |
| 🔮 Forecast | Holt ETS 3-year forecast with 95% confidence intervals |
| 🤖 ML Models | 3-model leaderboard, ROC curves, confusion matrix, SHAP |
| 🧩 Clusters | K-Means province tiers, silhouette analysis, PCA map |
| 🧪 Statistics | t-test, ANOVA, correlation, regression — formal inference |

---

## About SKYLearn-Innovation

SKYLearn-Innovation is a registered South African NPO that provides free STEM and coding education to community learners. Subjects taught include:

- Mathematics & Physical Sciences
- Python Programming
- C# / Unity / .NET
- Introductory Data Science & Analytics
- Introductory Cloud Computing
- Introductory Cybersecurity

**Mentor:** Dingaan Mahlatse Machethe — [Portfolio](https://www.mahlontebe.org.za/portfolio) | [GitHub](https://github.com/machetheDM)
