# ZA-STEM Insights 📊

**South African Matric Performance Analyser**

A data science + machine learning project built by five community learners from **SKYLearn-Innovation NPO** under the mentorship of Dingaan Mahlatse Machethe.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)

---

## What We Built

An interactive 5-page Streamlit dashboard that:
- Analyses 10 years (2014–2023) of South African NSC matric results
- Compares STEM vs. non-STEM subject performance across 9 provinces
- Identifies province-subject combinations most at risk of failing the national benchmark
- Predicts next-year at-risk status using a **RandomForest classifier** (binary: On Track / At Risk)

---

## Learner Contributions

| Learner | Module | Skills Demonstrated |
|---------|--------|---------------------|
| Learner 1 | `src/preprocess.py` — data loading & cleaning | Python, Pandas, data validation |
| Learner 2 | `notebooks/02_visualisations.ipynb` — charts | Matplotlib, Seaborn, data storytelling |
| Learner 3 | `src/analyse.py` — statistical analysis | Pandas aggregation, trend analysis, EDA |
| Learner 4 | `src/model.py` — ML prediction model | scikit-learn, RandomForest, ROC-AUC |
| Learner 5 | `app.py` — Streamlit dashboard + deployment | Streamlit, Plotly, cloud deployment |

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
│   ├── processed/                 # Clean data + saved model
│   └── generate_sample_data.py    # Synthetic SA matric data generator
├── src/
│   ├── preprocess.py              # Learner 1 — data pipeline
│   ├── analyse.py                 # Learner 3 — statistical analysis
│   └── model.py                   # Learner 4 — RandomForest ML model
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_visualisations.ipynb
│   └── 03_eda_trend_analysis.ipynb
├── app.py                         # Learner 5 — Streamlit dashboard
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
