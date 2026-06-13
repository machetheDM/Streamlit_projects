# Deployment Guide — ZA-STEM Insights

## Phase C — Deploy to Streamlit Community Cloud (free, public URL)

Streamlit Community Cloud hosts the dashboard for free and gives a public link
you can share in interviews.

### Steps

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. Click **"Create app"** → **"Deploy a public app from GitHub"**.
3. Fill in:
   - **Repository:** `machetheDM/Streamlit_projects`
   - **Branch:** `main`
   - **Main file path:** `za-stem-insights/app.py`   ← important (app is in a subfolder)
4. Click **"Deploy"**.

### What happens on first run
- Streamlit Cloud installs everything in `za-stem-insights/requirements.txt`
  (~2-3 min — XGBoost, SHAP, statsmodels are sizeable).
- On first page load, `preprocess.ensure_raw_data()` auto-generates the sample
  dataset (the CSVs are gitignored), so no manual data step is needed.

### Suggested app URL
`za-stem-insights` → `https://za-stem-insights.streamlit.app`
(You can set a custom subdomain in the deploy dialog.)

### If the build is slow or times out
SHAP pulls in `numba`/`llvmlite` which are heavy. If the cloud build struggles,
you can comment out `shap` in `requirements.txt`; the **ML Models** page falls
back gracefully (the SHAP chart simply won't render) and every other page works.

---

## Phase D — GitHub Profile Polish

### 1. Update the repository description
On `github.com/machetheDM/Streamlit_projects` → **About** (gear icon, top right):
- **Description:**
  `SKYLearn-Innovation NPO — community-built data science showcase. ZA-STEM Insights: ML, forecasting, clustering & explainable AI on SA matric data.`
- **Website:** paste your live Streamlit URL once deployed.
- **Topics:** `data-science` `machine-learning` `streamlit` `xgboost` `shap` `south-africa` `education` `time-series`

### 2. Pin your top repositories
On `github.com/machetheDM` → **"Customize your pins"**:
1. `ml-ids-zero-trust-cloud` — MSc cybersecurity research
2. `edu-portal-showcase` — full-stack Next.js platform
3. `edu-analytics-showcase` — PySide6 + ML desktop app
4. `Streamlit_projects` — this SKYLearn-Innovation showcase

### 3. Add the live link to your portfolio
Once you have the Streamlit URL, tell me and I'll add a project card for
ZA-STEM Insights to `mahlontebe.org.za/portfolio`.
