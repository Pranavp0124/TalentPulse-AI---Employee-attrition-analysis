# 💼 TalentPulse AI — Employee Attrition & Retention Intelligence

### 🎓 College Capstone & Academic Project Showcase | Developed by **Pranav**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Author: Pranav](https://img.shields.io/badge/Author-Pranav-blue.svg)](#-author--project-credits)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/plotly-v5.18+-blueviolet.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](test_pipeline.py)

> **An enterprise-grade, end-to-end Machine Learning web application and decision-support system designed to forecast employee turnover, diagnose organizational friction points, and deliver data-driven retention strategies.**

---

## 👨‍💻 Author & Project Credits

* **Developer:** **Pranav**
* **Project Type:** College Final Year Project / Capstone Portfolio Submission
* **Domain:** Data Science, Machine Learning, Predictive HR Analytics
* **GitHub Repository:** Designed for direct showcase on GitHub and live cloud deployment

---

## 🌟 Executive Summary & Academic Problem Statement

Employee turnover is one of the most critical and expensive challenges facing modern enterprises:
* **The Problem:** When high-performing employees leave unexpectedly, organizations incur severe costs (ranging between **50% and 150%** of an employee's annual salary) in recruitment, lost productivity, and team disruption.
* **Traditional Approach:** Human Resources typically relies on *exit interviews*, which are purely reactive—identifying problems only after the talent has already left.
* **Our Proposed Solution (TalentPulse AI):** Developed by **Pranav**, this system introduces an **AI-driven proactive retention paradigm**. By analyzing 30+ demographic, compensation, environmental, and behavioral signals from the standard IBM HR benchmark dataset, the system detects early attrition signals and equips leaders with personalized intervention playbooks.

---

## 🎯 Project Objectives

1. **Exploratory Data Analysis (EDA):** Identify the key root causes of employee turnover using interactive statistical visualizations.
2. **Predictive Modeling:** Build and evaluate a robust Machine Learning ensemble capable of handling class imbalance to predict voluntary turnover.
3. **Interactive Decision-Support UI:** Develop a production-grade web dashboard using Streamlit and Plotly for real-time risk simulation and batch cohort scoring.
4. **Actionable HR Insights:** Provide automated, personalized retention strategies and calculate financial ROI based on replacement cost metrics.

---

## 🚀 Key Modules & System Features

| Module | Purpose & Capabilities |
| :--- | :--- |
| **📊 Executive KPI Dashboard** | High-level workforce health indicators: Total Headcount (1,470), Real-time Attrition Rate (16.1%), Estimated Turnover Financial Cost ($3.6M+), Role Disparities, and dynamic sidebar filters (Department, Role, Overtime, Age). |
| **🔬 Exploratory Root Cause Analysis (EDA)** | Interactive Plotly deep-dives analyzing the 4 major turnover drivers: **Overtime Burnout**, **Satisfaction vs Work-Life Balance**, **Commute Distance Strain**, and **Managerial Tenure Stagnation**. Includes an interactive Custom Variable Explorer. |
| **🔮 Individual Flight Risk Predictor** | Interactive employee profile configurator with a real-time **Plotly Half-Gauge (0–100%)**, categorical risk status (🟢 Low, 🟡 Moderate, 🔴 High Hazard), automatic detection of primary risk drivers, and an actionable **HR Retention Action Plan**. |
| **📁 Batch Cohort Assessment & Talent Matrix** | Upload enterprise CSV files or load a pre-configured 50-employee test batch. Generates a **Talent Retention Scatter Matrix** and exports an enriched CSV report with assigned retention priorities (`P1 - Immediate Action`, `P2 - Watchlist`, `P3 - Healthy`). |
| **📈 ML Governance & Explainability** | Complete pipeline transparency displaying holdout validation metrics (**ROC-AUC ~0.79**, **Accuracy ~81%**), interactive **Confusion Matrix Heatmap**, and **Top 10 Global Feature Importances**. |

---

## 💡 Key Analytical Discoveries (IBM HR Benchmark)

Through exploratory data mining, several critical trends were identified:
1. **The Overtime Hazard:** Employees logging regular overtime experience an attrition rate of **~30.5%**, compared to only **10.4%** for standard-hours peers—a near **3x risk multiplier**.
2. **The 24-Month Window:** Voluntary departures are heavily concentrated within an employee's first 2 years at the company and under a current manager.
3. **Compensation Benchmark Disparity:** Employees earning below the median salary for their specific job role demonstrate significantly higher flight risk.
4. **Equity Alignment:** Level 0 Stock Option holders represent the highest proportion of departures; equity vesting serves as an effective retention lock-in mechanism.

---

## ⚙️ Machine Learning Architecture & Methodology

```
                  ┌───────────────────────────────┐
                  │   IBM HR Analytics Dataset    │
                  │   (1,470 Rows, 35 Columns)    │
                  └──────────────┬────────────────┘
                                 │
                 Stratified 80/20 Train-Test Split
                                 │
                   ┌─────────────┴─────────────┐
                   │    ColumnTransformer      │
                   ├─────────────┬─────────────┤
                   │StandardScale│OneHotEncode │
                   │  (Numerics) │(Categorical)│
                   └─────────────┬─────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │       Soft-Voting Ensemble      │
                ├─────────────────────────────────┤
                │ • Balanced Logistic Regression  │
                │ • Random Forest (250 Estimators)│
                │ • Gradient Boosting Classifier  │
                └────────────────┬────────────────┘
                                 │
                  Calibrated Decision Threshold
                       (Probability >= 0.40)
                                 │
         ┌───────────────────────┴───────────────────────┐
         │                                               │
 🟢 Low Risk (<35%)     🟡 Moderate (35-65%)      🔴 High Hazard (>65%)
```

### Validation Metrics (20% Holdout Test Set)

* **ROC-AUC Score:** `0.7915` (Excellent discriminatory ability across thresholds)
* **Classification Accuracy:** `80.61%`
* **Minority Recall (Sensitivity):** `53.19%` (Prioritizes identifying genuine flight risks)
* **Minority Precision:** `41.67%`
* **Macro F1-Score:** `0.67`

---

## 🎓 College Viva & Presentation Q&A Reference

* **Q: Why did you choose a Soft-Voting Ensemble over a single model?**
  * *A:* Individual models have complementary strengths. Logistic Regression provides calibrated, smooth probability baselines; Random Forest captures complex non-linear feature interactions; and Gradient Boosting minimizes residual errors. The ensemble produces more stable, generalizable risk scores.
* **Q: How does the model handle class imbalance (16% Attrition vs 84% Retention)?**
  * *A:* We utilized stratified train-test splits, balanced class weighting (`class_weight='balanced'`), and calibrated the decision threshold to `0.40` to maintain high sensitivity (recall) on the minority class.
* **Q: What are the primary predictive features?**
  * *A:* Monthly Income, Age, Total Working Years, Years at Company, OverTime status, Distance From Home, and Job Satisfaction.

---

## 📂 Project Directory Structure

```
├── .streamlit/
│   └── config.toml             # Custom enterprise dark theme & UI settings
├── app.py                      # Multi-tab Streamlit dashboard application
├── attrition_model.py          # Machine learning pipeline training, evaluation & export
├── generate_dummy_data.py      # Synthetic cohort generator matching IBM HR schema
├── HR_Employee_Attrition.csv   # Standard benchmark IBM HR dataset (1,470 records)
├── sample_test_cohort.csv      # Sample 50-employee test batch for instant demonstration
├── test_pipeline.py            # Automated unit and integration test suite
├── requirements.txt            # Production dependencies
├── Dockerfile                  # Containerized deployment blueprint
├── .dockerignore               # Docker ignore rules
├── .gitignore                  # Git tracking exclusions
├── LICENSE                     # MIT License (Pranav)
└── README.md                   # College documentation & project guide
```

---

## 🛠️ Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pranav/employee-attrition-analysis.git
cd employee-attrition-analysis
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Tests
```bash
python test_pipeline.py
```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ 1-Click Cloud Deployment (Streamlit Community Cloud)

1. Push this repository to your **GitHub** account (`github.com/pranav/...`).
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select your repository and specify `app.py` as the main entrypoint.
4. Click **Deploy** to generate a live, shareable public URL to showcase to professors, peers, and recruiters!

---

## 📜 License

Created by **Pranav** under the [MIT License](LICENSE). Free for academic and demonstration use.
