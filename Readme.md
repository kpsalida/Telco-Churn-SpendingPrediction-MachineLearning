# Telecom Customer Intelligence: Churn Risk & Spend Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Library-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green.svg)](https://shap.readthedocs.io/)
[![Academic Project](https://img.shields.io/badge/MSc-Data_Science_(Deree)-purple.svg)]()

An end-to-end Machine Learning strategy and operations project analyzing the **IBM Telco Customer Churn dataset** (7,043 customer profiles). 

The project delivers dual ML capabilities:
1. **Strategic Classification Engine:** Predicts binary customer churn to proactively trigger retention interventions.
2. **Operational Regression Engine:** Reverse-engineers itemized customer billing dynamics to forecast monthly spend, audit contract anomalies, and evaluate product-tier revenue contribution.

---

## 🎯 Key Findings & Strategic Recommendations

1. **Focus Retention on Critical Churn Profiles:** Month-to-month subscribers churn at 15× the rate of 2-year contract customers (42.7% vs. 2.8%). Targeted long-term contract conversion campaigns will yield the highest customer lifetime value.
2. **Deploy Tuned SVM for Proactive Intervention:** The optimized SVM (RBF kernel with `class_weight='balanced'`) captures **75.0% of true churners** (421 out of 561 test accounts) compared to Random Forest's 48.0%. This captures **+152 additional at-risk accounts per evaluation cycle**.
3. **Itemized Billing Transparency:** Customer spending behavior is strictly linear ($R^2 = 0.991$, $\text{MAE} \approx €2.02$) and determined purely by service tier subscriptions. Household configuration (`Family`, `Couple`, `Single`) has zero impact on billing.
4. **Targeted Service Upselling & Discounting:** Fiber Optic connectivity contributes **+€12.00/month** to the bill but represents the highest churn risk profile. Bundling premium streaming (+€5 to €8/month) into long-term contracts can protect margin while locking in retention.

---

## 🛠️ Data & Tools

* **Data source:** [IBM Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Data files:** `raw_telco.csv`, `ml_ready_dataset.parquet`, `ml_ready_dataset_nodummies.parquet`
* **Preprocessing, Pipeline & ML:** Python (pandas, scikit-learn, XGBoost)
* **Explainable AI (XAI):** SHAP (`LinearExplainer` & `TreeExplainer`), Permutation Importance
* **Hyperparameter Optimization:** `GridSearchCV`, `RandomizedSearchCV`
* **Reporting & Documentation:** Jupyter Notebook, Google Colab, MS Word, PowerPoint

---

## 📁 Files in This Repository

| File | Description |
| :--- | :--- |
| `cleaning_eda.ipynb` | Exploratory Data Analysis, categorical distributions, outlier checks, and baseline data cleaning. |
| `Regression_ML_Katerina.ipynb` | **[My Contribution]** Full operational regression pipeline: `ColumnTransformer`, collinearity pruning, 8-model benchmark, hyperparameter optimization, and SHAP explainability. |
| `telco_churn_rf_svm.ipynb` | Supervised classification experiments evaluating Random Forest vs. SVM (RBF kernel) with class imbalance adjustments. |
| `read_data.ipynb` | Data loading and inspection routines converting raw data into ML-ready formats. |
| `tuned_xgboost_model.pkl` | Serialized binary file containing the trained and hyperparameter-tuned XGBoost model. |
| `ml_ready_dataset.parquet` | Cleaned and encoded feature matrix saved in compressed Parquet format. |
| `ml_ready_dataset_nodummies.parquet` | Cleaned pre-encoding dataset utilized for modular `ColumnTransformer` pipelines. |
| `raw_telco.csv` | Raw IBM Telco customer dataset (7,043 rows, 21 attributes). |
| `environment.yml` | Conda virtual environment file detailing package versions and project dependencies. |
| `Report.pdf` | Complete academic group research paper (36 pages) detailing methodology, models, and findings. |
| `Presentation.pdf` | Executive stakeholder slide deck summarizing the business challenge, classification, and regression findings. |
| `Readme.md` | Project overview, technical breakdown, and model documentation. |

---

## 👤 My Personal Focus and Key Technical Contributions

Within this collaborative engagement, I was the **architect and author of the complete Operational Regression Engine** (detailed in `Regression_ML_Katerina.ipynb` and Section 4 of `Report.pdf`).

### Key Methodological Contributions:
* **Target Feature Formulation:** Identified that raw `Total_Charges` was heavily correlated with contract duration ($r = 0.83$ with `Tenure`). We designed and justified the feature `Avg_Monthly_Charge` (`Total_Charges / Tenure`), reducing the correlation with tenure to 0.25 and isolating the true monthly spending behavior.
* **Leakage-Free Preprocessing Architecture:** Re-engineered the pipeline to eliminate pre-split data leakage. Designed a modular `ColumnTransformer` fitted strictly on training subsets—applying `StandardScaler` to continuous variables and `OneHotEncoder(drop='first', handle_unknown='ignore')` to categorical features.
* **Collinearity Remediation:** Analyzed post-encoding correlation matrices and removed 7 redundant dummy columns (e.g., `_NoInternet` and `_NoPhone` variants) that exhibited perfect linear dependence ($r = 1.0$) with base service indicators.
* **Domain Feature Engineering (`Household_Type`):** Tested behavioral hypotheses by engineering composite customer tiers (`Family`, `Couple`, `Single Parent`, `Single Adult`). Validated that household status yielded zero predictive gain ($R^2$ remained 0.991), proving pricing is strictly additive and service-driven rather than demographically tiered.
* **Multi-Model Benchmark & Optimization:** Evaluated 8 regression architectures (Linear, Ridge, Lasso, SVR Linear/RBF, Random Forest, KNN, XGBoost). Implemented `Lasso` feature pruning (reducing dimensions from 28 to 14) and hyperparameter tuning with cross-validated grid search.
* **Explainable AI (XAI):** Built `SHAP (LinearExplainer & TreeExplainer)` summary and beeswarm visualizations to extract exact monetary price drivers (+€12/mo for Fiber Optic, +€5–8/mo for streaming add-ons, -€20/mo without internet).

---

## 🔍 In-Depth: Operational Regression Engine (Katerina Psallida)

### 1. Collinearity Fix & Feature Selection
Post-encoding inspection revealed severe multicollinearity:

`Corr(Has_<Service>_NoInternet, Is_InternetService_None) = 1.0`

* Dropped 6 `_NoInternet` fields (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`).
* Dropped `Has_MultipleLines_NoPhone` ($r = 1.0$ with `Has_PhoneService`).
* Dropped 11 unbilled customer accounts with `Tenure = 0`, boosting baseline $R^2$ from **0.985 to 0.991**.

### 2. Multi-Model Benchmark & Hyperparameter Tuning

| Model Architecture | Test $R^2$ | MAE (€) | Hyperparameters & Key Insights |
| :--- | :---: | :---: | :--- |
| **Linear Regression (OLS)** | **0.991** | **2.018** | Baseline parametric model. Captures additive pricing perfectly. |
| **Lasso Regression** | **0.991** | **2.018** | Best $\alpha = 0.01$. Pruned 14 zero-weight features with no loss in accuracy. |
| **Ridge Regression** | **0.991** | **2.018** | Best $\alpha = 0.1$. L2 shrinkage confirms stability of service weights. |
| **Linear SVR** | **0.991** | **2.020** | Support vectors mirror the linear hyperplane. |
| **XGBoost Regressor (Tuned)** | **0.991** | **2.022** | Optimized via GridSearchCV (`max_depth=1`, shallow additive trees). |
| **Random Forest** | 0.988 | 2.337 | 100 estimators. Slight underperformance vs. pure linear models. |
| **SVR (RBF Kernel)** | 0.987 | 2.418 | Non-linear margin mapping. |
| **K-Nearest Neighbors (KNN)** | 0.956 | 4.354 | Distance metric sensitive to binary feature sparsity. |

---

## 💡 Model Explainability & Marginal Price Drivers (Slide 15)

Through `SHAP` analysis (`LinearExplainer` on Lasso and `TreeExplainer` on XGBoost), the "black box" of customer charges was decoded into explicit monetary drivers:

```text
================================================================================================

                         MARGINAL MONTHLY BILLING IMPACT (SHAP / LASSO)

================================================================================================

  [+] Fiber Optic Service
      ████████████████████████████████████████████████    +€12.00 / month

  [+] Streaming Entertainment (Movies / TV)
      ████████████████████████                            +€5.00 to +€8.00 / month

  [+] Phone Service & Multiple Lines
      ██████████████                                      +€3.00 to +€5.00 / month

  [+] Premium Tech Support & Online Security
      ████████                                            +€2.00 to +€3.00 / month

  [0] Demographics (Age, Household, Billing Channel)
      —                                                    €0.00 (Zero billing impact)

  [-] No Internet Service (Baseline Displacement)
      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  -€20.00 / month

================================================================================================