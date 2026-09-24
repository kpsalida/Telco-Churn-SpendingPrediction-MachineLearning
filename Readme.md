# Telecom Customer Intelligence: Churn Risk & Spend Prediction Pipeline

An end-to-end Machine Learning strategy and operations project analyzing the **IBM Telco Customer Churn dataset** (7,043 customer profiles). 

The project delivers dual ML capabilities:
1. **Strategic Classification Engine:** Predicts binary customer churn to proactively trigger retention interventions.
2. **Operational Regression Engine:** Reverse-engineers itemized customer billing dynamics to forecast monthly spend, audit contract anomalies, and evaluate product-tier revenue contribution.

---

## My Personal Focus and Key Technical Contributions

Within this collaborative engagement, I was the **architect and author of the complete Operational Regression Engine** (detailed in `Regression_ML_Katerina.ipynb` and Section 4 of the technical report).

### Key Methodological Contributions:
* **Target Feature Formulation:** Identified that raw `Total_Charges` was heavily correlated with contract duration ($r = 0.83$ with `Tenure`). We designed and justified the feature `Avg_Monthly_Charge` (`Total_Charges / Tenure`), reducing the correlation with tenure to 0.25 and isolating the true monthly spending behavior.
* **Leakage-Free Preprocessing Architecture:** Re-engineered the pipeline to eliminate pre-split data leakage. Designed a modular `ColumnTransformer` fitted strictly on training subsets—applying `StandardScaler` to continuous variables and `OneHotEncoder(drop='first', handle_unknown='ignore')` to high-dimensional categorical features.
* **Collinearity Remediation:** Analyzed post-encoding correlation matrices and removed 7 redundant dummy columns (e.g., `_NoInternet` and `_NoPhone` variants) that exhibited perfect linear dependence ($r = 1.0$) with base service indicators.
* **Domain Feature Engineering (`Household_Type`):** Tested behavioral hypotheses by engineering composite customer tiers (`Family`, `Couple`, `Single Parent`, `Single Adult`). Validated that household status yielded zero predictive gain ($R^2$ remained 0.991), proving pricing is strictly additive and service-driven rather than demographically tiered.
* **Multi-Model Benchmark & Optimization:** Evaluated 8 regression architectures (Linear, Ridge, Lasso, SVR Linear/RBF, Random Forest, KNN, XGBoost). Implemented `Lasso` feature pruning (reducing dimensions from 28 to 14) and hyperparameter tuning with cross-validated grid search.
* **Explainable AI (XAI):** Built `SHAP (LinearExplainer & TreeExplainer)` summary and beeswarm visualizations to extract exact monetary price drivers (+€12/mo for Fiber Optic, +€5–8/mo for streaming add-ons, -€20/mo without internet).

---

## End-to-End System Architecture

```text
                        IBM Telco Raw Dataset (7,043 records)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [Strategic Classification]                      [Operational Regression]
         Target: Is_Churned                             Target: Avg_Monthly_Charge
                  │                                               │
    Stratified 70/30 Split                          Data Sanitization (Tenure > 0)
                  │                                               │
    StandardScaler + OneHotEncoder                  ColumnTransformer (No Leakage)
                  │                                               │
       Class Imbalance Handling                    Collinearity Pruning (7 features)
   (class_weight='balanced')                                      │
                  │                                  8-Model Benchmark & Tuning
     Random Forest vs. SVM (RBF)                     (Lasso, SVR, XGBoost, etc.)
                  │                                               │
   Winner: Tuned SVM (75% Recall)                  Winner: Lasso / XGBoost (R² = 0.991)
                  │                                               │
       Permutation Importance                         SHAP Explainer (Price Drivers)

```

<br><br>
> **Project Attribution:**  
> Developed as part of the *ITC6103B1 Applied Machine Learning (Winter Term 2026)* graduate curriculum at **The American College of Greece (Deree)**, under the supervision of Dr. Elena Chatzimichali.  
> *Collaborators:* Kimon Lappas, Ioannis Logothetis, Milena Mirumyan, Katerina Psallida.  
> *Original Team Repository:* [`kitlapp/Telco_ML`](https://github.com/kitlapp/Telco_ML).
