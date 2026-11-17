# IITM BS Degree — Machine Learning Practice (MLP) Project

**Student Roll Number:** `24f2006569`  
**Student Name:** Sanjai Kanna  
**Academic Term:** 2026 T3  
**Competition:** Workspace Rental Price Tier Prediction (Kaggle)  
**Registered Kaggle Notebook:** `24f2006569-notebook-2026t3`  
**Current Public Leaderboard Best Score:** `0.69873` (Macro F1)  

---

## 📁 Repository Structure

```
IITM BS MLP/
├── 24f2006569-notebook-2026t3.ipynb      # Registered standalone Kaggle notebook
├── GA_2_Assessment_Practice.ipynb         # Graded Assessment 2 practice
├── GA_2_dataset.csv                       # GA 2 practice dataset
│
├── antigravity_project/                   # Main Milestone 4 & Final Project Workspace
│   ├── 24f2006569-notebook-2026t3.ipynb   # Official registered Kaggle submission notebook
│   ├── main_submission_notebook.ipynb     # High-performance pipeline mirror
│   ├── FinalNotebook_M4.ipynb             # Milestone 4 baseline notebook
│   ├── 24f2006569_26t3_M4.pdf             # Milestone 4 Report (27 pages, all 10 questions answered)
│   ├── 24f2006569_26t3_M4.docx            # Milestone 4 Report Word document
│   ├── 24f2006569_26t3_M1 (1).pdf         # Milestone 1 submission
│   ├── 24f2006569_26t3_M3.pdf             # Milestone 3 submission
│   │
│   ├── src/                               # Production scripts & modeling pipeline
│   │   ├── feature_engineering.py         # Base domain feature extractor
│   │   ├── train_and_evaluate.py          # 8-model benchmark suite (M4 Q1-Q10)
│   │   ├── train_xgboost_catboost.py      # GBDT Holy Trinity training
│   │   ├── train_full_power_pipeline.py   # Full 100% data training pipeline
│   │   └── build_full_power_notebook.py   # Automated notebook compiler
│   │
│   ├── data/                              # Competition artifacts & submissions
│   │   ├── sample_submission.csv          # Official Kaggle format template
│   │   ├── submission.csv                 # Latest verified Kaggle submission (114,538 rows)
│   │   ├── submission_full_power_100pct.csv
│   │   └── submission_true_breakthrough.csv
│   │
│   ├── figures/                           # High-resolution report visualizations (11 plots)
│   │   ├── fig_q1_1.png                   # Model performance comparison (Macro F1)
│   │   ├── fig_q2_1.png                   # Multiclass Confusion Matrix
│   │   ├── fig_q3_1.png                   # Feature Importances (Top 20)
│   │   ├── fig_q4_1.png                   # Learning Curves (Train vs Val)
│   │   ├── fig_q5_1.png                   # ROC Curves per Price Tier
│   │   ├── fig_q6_1.png                   # Precision-Recall Curves
│   │   ├── fig_q7_1.png                   # Prediction Error Distribution
│   │   ├── fig_q8_1.png                   # Hyperparameter Tuning Validation Curves
│   │   ├── fig_q9_1.png                   # Residual Analysis across Tiers
│   │   ├── fig_q10_1.png                  # Threshold Calibration Impact
│   │   └── fig_boosting_triad_comparison.png # GBDT Holy Trinity architecture
│   │
│   └── document/                          # Comprehensive Knowledge Transfer (KT) & Viva Guides
│       ├── 01_project_guidelines_and_rules.md
│       ├── 02_dataset_deep_dive_eda.md
│       ├── 03_past_milestones_and_preprocessing_review.md
│       ├── 04_milestone_4_detailed_plan_and_architecture.md
│       ├── 05_model_experimentation_results.md
│       ├── 06_comprehensive_kt_plan.md    # Viva defense preparation & FAQs
│       ├── 07_xgboost_catboost_lightgbm_comparison.md
│       ├── 08_how_to_submit_on_kaggle.md
│       ├── 09_advanced_reanalysis_and_leakage_forensics.md
│       └── 10_pushing_past_75_breakthrough_architecture.md
│
└── proj/                                  # Milestone 2 & Data Preprocessing Workspace
    ├── column_audit.csv                   # 60-column data audit table
    ├── column_decision_log.csv            # Preprocessing decision log
    ├── explore.ipynb                      # Exploratory data analysis
    └── MLP_Milestone_2_Preprocessing_Starter (1).ipynb
```

---

## 🏆 Model Architecture & Methodology

* **Target Variable:** Multiclass Workspace `price_tier` ($\{1, 2, 3, 4\}$) evaluated on **Macro F1 Score**.
* **Model Trinity:** Weighted blending of **LightGBM** (40%), **XGBoost** (35%), and **CatBoost** (25%).
* **Key Engineered Signals:**
  - **Compound Value Index ($r = 0.5321$):** Capacity Volume $\times$ Amenity Quality Score across all 144 facility codes.
  - **Compound Capacity Volume ($r = 0.5122$):** $\ln(1 + \text{max\_seats} \times (\text{zones}+1) \times (\text{utility\_rooms}+1))$.
  - **Amenity Quality Index ($r = 0.4629$):** Continuous quality score derived from empirical luxury vs budget pricing distributions.
  - **Regional Bayesian Target Encoding:** District-level pricing power on `zone`, `space_class`, and `metro` with $m=15$ Bayesian smoothing.
  - **Multi-Horizon Occupancy Dynamics:** Availability gradients across 30d, 60d, 90d, and 365d booking windows.
* **Middle-Tier Calibration:** Calibrated decision multipliers ($\boldsymbol{\alpha}^* = [1.0000, 1.0575, 1.0094, 0.9941]$) to resolve the Tier 3 squeeze.
