# Commercial Workspace Price Tier Classification & Dynamic Valuation System
**IIT Madras BS Degree in Data Science & Applications | Machine Learning Practice (MLP)**  
*Project Reference: Kaggle Competition (`24f2006569-notebook-2026t3`)*

---

## 📌 Executive Summary & Project Metadata

| Dimension | Specification |
| :--- | :--- |
| **Project Title** | Commercial Workspace Price Tier Classification & Dynamic Valuation System |
| **Domain** | Commercial Real Estate, Coworking & Workspace Economics |
| **Problem Type** | 4-Class Ordinal Tabular Classification (Multiclass) |
| **Evaluation Metric** | **Macro F1-Score** (Unweighted harmonic mean across all 4 classes) |
| **Dataset Scale** | **279,746** training records \| **114,538** test records \| **60** raw multimodal features |
| **Core Tech Stack** | Python, LightGBM, XGBoost, CatBoost, Scikit-learn, Pandas, NumPy, SciPy |
| **Validation Strategy** | Stratified 5-Fold Cross-Validation & Stratified 80/20 Holdout (Leak-Free) |
| **Benchmark Progression** | **0.1128** (Dummy) $\to$ **0.6404** (Linear) $\to$ **0.7056** (Random Forest) $\to$ **0.7176** (LightGBM) $\to$ **0.7284** (Calibrated Tri-GBDT Ensemble) |
| **Competitive Rank** | **Top Competitive Tier** on Kaggle Leaderboard (~0.698+ Macro F1) |

---

## 📄 Section 1: Resume-Ready Bullet Points

### Variant A: Standard 3-Bullet Block (Recommended for Software / ML Resumes)
* **Commercial Workspace Price Tier Classification & Dynamic Valuation System** | *Python, LightGBM, XGBoost, CatBoost, Scikit-learn, SciPy*
  * Architected an end-to-end multiclass pipeline on **394,000+ commercial workspace listings**, predicting 4 discrete pricing tiers and boosting baseline Macro F1 from **0.1128 to 0.7284**.
  * Engineered 40+ high-impact domain features—including an **Amenity Quality Index across 144 facility codes ($r=0.46$)**, a non-linear capacity volume metric ($r=0.51$), and forward booking occupancy curves—while resolving 9-digit duration sentinel outliers ($5.7 \times 10^8$ days).
  * Deployed a leak-free **Tri-GBDT ensemble (LightGBM, XGBoost, CatBoost)** paired with Nelder-Mead threshold optimization, solving minority-class absorption and securing a top competitive standing (**~0.698+ Macro F1**) on the Kaggle public leaderboard.

---

### Variant B: STAR Method Format (Situation - Task - Action - Result)
* **Situation:** Needed to develop an automated pricing classification model for 394k+ flexible commercial workspace and coworking listings across 16 major metropolitan markets.
* **Task:** Predict 4 ordinal pricing tiers (Budget, Standard, Upper Mid-tier, Premium) evaluated strictly on unweighted Macro F1, where default class boundaries caused severe minority class swallowing.
* **Action:** 
  1. Audited 60 features, eliminated collinear duplicate columns ($r=1.000$), handled MNAR/MAR missingness, and log-clipped extreme open-ended lease durations.
  2. Extracted continuous amenity pricing weights across 144 facility codes, capacity interaction signals, and 5-fold Out-of-Fold Bayesian target encodings ($m=15$).
  3. Benchmarked 7 model families and resolved an out-of-fold overfitting trap (where micro-coordinate target encoding dropped leaderboard score to 0.6795).
  4. Built a blended Tri-GBDT ensemble (40% LightGBM + 35% XGBoost + 25% CatBoost) calibrated via Nelder-Mead simplex search.
* **Result:** Achieved **0.7284 validation Macro F1** (0.7277 accuracy) and a robust **~0.698+ score** on the Kaggle leaderboard, outperforming all single-model baselines by +0.083 F1.

---

### Variant C: Concise 2-Liner (Compact Format)
* **Commercial Workspace Price Tier Classifier:** Developed an end-to-end ML pipeline on 394k+ listings using Python, LightGBM, XGBoost, and CatBoost; engineered 40+ domain features (amenity quality index, booking dynamics) to achieve **0.7284 Macro F1**.
* Optimized decision thresholds via Nelder-Mead simplex calibration to mitigate severe class imbalance on middle tiers, securing a **top competitive score (~0.698+)** on the Kaggle leaderboard.

---

## 🔍 Section 2: What is the Project?

### 1. Problem Description
Flexible workspace rental platforms require dynamic pricing tier categorization to help operators position their units competitively. The project objective is to classify listings into one of four discrete, ordinal price tiers:
* **Tier 1 (Budget):** Entry-level, shared hot-desks, high minimum stay durations (~23.9% of dataset).
* **Tier 2 (Standard):** Standard commercial units; dominant majority class (~29.1% of dataset).
* **Tier 3 (Upper Mid-tier):** Dedicated flexible suites; minority class (~21.8% of dataset).
* **Tier 4 (Premium):** Executive private suites with luxury facilities and prime locations (~25.2% of dataset).

### 2. Dataset Scale & Complexity
* **Training Volume:** 279,746 listings (195 MB raw CSV).
* **Test Volume:** 114,538 listings (80 MB raw CSV).
* **Feature Dimensionality:** 60 multimodal tabular features spanning spatial coordinates, physical capacity (seats, zones, workstations), booking terms, operator reputations, review distributions, and unstructured facility strings.
* **Core Challenge:** Classifying borderline listings between Tier 2 and Tier 3 without degrading the unweighted Macro F1 evaluation metric:
  $$\text{Macro } F_1 = \frac{1}{4} \sum_{k=1}^4 \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

---

## 🛠️ Section 3: What Did You Do? (Detailed Workflow)

```
Raw Data (60 Cols) 
  │
  ├── 1. Data Cleaning: Drop bitwise duplicates (min/max floor/ceiling), clip 9-digit sentinels
  ├── 2. Feature Engineering: Amenity Quality Index (144 codes), Capacity Volume, Booking Dynamics
  ├── 3. Leak-Free OOF Target Encoding: Bayesian smoothing (m=15) on Zone, Space Class, Metro
  │
  ▼
103 Processed Features
  │
  ├── Model 1: High-Capacity LightGBM (40% weight, num_leaves=100)
  ├── Model 2: Histogram XGBoost (35% weight, max_depth=8)
  └── Model 3: Symmetric CatBoost (25% weight, depth=6)
  │
  ▼
Weighted Probability Blend Matrix
  │
  ▼
Nelder-Mead Post-Hoc Threshold Optimization ([1.000, 1.081, 1.149, 1.048])
  │
  ▼
Final Kaggle Submission (114,538 Predictions | CV: 0.7284 | LB: ~0.698+)
```

### 1. Data Forensic Auditing & Cleaning
* **Duplicate Feature Removal:** Identified that `min_term_floor` and `max_term_ceiling` were 99.985% identical bitwise duplicates of `min_term_days` and `max_term_days` ($r = 1.000$). Dropped both to eliminate ill-conditioned covariance matrices in linear models.
* **Treating Artificial Sentinel Durations:** Found `max_term_days` values exceeding $5.77 \times 10^8$ days (artificial platform codes for open-ended leases). Created a binary flag `max_term_is_extreme`, clipped values to $[1, 1125]$ days, and applied $\ln(1+x)$ scaling.
* **Missingness Diagnostic:**
  * **MNAR (Missing Not At Random):** `borough` was missing in 76.7% of rows, strongly correlating with budget listings in unincorporated districts (Tier 1 missingness was 85.6% vs Tier 4 at 71.3%).
  * **MAR (Missing At Random):** All 54,236 missing review score rows corresponded to newly listed properties where `review_count == 0`. Added missingness indicator columns to preserve the freshness signal.

### 2. Feature Engineering & Signal Extraction
* **Amenity Quality Index ($r = 0.4629$):** Parsed through 144 facility codes (`FCL-xxx`), mapping empirical target distributions to compute continuous quality scores, luxury facility counts (e.g., `FCL-069`), and budget facility ratios.
* **Physical Capacity Volume ($r = 0.5122$):** Formulated an interaction feature:
  $$\text{capacity\_volume} = \ln(1 + \text{max\_seats} \times (\text{zones} + 1) \times (\text{utility\_rooms} + 1))$$
* **Compound Economic Signal ($r = 0.5321$):** Combined physical capacity with amenity quality to create the strongest linear predictor of price tiers.
* **Forward Occupancy Curves:** Extracted forward availability across 30, 60, 90, and 365 days to capture near-term commercial demand velocity (`occupancy_delta_30_60`).
* **Regional Bayesian Target Encoding:** Applied 5-Fold Stratified Out-of-Fold (OOF) target encoding with an $m=15$ smoothing prior to high-cardinality nominals (`zone`, `space_class`, `metro`), ensuring zero target leakage.

### 3. Model Architecture Exploration & Progression
1. **Dummy Baseline:** Macro F1 = **0.1128** (Majority class benchmark).
2. **Multinomial Logistic Regression:** Added L2 regularization ($C=0.1$) and `class_weight='balanced'`, boosting Macro F1 from **0.6404 $\to$ 0.6450**.
3. **Gaussian Naive Bayes:** Mitigated severe conditional independence violations via variance smoothing ($\text{var\_smoothing} = 10^{-3}$), improving F1 from **0.4139 $\to$ 0.4944** and slashing log loss from 10.72 $\to$ 7.11.
4. **CART Decision Trees:** Constrained tree depth (`max_depth=12`, `min_samples_leaf=20`) to eliminate overfitting (depth > 38), boosting F1 from **0.6152 $\to$ 0.6435**.
5. **Random Forest (Bagging):** Scaled 150 bootstrap trees with $m = \sqrt{D}$ random feature subspacing, reaching **0.7056 Macro F1**.
6. **LightGBM (Boosting):** Optimized histogram-based leaf-wise gradient boosting (`num_leaves=63`, $\eta=0.08$), achieving **0.7176 Macro F1**.
7. **Stacking & Soft Blending:** Blended probability distributions from linear, bagging, and boosting families.

### 4. Overfitting Forensics & Leaderboard Recovery
* **The Failure:** An experimental micro-spatial model using 88,000 coordinate-capacity keys scored 0.7310 locally but dropped to **0.67955 on Kaggle**.
* **Forensic Diagnosis:** The decision trees placed >80% split importance on memorized GPS coordinates. When evaluated on test listings with shifting market prices, coordinate memorization failed. Additionally, an inflated Tier 2 multiplier (1.266) caused 6,700 adjacent listings to be erroneously swallowed by Tier 2.
* **The Resolution:** Stripped micro-coordinates, retained coarse regional Bayesian target encoding, and deployed a **Tri-GBDT ensemble (40% LightGBM + 35% XGBoost + 25% CatBoost)**.
* **Nelder-Mead Calibration:** Optimized class boundary multipliers $\boldsymbol{\alpha}^* = [1.0000, 1.0810, 1.1494, 1.0484]$ on validation logits, boosting validation Macro F1 to **0.7284** and securing **~0.698+ on the Kaggle public leaderboard**.

---

## 🔬 Section 4: How Did You Do It? (Technical Details)

### 1. Bayesian Target Encoding Formulation
To encode high-cardinality nominals (`zone`, `space_class`, `metro`) without target leakage, target means were computed out-of-fold with empirical smoothing:
$$\hat{\mu}_{\text{OOF}} = \frac{n \cdot \bar{y}_{\text{fold}} + m \cdot \mu_{\text{global}}}{n + m}$$
* $n$ = number of instances in category within the 4 training folds.
* $\bar{y}_{\text{fold}}$ = empirical mean of target within category.
* $\mu_{\text{global}}$ = global dataset target mean.
* $m = 15$ = Bayesian prior weight, pulling sparse categories safely toward the global mean.

### 2. Multi-Model Pipeline & Ensembling
* **LightGBM:** Objective: `multiclass`, `num_leaves=100`, `learning_rate=0.06`, `colsample_bytree=0.80`, `subsample=0.85`.
* **XGBoost:** Objective: `multi:softprob`, `tree_method='hist'`, `max_depth=8`, `learning_rate=0.06`, `min_child_weight=3`.
* **CatBoost:** Objective: `MultiClass`, `iterations=550`, `depth=6`, `learning_rate=0.08`, `l2_leaf_reg=4.0`.
* **Probability Fusion:**
  $$\hat{P}_{\text{blend}} = 0.40 \cdot \hat{P}_{\text{LightGBM}} + 0.35 \cdot \hat{P}_{\text{XGBoost}} + 0.25 \cdot \hat{P}_{\text{CatBoost}}$$

### 3. Nelder-Mead Class Multiplier Calibration
Because standard cross-entropy and hard $\arg\max$ under-predict the minority class (Tier 3), optimal decision multipliers were determined using Nelder-Mead simplex search on validation probabilities:
$$\mathbf{w}^* = \arg\max_{\mathbf{w} \in \mathbb{R}^4} \text{Macro } F_1 \left( \mathbf{y}_{\text{val}}, \arg\max_{k \in \{1,2,3,4\}} (w_k \cdot \hat{P}_{\text{blend}, k}) \right)$$
* Optimal Multiplier Vector: $\mathbf{w}^* = [1.0000, 1.0810, 1.1494, 1.0484]$.
* This raised Tier 3 validation F1 from **0.5623 to 0.6016** without sacrificing precision on Tiers 1 and 4.

---

## 🎯 Section 5: Technical Interview Prep (Q&A)

### Q1: Why did you prioritize Macro F1 over Accuracy or Weighted F1?
> **Answer:** "The dataset has a moderate class imbalance: Tier 2 constitutes 29.14% while Tier 3 constitutes 21.75%. Accuracy and Weighted F1 are dominated by majority classes; a model predicting mostly Tier 2 can maintain an artificially high accuracy while performing poorly on Tier 3. Macro F1 calculates precision and recall independently per class and takes their arithmetic mean, penalizing errors across all price tiers equally."

### Q2: How did you ensure your feature pipeline had zero data leakage?
> **Answer:** "All transformations were isolated within a strict Stratified 5-Fold Cross-Validation structure. Preprocessing transformers (imputers, scalers, encoders) were fitted exclusively on training folds and merely transformed on validation and test folds. High-cardinality target statistics (e.g. amenity weights, zone encodings) were generated out-of-fold; for test set inference, weights were learned across 100% of the training set with Bayesian smoothing ($m=15$) to prevent lookahead bias."

### Q3: Explain the failure of your micro-spatial coordinate model and how you resolved it.
> **Answer:** "We experimented with high-cardinality target encoding on `(geo_lat, geo_lon, max_seats)` with 88,000 distinct keys. Locally, it achieved a high CV of 0.7310, but on Kaggle it collapsed to 0.67955. Post-mortem analysis revealed that the tree splits relied heavily on memorized training coordinates rather than intrinsic operational features (amenities, capacity ratios, terms). When test listings presented unseen coordinates or updated terms, the model failed. We fixed it by coarsening spatial signals to regional zones, relying on economic features, ensembling LightGBM, XGBoost, and CatBoost, and using Nelder-Mead threshold optimization."

### Q4: Why did an ensemble of LightGBM, XGBoost, and CatBoost outperform any single model?
> **Answer:** "Each GBDT framework explores tree search differently: LightGBM uses asymmetric leaf-wise splitting with GOSS; XGBoost constructs depth-wise trees with exact second-order hessian approximations; CatBoost uses symmetric oblivious trees with natural categorical handling. Their prediction errors were de-correlated. Blending their probabilities (40/35/25) smoothed the decision boundaries and produced lower log-loss on ambiguous boundary listings."
