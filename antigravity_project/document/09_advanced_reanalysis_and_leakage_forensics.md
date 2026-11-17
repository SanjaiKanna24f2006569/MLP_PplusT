# Advanced Data Forensic Audit, Leak-Free Target Statistics & Class Balancing
**IITM BS Degree — Machine Learning Practice (MLP) Project**  
**Student Roll Number:** 24F2006569 | **Term:** 2026 T3  

---

## 1. Executive Summary & Diagnostic Motivation

In response to the analytical ceiling on raw GBDT models, we conducted a forensic comparative audit across `train.csv` and `test.csv`. Our investigation revealed three fundamental structural characteristics of the competition data:
1. **Massive Spatial & Operator Overlap:** Test instances are not drawn from an unseen universe; rather, **99.97%** of test rows belong to zones present in the training set, and **66.25%** of test rows belong to operators who already have listings in train.
2. **Exact Multicollinearity & Redundancy:** Multiple columns (`min_term_floor`, `max_term_ceiling`) are exact bitwise duplicates ($r = 1.000$) of base columns, which introduces noise and slows down split search.
3. **Ordinal Squeeze on the Minority Class (Tier 3):** While Tiers 1 and 4 achieve $F_1 > 0.81$, Tier 3 suffers an $F_1 \approx 0.56$. Standard cross-entropy with default $\arg\max$ thresholds causes Tier 3 instances to be absorbed by Tier 2 (the largest class with 29.14% prevalence).

---

## 2. Train vs. Test Forensic Comparison Table

| Feature Dimension | Training Set (`train.csv`) | Test Set (`test.csv`) | Overlap Metric | Business & Modeling Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Instance Count ($N$)** | 279,746 | 114,538 | Ratio: $2.44 : 1$ | Stationary generation process without temporal covariate drift. |
| **Unique Zones (`zone`)** | 1,380 | 1,312 | **1,296 overlapping (98.78%)** | **99.97% of test listings** fall into known training zones. High-cardinality Target Encoding captures strong neighborhood pricing power. |
| **Operators (`operator_ref`)** | 41,903 | 21,019 | **7,567 overlapping (36.00%)** | **66.25% of test listings** are managed by operators with historical training data. Operators maintain consistent pricing strategies. |
| **Space Classes (`space_class`)**| 131 | 112 | **102 overlapping (99.99%)** | Virtually 100% test coverage. Space classes determine architectural layout and baseline pricing tiers. |
| **Metros (`metro`)** | 16 | 16 | **16 overlapping (100.0%)** | All 16 metropolitan clusters are shared equally between train and test. |
| **Recurring Physical Units** | 31,198 instances | 7,441 test instances | **6.50% exact physical match** | Exact `(geo_lat, geo_lon, max_seats, operator_ref)` matches represent recurring property leases. |

---

## 3. Duplicate & Redundant Column Elimination

Our audit identified several columns that added noise, dimensionality, or computation overhead without providing novel predictive signal:

1. **`min_term_floor`:** Exactly identical to `min_term_days` in **99.985%** of records ($r = 1.000000$). **Action:** Dropped.
2. **`max_term_ceiling`:** Exactly identical to `max_term_days` in **99.985%** of records ($r = 1.000000$). **Action:** Dropped.
3. **`snapshot_start_date` & `snapshot_end_date`:** Static recording dates spanning 2024–2025 that reflect administrative scraping cycles rather than intrinsic workspace qualities. **Action:** Dropped.
4. **`operator_has_avatar`:** 99.8% constant (`'t'`). Near-zero variance. **Action:** Retained as binary or dropped.
5. **`bookable`:** 99.3% constant (`'t'`). Near-zero variance. **Action:** Retained as binary or dropped.

---

## 4. Why Tier 3 is the Bottleneck & How Class Balancing Solves It

### The "Middle-Class Squeeze" Problem:
- **Tier 1 (Budget):** Strong negative indicators (e.g., `access_type == 'shared_zone'` is 85% Tier 1; low seat count; `min_term_days` high). $F_1 = \mathbf{0.8211}$.
- **Tier 4 (Premium):** Strong positive indicators (e.g., `max_seats \ge 8`, `zones \ge 4`, prime `geo_lat`, executive facilities). $F_1 = \mathbf{0.8013}$.
- **Tier 2 (Standard):** Majority class ($N = 81,530$, 29.14%). $F_1 = \mathbf{0.6858}$.
- **Tier 3 (Upper Mid-tier):** Minority class ($N = 60,839$, 21.75%). Sandwiched between Tier 2 and Tier 4. $F_1 = \mathbf{0.5623}$.

Because Tier 2 has 34% more training examples than Tier 3, a standard classifier trained on unweighted cross-entropy assigns higher probability mass to Tier 2 on ambiguous cases. A prediction like $[P_1=0.05, P_2=0.36, P_3=0.34, P_4=0.25]$ defaults to Tier 2 under standard $\arg\max$, destroying Tier 3 Recall!

### The Two-Pronged Solution:
1. **Inverse Frequency Sample Weighting:**
   Assigning loss weights during training:
   $$w_{\text{Tier 1}} = 1.05, \quad w_{\text{Tier 2}} = 0.85, \quad w_{\text{Tier 3}} = \mathbf{1.25}, \quad w_{\text{Tier 4}} = 0.95$$
   Forces gradient descent to pay 47% more penalty for misclassifying Tier 3 relative to Tier 2.
2. **Direct Macro F1 Threshold Optimization (Post-Processing):**
   Instead of hard $\arg\max$, we find optimal class multipliers $\mathbf{w}^* = [w_1^*, w_2^*, w_3^*, w_4^*]$ via Nelder-Mead optimization on the validation predictions:
   $$\hat{y}_i = \arg\max_{k \in \{1,2,3,4\}} \left( w_k^* \cdot \hat{P}_k(\mathbf{x}_i) \right)$$
   This directly maximizes Macro F1 by recalibrating the decision boundaries.

---

## 5. Bayesian Smoothed Out-of-Fold (OOF) Target Encoding

Because `zone`, `space_class`, and `metro` have near 100% test coverage, they carry immense pricing information. To exploit this without target leakage:
- For training instances, we partition data into 5 Stratified folds.
- Target statistics for fold $k$ are computed strictly from the other 4 folds:
  $$\hat{\mu}_{\text{OOF}} = \frac{n \cdot \bar{y}_{\text{fold}} + m \cdot \mu_{\text{global}}}{n + m}$$
  Where $m = 10$ is the Bayesian smoothing prior, shrinking low-frequency categories toward the global dataset mean.
- For test instances, the full training set smoothed statistics are applied.
