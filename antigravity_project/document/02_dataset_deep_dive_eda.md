# Comprehensive Exploratory Data Analysis & Dataset Deep-Dive
**IITM BS Degree — Machine Learning Practice (MLP) Project**  
**Student Roll Number:** 24F2006569 | **Term:** 2026 T3  

---

## 1. Executive Summary & Dataset Overview

The dataset represents a commercial workstation and flexible workspace rental platform operating across 16 major metropolitan clusters. The analytical objective is to predict `price_tier`, a multiclass ordinal categorical variable representing 4 distinct price categories ($1, 2, 3, 4$), using a rich feature set spanning physical space specifications, booking flexibility, operator reputations, geographic coordinates, and historical review metrics.

### Key Dimensions & File Layout
| File Name | Row Count ($N$) | Column Count ($D$) | Memory Footprint (Uncompressed) | Role in Project |
| :--- | :---: | :---: | :---: | :--- |
| `train.csv` | **279,746** | **60** | ~195.0 MB | Labeled historical listings used for training, feature engineering, and cross-validation. |
| `test.csv` | **114,538** | **59** | ~79.8 MB | Unlabeled future listings requiring `price_tier` predictions for Kaggle leaderboard evaluation. |
| `sample_submission.csv`| **114,538** | **2** | ~920 KB | Submission schema reference requiring `ID` (corresponding to `space_id`) and `price_tier`. |

---

## 2. Target Variable Analysis (`price_tier`)

The target variable is discrete with four distinct tiers:
- **Tier 1:** Budget / Affordable Entry Workspaces
- **Tier 2:** Mid-tier Standard Commercial Units
- **Tier 3:** Upper Mid-tier / Dedicated Flexible Suites
- **Tier 4:** Premium / Executive Commercial Workstations

### Empirical Distribution in Training Data:
$$\begin{array}{lcccc}
\hline
\textbf{Price Tier} & \textbf{Class Label} & \textbf{Sample Count } (N_k) & \textbf{Class Proportion } (\%) & \textbf{Imbalance Ratio } (N_{\max}/N_k) \\
\hline
\text{Tier 1} & 1 & 66,889 & 23.91\% & 1.22 \\
\text{Tier 2} & 2 & 81,530 & 29.14\% & 1.00 \\
\text{Tier 3} & 3 & 60,839 & 21.75\% & 1.34 \\
\text{Tier 4} & 4 & 70,488 & 25.20\% & 1.16 \\
\hline
\textbf{Total} & - & \mathbf{279,746} & \mathbf{100.00\%} & - \\
\hline
\end{array}$$

### Critical Findings on Target Distribution:
1. **Moderate Class Balance:** The largest class is Tier 2 (29.14%) and the smallest is Tier 3 (21.75%), yielding an imbalance ratio of only $1.34:1$. While extreme class-rebalancing techniques (e.g., SMOTE) are unnecessary, **Stratified K-Fold Cross-Validation** is mandatory to ensure every validation fold preserves exact class proportions.
2. **Metric Alignment with Macro F1:** The competition evaluation metric is **Macro F1 Score**:
   $$\text{Macro } F_1 = \frac{1}{4} \sum_{k=1}^4 \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
   Because Macro F1 gives equal weight to all four classes, misclassifications on the minority class (Tier 3) carry the same penalty as misclassifications on Tier 2. Models must avoid biasing predictions toward the majority class.

---

## 3. Missing Value Diagnostics & Structural Missingness

Out of 60 columns in `train.csv`, exactly 30 contain missing values. The missingness patterns between `train.csv` and `test.csv` are virtually identical, confirming that the data generating process is stationary across train and test partitions.

### Detailed Breakdown of Missing Features:
| Feature Category | Features | Missing in Train (%) | Missing in Test (%) | Mechanism & Underlying Domain Cause |
| :--- | :--- | :---: | :---: | :--- |
| **Geographic** | `borough` | **76.66%** | **75.12%** | **Structural / Missing Not At Random (MNAR):** Listings outside designated administrative boroughs or within unincorporated metro zones lack borough identifiers. Missingness correlates with Price Tier (Tier 1 missingness = 85.6%, Tier 4 missingness = 71.3%). |
| **Review Scores** | `score_overall`, `score_accuracy`, `score_condition`, `score_onboarding`, `score_support`, `score_access`, `score_value`, `first_review_date`, `last_review_date`, `reviews_per_month` | **19.39%** (54,240 rows) | **18.73%** (21,452 rows) | **Structural (MAR / Deterministic):** Exactly 54,236 of the 54,240 missing rows correspond to listings where `review_count == 0`. These represent newly onboarded listings without review history. Imputing with an arbitrary mean/median without an indicator would destroy this valuable freshness signal. |
| **Physical Spaces** | `zones`<br>`utility_rooms`<br>`workstations` | **17.31%**<br>**10.69%**<br>**6.71%** | **18.42%**<br>**10.59%**<br>**7.04%** | **Missing At Random (MAR):** Smaller single-room listings frequently omit zone divisions or specific utility room partitions. |
| **Operator Attributes** | `operator_platform_age_years/months`, `operator_active_years/months`, `operator_portfolio_size`, `operator_certified`, `operator_has_avatar`, `operator_id_verified`, `operator_ref` | **0.22%** (604 rows) | **0.19%** (219 rows) | **Missing Completely At Random (MCAR):** Minor platform telemetry dropouts during operator registration. |
| **Booking Terms** | `bookable`, `min_term_days`, `max_term_days`, `min_term_floor`, `min_term_ceiling`, `max_term_floor`, `max_term_ceiling` | **< 0.05%** | **< 0.05%** | Rare edge cases with unconfigured term rules. |

---

## 4. Anomalies, Redundancies, and Extreme Sentinel Values

Thorough exploratory analysis revealed several subtle data integrity issues that could severely degrade model performance if untreated:

### A. Perfect Multicollinearity & Duplicate Columns
- `min_term_days` vs `min_term_floor`: Exactly identical in **99.985%** of rows ($r = 1.000000$).
- `max_term_days` vs `max_term_ceiling`: Exactly identical in **99.985%** of rows ($r = 1.000000$).
- **Implication:** Retaining both in linear models produces singular or ill-conditioned covariance matrices ($\mathbf{X}^T\mathbf{X}$ non-invertible), causing explosive coefficient variance. `min_term_floor` and `max_term_ceiling` must be dropped.

### B. Extreme Sentinel Values in Rental Durations
- While the median `max_term_days` is **365 days** (1 year) and the 99th percentile is **1,125 days** (approx. 3 years), the maximum value reaches **577,341,108 days**!
- Similarly, `max_term_avg` reaches **2,355,760,000 days**.
- **Root Cause Diagnosis:** In property listing platforms, open-ended or indefinite leases are frequently recorded using artificial sentinel numbers (such as maximum integer limits or arbitrary 9-digit values).
- **Impact on Models:** In unscaled or unclipped models, a single outlier of $5 \times 10^8$ dominates gradients during gradient descent, completely distorting weight updates.
- **Treatment:** 
  1. Create a binary feature `max_term_is_extreme = (max_term_days > 1125).astype(int)`.
  2. Clip rental durations to $[1, 1125]$ and apply logarithmic transformation $\ln(1 + x)$.

---

## 5. Bivariate Relationships & Feature Information Ranking

### Top Numerical Predictors Correlated with `price_tier`:
$$\begin{array}{lcc}
\hline
\textbf{Feature Name} & \textbf{Pearson Correlation } (r) & \textbf{Physical / Domain Interpretation} \\
\hline
\texttt{max\_seats} & \mathbf{+0.467} & \text{Strongest positive driver: workspace seating capacity directly scales rental cost.} \\
\texttt{zones} & \mathbf{+0.371} & \text{Number of partitioned working areas; multi-zone offices command premium tiers.} \\
\texttt{workstations} & \mathbf{+0.340} & \text{Number of dedicated desks/cubicles within the workspace.} \\
\texttt{geo\_lat} & \mathbf{+0.299} & \text{Latitude coordinate; proxy for prime business districts and city centers.} \\
\texttt{utility\_rooms} & \mathbf{+0.254} & \text{Conference, server, and storage rooms included in the space.} \\
\texttt{min\_term\_days} & \mathbf{-0.208} & \text{Longer minimum stay commitments are heavily discounted by operators.} \\
\texttt{open\_slots\_a} (30\text{d}) & \mathbf{-0.152} & \text{High immediate availability correlates with lower tier; premium spaces are booked up.} \\
\texttt{operator\_metro\_private} & \mathbf{-0.146} & \text{Operators with many small private rooms tend to operate in budget tiers.} \\
\hline
\end{array}$$

### Categorical Feature Dominance:
1. **`access_type`:**
   - `shared_zone`: **84.95%** belong to **Tier 1** (budget hot-desking).
   - `private_zone`: **45.38%** Tier 1, **34.88%** Tier 2.
   - `whole_unit`: **30.84%** Tier 4, **24.77%** Tier 3.
   - `managed_suite`: **37.80%** Tier 4 (executive enterprise solutions).
2. **`compliance_status`:**
   - `licensed`: **33.79%** Tier 4, **30.60%** Tier 3, only **8.18%** Tier 1.
   - `unlicensed`: **34.45%** Tier 1.
3. **`metro` Clusters:**
   - Metropolitan areas exhibit massive price divergence:
   - `MTR-07`: **75.40%** Tier 1 (predominantly budget regional cluster).
   - `MTR-09`: **45.97%** Tier 4 (high-cost financial / capital metropolitan hub).
   - `MTR-01`: **43.57%** Tier 4.
   - `MTR-11`: **42.76%** Tier 4.
4. **`facilities` JSON Array:**
   - Listings contain an average of **30.0 facilities** (ranging from 0 to 101) chosen from a universe of **144 distinct facility codes** (`FCL-001` through `FCL-145`).
   - The total number of facilities (`facility_count`) has a strong positive monotonic association with workspace tier.
