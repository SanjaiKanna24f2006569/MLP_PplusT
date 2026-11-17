# Analysis of Prior Milestones & Data Preprocessing Synthesis
**IITM BS Degree — Machine Learning Practice (MLP) Project**  
**Student Roll Number:** 24F2006569 | **Term:** 2026 T3  

---

## 1. Overview of Prior Submissions

To ensure end-to-end analytical continuity, we conducted a rigorous audit of the student's prior submissions:
1. **Milestone 1 (`24f2006569_26t3_M1 (1).pdf`):** Completed and submitted. Focused on Exploratory Data Analysis, univariate shapes, missingness rates, basic correlation analysis, and data schema verification.
2. **Milestone 2 (`MLP Project Milestone 2_ Data Preprocessing.docx`):** **Not submitted / Skipped.**
3. **Milestone 3 (`24f2006569_26t3_M3.pdf`):** Completed and submitted. Explored early feature engineering experiments (such as `term_range`, `seats_per_portfolio`, and logarithmic transforms).
4. **Milestone 4 (Current Deliverable):** Model Building, Model Families Exploration, Hyperparameter Optimization, and Metric Benchmarking.

Because Milestone 2 was skipped, **Milestone 4 must encapsulate a complete, production-grade preprocessing and feature pipeline** that addresses all data quality issues, prevents data leakage, and provides optimal inputs for every model family.

---

## 2. Milestone 1 Key Findings & Analytical Takeaways

From our deep reading of the student's Milestone 1 report:
- **Target Distribution:** Verified that the 4 price tiers ($1, 2, 3, 4$) are moderately balanced ($23.9\%$, $29.1\%$, $21.8\%$, $25.2\%$).
- **Missingness Discovery:** Identified that `borough` has $76.66\%$ missing values, and review scores have $\sim 19.39\%$ missing values.
- **Physical Capacity Driver:** Observed that physical size attributes (`max_seats`, `zones`, `workstations`) exhibit the strongest positive correlations ($r \approx 0.34 - 0.47$) with higher price tiers.
- **Integrity Check:** Confirmed zero exact duplicate rows and zero duplicate `space_id` values.
- **Anomaly Detection:** Flagged extreme sentinel values in `max_term_days` (up to $577\text{M}$) and `max_term_avg` (up to $2.35\text{B}$).

---

## 3. Milestone 2 Gaps & The Remediation Implemented in Milestone 4

Milestone 2 was designed to cover data preprocessing systematically. In our current pipeline, we fully implemented and solved every requirement originally designated for Milestone 2:

### A. Missing Value Imputation Strategy
- **Numerical Features:** Imputed using **fold-wise medians** computed exclusively on the training split to prevent lookahead bias.
- **Categorical Features:** Imputed using an explicit `'missing'` category level, allowing tree-based and linear models to recognize missingness as an informative signal.
- **Structural Review Missingness:** Added explicit binary indicators `is_missing_reviews = (score_overall.isna())`. Because listings without reviews are simply newly listed workspaces ($N=54,236$), this separates new listings from established listings with low scores.

### B. Outlier & Sentinel Capping
- Capped `max_term_days` and `max_term_avg` at the 99.9th percentile ($1,125$ days).
- Created a binary feature `max_term_is_extreme = (max_term_days > 1125)` to preserve the distinction of open-ended leases.
- Applied monotonic $\ln(1 + x)$ compression to stabilize gradients.

### C. Multicollinearity & Redundancy Removal
- Dropped `min_term_floor` (identical to `min_term_days`, $r=1.000$).
- Dropped `max_term_ceiling` (identical to `max_term_days`, $r=1.000$).
- Unified `operator_platform_age` and `operator_active_age` into total months.

### D. Encodings & Scaling
- **Linear & Naive Bayes Pipeline:** Standardized all continuous features with `StandardScaler` ($\mu=0, \sigma=1$) and One-Hot Encoded low-cardinality nominals (`OneHotEncoder(handle_unknown='ignore')`).
- **Tree & Boosting Pipeline:** Preserved natural numerical ranges and applied Ordinal / Frequency encoding to high-cardinality features (`space_class`, `zone`, `borough`) to avoid explosive dimensional sparsity.

---

## 4. Milestone 3 Engineering Audit & Advanced Enhancements

In Milestone 3, preliminary features like `term_range` and `seats_per_portfolio` were explored on small subsets. In our unified Milestone 4 pipeline, we expanded feature engineering into 6 robust domain categories:
1. **Capacity & Density Ratios:** `seats_per_workstation`, `seats_per_zone`, `workstations_per_zone`, `utility_rooms_per_seat`.
2. **Operator Platform Dynamics:** `total_platform_age_months`, `total_active_age_months`, `operator_inactivity_months`.
3. **Availability & Occupancy Rates:** `avail_rate_30`, `avail_rate_60`, `avail_rate_90`, `avail_rate_365`, `occupancy_30`, `occupancy_365`.
4. **Facility Intelligence:** Parsing the JSON `facilities` column to extract both total facility count (`facility_count`) and binary indicators for the top 10 most informative amenities (`has_FCL-078`, `has_FCL-145`, etc.).
5. **Review Chronometry:** `review_span_days` (tenure between first and last review) and `days_since_last_review` (recency relative to snapshot date).
6. **Location Frequency Signals:** Frequency encodings for `space_class`, `zone`, and `borough`, computed strictly on training folds.
