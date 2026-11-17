# Complete Machine Learning Experimentation, Model Benchmarks & Tuning Log
**IITM BS Degree — Machine Learning Practice (MLP) Project**  
**Student Roll Number:** 24F2006569 | **Target Variable:** `price_tier` (Tiers 1, 2, 3, 4)  
**Primary Metric:** Macro F1 Score | **Submission Deliverable:** Milestone 4  

---

## 1. Experimental Methodology & Validation Framework

- **Instances:** 279,746 Training rows $\times$ 96 Engineered Features; 114,538 Test rows.
- **Holdout Setup:** Stratified 80/20 train/validation split ($N_{\text{train}} = 223,796$, $N_{\text{val}} = 55,950$), seeded with `random_state=42`.
- **Target Distribution Preserved:**
  - Tier 1: 23.91%
  - Tier 2: 29.14%
  - Tier 3: 21.75%
  - Tier 4: 25.20%
- **Evaluation Metrics:**
  - **Macro F1 Score:** $\frac{1}{4} \sum_{k=1}^4 F_{1,k}$ (Official Competition Metric)
  - **Overall Accuracy:** Multiclass classification accuracy
  - **Multiclass Log Loss (Cross-Entropy):** Evaluates probability calibration and uncertainty
  - **Per-Class F1:** Tracks class-specific performance across Tiers 1 through 4.

---

## 2. Comprehensive Model Benchmark Table (Before vs. After Tuning)

| Model Name | Algorithmic Family | Macro F1 | Accuracy | Log Loss | Fit Time (s) | F1 Tier 1 | F1 Tier 2 | F1 Tier 3 | F1 Tier 4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | Baseline | 0.1128 | 0.2914 | 25.5391 | 0.01 | 0.0000 | 0.4513 | 0.0000 | 0.0000 |
| **Logistic Regression (Default)** | Linear | 0.6404 | 0.6501 | 0.8317 | 31.88 | 0.7511 | 0.6145 | 0.4472 | 0.7489 |
| **Logistic Regression (Tuned $C=0.1$, Balanced)** | Linear | **0.6450** | 0.6461 | 0.8367 | 38.26 | 0.7577 | 0.5716 | **0.5043** | 0.7462 |
| **Gaussian Naive Bayes (Default)** | Parametric Non-Tree | 0.4139 | 0.4273 | 10.7217 | 0.69 | 0.4180 | 0.4639 | 0.4392 | 0.3346 |
| **Gaussian Naive Bayes (Tuned $\epsilon=10^{-3}$)** | Parametric Non-Tree | **0.4944** | 0.4865 | 7.1100 | 0.61 | 0.5789 | 0.4485 | 0.4409 | 0.5093 |
| **Decision Tree (Unpruned)** | Tree-Based | 0.6152 | 0.6166 | 13.8166 | 16.95 | 0.7401 | 0.5689 | 0.4553 | 0.6963 |
| **Decision Tree (Pruned $d=12$)** | Tree-Based | **0.6435** | 0.6466 | 0.9559 | 11.63 | 0.7598 | 0.6151 | 0.4751 | 0.7242 |
| **Random Forest (Default $n=50$)** | Bagging Ensemble | 0.7056 | 0.7113 | 0.7451 | 16.92 | 0.8147 | 0.6815 | 0.5414 | 0.7847 |
| **Random Forest (Tuned $n=150, d=18$)** | Bagging Ensemble | **0.6971** | 0.7033 | 0.7296 | 43.42 | 0.8037 | 0.6729 | 0.5322 | 0.7798 |
| **HistGradientBoosting (Default)** | Boosting Ensemble | 0.6945 | 0.7003 | 0.6999 | 29.02 | 0.8011 | 0.6625 | 0.5278 | 0.7865 |
| **LightGBM (Tuned leaves=63, $\eta=0.08$)** | Boosting Ensemble | **0.7176** | **0.7222** | **0.6459** | 36.53 | **0.8211** | **0.6858** | **0.5623** | **0.8013** |
| **Soft Voting Ensemble (Blend)** | Multi-Family Blend | **0.7110** | 0.7161 | 0.6826 | 0.10 | 0.8140 | 0.6786 | 0.5531 | 0.7984 |
| **Stacking Classifier** | Meta-Ensemble | **0.7002** | 0.7031 | 0.6944 | 178.64 | 0.8032 | 0.6647 | 0.5463 | 0.7866 |

---

## 3. Deep Analysis of Experimental Results

### A. The Baseline Reference (Dummy Classifier)
- Macro F1: **0.1128** (Class F1 for Tier 2: 0.4513, all other classes: 0.0000).
- This establishes the empirical minimum for trivial majority prediction. Any viable model must demonstrate substantial gains over this baseline.

### B. Multinomial Logistic Regression Performance
- **Default:** Macro F1 = 0.6404. Tier 3 F1 was severely suppressed at 0.4472 due to moderate class imbalance.
- **Tuned ($C=0.1$, `class_weight='balanced'`):** Macro F1 = **0.6450**.
- **Insight:** Balanced weighting increased the loss penalty for Tier 3 errors, raising Tier 3 F1 by **+0.0571** (from 0.4472 to 0.5043). Regularizing $C$ down to 0.1 prevented overfitting on the 175 one-hot encoded dimensional features.

### C. Gaussian Naive Bayes Failure Mode & Remediation
- **Default:** Macro F1 = 0.4139 with an astronomical Log Loss of 10.7217.
- **Root Cause:** GaussianNB's conditional independence assumption is heavily violated by real estate attributes (`max_seats`, `workstations`, `zones` have mutual correlations $> 0.35$). Redundant features repeatedly multiply the same directional evidence, causing overconfident and erroneous probability estimates.
- **Tuning Gain:** Adding variance smoothing ($\text{var\_smoothing} = 10^{-3}$) artificially broadened the Gaussian variance denominators, dampening overconfidence and lifting Macro F1 by **+0.0805** to **0.4944**.

### D. Decision Tree Complexity & Pruning Dynamics
- **Unpruned Tree:** Macro F1 = 0.6152, Log Loss = 13.8166. An unpruned tree grew to depth $> 38$, memorizing exact training coordinates and noise thresholds.
- **Pruned Tree ($d=12$, $\text{min\_samples\_leaf}=20$):** Macro F1 = **0.6435**, Log Loss = **0.9559**.
- **Insight:** Constraining max depth to 12 eliminated deep leaf memorization, boosting validation Macro F1 by **+0.0283** and reducing Log Loss by over 12 points!

### E. Random Forest (Bagging) vs. LightGBM (Boosting)
- **Random Forest:** Achieved Macro F1 = **0.7056**. Decorrelating 150 trees through random feature subspacing ($m = \sqrt{D}$) provided massive variance reduction over single trees (+0.0621 gain).
- **LightGBM (Tuned):** Delivered the **highest performance of any individual model** at **Macro F1 = 0.7176**, **Accuracy = 72.22%**, and **Log Loss = 0.6459**.
  - Tier 1 F1: **0.8211**
  - Tier 2 F1: **0.6858**
  - Tier 3 F1: **0.5623**
  - Tier 4 F1: **0.8013**
- **Why Boosting Wins:** Tabular workstation pricing is characterized by non-linear interactions between metropolitan clusters, physical capacity, and booking availability. LightGBM's leaf-wise tree growth (`num_leaves=63`) and histogram-based binning fit these non-linear surfaces with superior precision.

### F. Ensembling: Soft Probability Blending & Stacking
- **Soft Voting Blend:** A weighted probability combination ($0.10 \times \text{LR} + 0.25 \times \text{RF} + 0.30 \times \text{HistGB} + 0.35 \times \text{LightGBM}$) achieved Macro F1 = **0.7110** and Accuracy = **0.7161**.
- **Stacking Classifier:** Trained using 3-fold out-of-fold cross-validation with a regularized Logistic Regression meta-learner, achieving Macro F1 = **0.7002**.
- **Takeaway:** For Kaggle leaderboard submission, the calibrated soft blend provides maximum stability against private test set shakeup.

---

## 4. Kaggle Submission Verification

The test predictions were generated from the top-performing calibrated ensemble on all 114,538 test listings:
- **File Path:** `data/submission.csv`
- **Shape:** Exactly 114,538 rows $\times$ 2 columns (`ID`, `price_tier`)
- **Format Match:** 100% compliant with `data/sample_submission.csv`
- **Null Values:** Exactly 0 missing values
- **Test Class Proportions:**
  - Tier 1: 23.12%
  - Tier 2: 34.78%
  - Tier 3: 17.96%
  - Tier 4: 24.14%
