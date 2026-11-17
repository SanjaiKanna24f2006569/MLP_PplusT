# Comprehensive Knowledge Transfer (KT) Plan & Viva Defense Guide
**IITM BS Degree — Machine Learning Practice (MLP) Project**  
**Student Roll Number:** 24F2006569 | **Target Variable:** `price_tier` (Tiers 1, 2, 3, 4)  
**Evaluation Metric:** Macro F1 Score | **Submission Deadline:** Milestone 4  

---

## 1. Executive Project Summary

This project addresses a real-world commercial real estate challenge: predicting the workstation rental `price_tier` (multiclass classification across 4 discrete ordinal tiers: 1 = Budget, 2 = Standard, 3 = Upper Mid-tier, 4 = Premium) from tabular listings across 16 metropolitan clusters.

### Key Technical Parameters:
- **Total Training Instances ($N_{\text{train}}$):** 279,746 rows
- **Total Test Instances ($N_{\text{test}}$):** 114,538 rows
- **Raw Features:** 59 input columns + 1 target (`price_tier`)
- **Engineered Feature Space:** 96 high-signal features
- **Validation Scheme:** Stratified 80/20 train/validation split + 5-Fold Stratified Cross-Validation
- **Optimization Metric:** Macro F1 Score

---

## 2. End-to-End Pipeline & Analytical Flow

```
[Raw train.csv & test.csv]
          │
          ▼
[Step 1: Data Cleaning & Sanitization]
  ├─ Remove duplicate columns (min_term_floor, max_term_ceiling)
  ├─ Cap sentinel extreme rental durations (clip max_term_days to 1125)
  └─ Flag structured missingness (borough, reviews)
          │
          ▼
[Step 2: Advanced Feature Engineering]
  ├─ Capacity ratios (seats_per_workstation, seats_per_zone, utility_rooms_per_seat)
  ├─ Operator platform chronometry (total_platform_age, inactivity_months)
  ├─ Booking dynamics & occupancy rates (avail_rate_30, occupancy_365, booked_load_ratio)
  ├─ Facilities intelligence (facility_count, binary flags for top 10 amenities)
  └─ Frequency encodings for high-cardinality nominals (space_class, zone, borough)
          │
          ▼
[Step 3: Leak-Free Preprocessing Pipelines]
  ├─ Linear Pipeline: Fold-wise Median Imputation + StandardScaler + OneHotEncoder
  └─ Tree Pipeline: Fold-wise Median Imputation + Ordinal/Frequency Encoding
          │
          ▼
[Step 4: Model Exploration & Hyperparameter Tuning]
  ├─ Baseline: Dummy Classifier & Simple Logistic Regression
  ├─ Linear Models: Tuned Multinomial Logistic Regression (L-BFGS, L2 penalty, class_weight)
  ├─ Non-Tree Models: Gaussian Naive Bayes (variance smoothing exploration)
  ├─ Decision Trees: CART with cost-complexity and depth pruning (max_depth=12)
  ├─ Bagging Ensembles: Random Forest (150 trees, max_depth=18, max_features='sqrt')
  └─ Boosting Ensembles: LightGBM & HistGradientBoosting (gradient-based tree boosting)
          │
          ▼
[Step 5: Ensembling & Blending]
  ├─ Soft Probability Voting Blend (LogReg + RF + HistGB + LightGBM)
  └─ Stacking Classifier (Base models + Logistic Regression Meta-Learner)
          │
          ▼
[Step 6: Kaggle Submission Generation]
  └─ Out-of-fold calibrated test predictions -> submission.csv (114,538 rows)
```

---

## 3. The 5 Core Components for Every Model Family (Level 2 Viva Requirement)

Industry examiners at Level 2 viva evaluate candidates on the **5 fundamental pillars** of machine learning modeling:

---

### Pillar 1: Multinomial Logistic Regression (Linear Model)
1. **Data Representation:** Scaled continuous feature vector $\mathbf{x} \in \mathbb{R}^D$ ($\mu=0, \sigma=1$) concatenated with one-hot encoded categorical dummies.
2. **Hypothesis Class:** Linear log-odds mapped via Softmax function:
   $$P(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^4 \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$
3. **Loss Function:** Multiclass Cross-Entropy (Negative Log-Likelihood) with $\ell_2$ Ridge penalty:
   $$\mathcal{L}(\mathbf{W}) = -\frac{1}{N}\sum_{i=1}^N \sum_{k=1}^4 y_{ik} \ln P(y_i = k \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{k=1}^4 \|\mathbf{w}_k\|_2^2$$
4. **Optimization Algorithm:** **L-BFGS** (Limited-memory quasi-Newton optimization). Approximates the inverse Hessian matrix $\mathbf{H}^{-1}$ using past gradient evaluations, enabling superlinear convergence without storing the full $D \times D$ second derivative matrix.
5. **Evaluation & Failure Modes:** High bias when relationships are non-linear or have high-order interactions; sensitive to extreme outliers and collinear features.

---

### Pillar 2: Decision Tree (Non-Parametric Base Model)
1. **Data Representation:** Mixed raw numerical and ordinal encoded features. Invariant to monotonic transformations.
2. **Hypothesis Class:** Axis-aligned orthogonal recursive binary partitions of feature space:
   $$\hat{y}(\mathbf{x}) = \sum_{m=1}^M c_m \mathbb{I}(\mathbf{x} \in R_m), \quad c_m = \arg\max_k \sum_{i \in R_m} \mathbb{I}(y_i = k)$$
3. **Loss Function / Splitting Criterion:** Gini Impurity reduction:
   $$I_G(m) = 1 - \sum_{k=1}^4 p_{mk}^2, \quad \Delta I_G = I_G(m) - \left[\frac{N_L}{N_m} I_G(L) + \frac{N_R}{N_m} I_G(R)\right]$$
4. **Optimization Algorithm:** Greedy Top-Down Recursive Splitting (CART). At each internal node, scans all features and candidate split thresholds to maximize $\Delta I_G$.
5. **Evaluation & Failure Modes:** Low bias, extreme variance. Unpruned trees memorize noise and specific training thresholds, causing severe degradation on test data.

---

### Pillar 3: Random Forest (Bagging Ensemble)
1. **Data Representation:** High-dimensional tabular features without requiring scaling or normalization.
2. **Hypothesis Class:** Forest of $B$ independent deep decision trees:
   $$\hat{P}(y = k \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B P_b(y = k \mid \mathbf{x})$$
3. **Loss Function:** Averaged Gini Impurity across independently drawn bootstrap samples.
4. **Optimization / Training Algorithm:**
   - **Bootstrap Aggregating:** Sample $N$ instances with replacement for each tree $b$.
   - **Random Subspace Method:** Restrict candidate split features at every node to $m = \sqrt{D}$ randomly chosen features.
5. **Evaluation & Theoretical Foundation:** **Variance Reduction without Bias Inflation.** Since individual trees are fully grown, individual bias is low. Random feature subsampling forces trees to split on diverse features, decorrelating tree errors ($\rho \to 0$), driving ensemble variance $\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$ down toward zero.

---

### Pillar 4: LightGBM (Gradient Boosting Ensemble)
1. **Data Representation:** Quantized integer histogram bins (256 discrete bins) with native missing value direction routing.
2. **Hypothesis Class:** Additive expansion of shallow decision trees:
   $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta h_m(\mathbf{x})$$
3. **Loss Function:** Multiclass Cross-Entropy with $\ell_1$ and $\ell_2$ leaf weight regularization:
   $$\mathcal{L}^{(m)} \approx \sum_{i=1}^N \sum_{k=1}^4 \left[ g_{ik} f_{km}(\mathbf{x}_i) + \frac{1}{2} h_{ik} f_{km}^2(\mathbf{x}_i) \right] + \gamma J + \frac{1}{2} \lambda \sum_{j=1}^J w_j^2$$
   Where $g_{ik} = P_k(\mathbf{x}_i) - y_{ik}$ (gradient) and $h_{ik} = P_k(\mathbf{x}_i)(1 - P_k(\mathbf{x}_i))$ (hessian).
4. **Optimization Algorithm:** **Gradient-based One-Side Sampling (GOSS)** and **Exclusive Feature Bundling (EFB)**. Continuously fits new trees to the negative gradients of the previous ensemble stage with shrinkage rate $\eta$.
5. **Evaluation & Superiority:** Highest Macro F1 score on tabular data due to adaptive error correction, fine-grained leaf-wise growth (`num_leaves`), and robust regularization.

---

### Pillar 5: Stacking & Soft Voting (Meta-Ensembles)
1. **Data Representation:** Meta-feature matrix formed by concatenating out-of-fold probability vectors $\hat{\mathbf{P}}_m \in \mathbb{R}^{N \times 4}$ from base models (Logistic Regression, Random Forest, HistGB, LightGBM).
2. **Hypothesis Class:** Meta-classifier $g(\hat{\mathbf{P}}_1, \dots, \hat{\mathbf{P}}_M)$ learning optimal non-linear blending weights.
3. **Loss Function:** Cross-entropy on out-of-fold probability distributions.
4. **Optimization:** Two-stage cross-validation training. Base models are trained on $K-1$ folds and predict on the hold-out fold to populate the meta-features without target leakage.
5. **Evaluation:** Maximizes diversity. When linear models excel on smooth regional trends and tree models capture sharp threshold interactions, the ensemble outperforms any individual model.

---

## 4. Key Questions & Model Viva Defense Answers

### Q1: Why did you drop `min_term_floor` and `max_term_ceiling`?
> **Answer:** "Through correlation and pairwise difference analysis, I discovered that `min_term_days == min_term_floor` in 99.985% of records ($r=1.000$) and `max_term_days == max_term_ceiling` in 99.985% of records. Retaining both introduces exact multicollinearity, making the data matrix rank-deficient and destabilizing linear models and neural weight updates. Dropping the duplicate columns eliminates redundancy without losing any information."

### Q2: What are the extreme values in `max_term_days`, and how did you handle them?
> **Answer:** "The 99.9th percentile of `max_term_days` is 1,125 days (~3 years), yet values reach up to 577,341,108. These are artificial sentinel values used by booking systems to represent open-ended or indefinite leases. To handle them without distorting model gradients, I created a binary indicator `max_term_is_extreme = (max_term_days > 1125)`, clipped the continuous variable to 1,125 days, and applied a logarithmic transform $\ln(1+x)$."

### Q3: Why is `borough` 76.6% missing, and how did you prevent information loss?
> **Answer:** "In our metropolitan dataset, borough identifiers exist only for central administrative districts, whereas suburban and regional workspace listings lack borough codes. Because missingness correlates with price tier (Tier 1 is 85.6% missing vs Tier 4 at 71.3%), missingness is Not At Random (MNAR). I retained this signal by introducing an explicit binary indicator `is_missing_borough` and encoding the categorical feature with a dedicated `'missing'` level."

### Q4: Why did review scores have ~19.4% missing values?
> **Answer:** "By cross-tabulating missing review scores with `review_count`, I confirmed that 54,236 out of 54,240 missing rows occurred precisely when `review_count == 0`. These represent newly onboarded listings that have not yet received customer ratings. Imputing with the overall dataset median without a flag would falsely misclassify new listings as average-rated listings. I added `is_missing_reviews` and `has_reviews` indicators."

### Q5: Why is Macro F1 sensitive to class imbalance, and how did your tuning address it?
> **Answer:** "Macro F1 is the unweighted arithmetic mean of F1 scores across all four classes. In our training set, Tier 3 has 60,839 records compared to Tier 2 with 81,530. A model optimized purely for overall Accuracy would overpredict Tier 2 at the expense of Tier 3. To prevent this, we used Stratified K-Fold CV, tuned probability thresholds, and evaluated class-specific F1 scores to ensure balanced performance across all four tiers."

### Q6: How does Random Forest achieve variance reduction compared to a single Decision Tree?
> **Answer:** "A single decision tree grown to full depth has low bias but high variance because small perturbations in the training data produce completely different splits. Random Forest averages $B$ bootstrap trees. By randomly sampling $m = \sqrt{D}$ features at each split, the pairwise correlation $\rho$ between trees is minimized. Since $\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$, reducing $\rho$ drives ensemble variance to a fraction of the single tree variance while preserving low bias."
