# Milestone 4: Modeling Architecture, Hypotheses & Tuning Strategy
**IITM BS Degree — Machine Learning Practice (MLP) Project**  
**Student Roll Number:** 24F2006569 | **Term:** 2026 T3  

---

## 1. Executive Strategy & Problem Formulation

The goal of Milestone 4 is to systematically build, evaluate, interpret, and tune multiple machine learning model families for the 4-class workstation price tier classification problem. 

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N, \quad \mathbf{x}_i \in \mathbb{R}^D, \quad y_i \in \{1, 2, 3, 4\}$$

Because the Kaggle evaluation metric is **Macro F1 Score**, our optimization objective is to maximize the unweighted arithmetic mean of $F_1$ across all four classes:

$$\max_{\boldsymbol{\theta}} \text{Macro } F_1 = \frac{1}{4} \sum_{k=1}^4 \frac{2 \cdot P_k(\boldsymbol{\theta}) \cdot R_k(\boldsymbol{\theta})}{P_k(\boldsymbol{\theta}) + R_k(\boldsymbol{\theta})}$$

---

## 2. Model Families & Theoretical Formulations

We investigate six distinct hypothesis spaces representing linear, parametric non-tree, non-parametric tree, bagging, gradient boosting, and meta-learning architectures:

### 1. Linear Models: Multinomial Logistic Regression
- **Mathematical Form:**
  $$P(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^4 \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$
- **Loss Function:** Multiclass Cross-Entropy with $\ell_2$ Ridge Regularization:
  $$\mathcal{L}(\mathbf{W}) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^4 \mathbb{I}(y_i = k) \ln P(y_i = k \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{k=1}^4 \|\mathbf{w}_k\|_2^2$$
- **Assumptions:** Linear decision boundaries in log-odds space; linear additivity of feature effects; feature standardization is critical so regularization penalizes weights uniformly.
- **Optimization:** L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) quasi-Newton algorithm.

### 2. Non-Tree Parametric Models: Gaussian Naive Bayes
- **Mathematical Form:**
  $$P(y = k \mid \mathbf{x}) \propto P(y = k) \prod_{j=1}^D P(x_j \mid y = k), \quad x_j \mid y = k \sim \mathcal{N}(\mu_{jk}, \sigma_{jk}^2 + \epsilon)$$
- **Assumptions:** Conditional independence of all features given class label: $P(\mathbf{x} \mid y) = \prod_j P(x_j \mid y)$. Features follow univariate Gaussian distributions within each tier.
- **Strengths & Limitations:** Extremely fast ($O(ND)$ time complexity); however, the conditional independence assumption is heavily violated in spatial and capacity features (e.g., `max_seats` and `workstations`), causing overconfident, distorted posterior probabilities.

### 3. Decision Trees: CART (Classification and Regression Trees)
- **Mathematical Form:** Recursive binary partition of feature space into axis-aligned hyper-rectangles:
  $$\hat{y}(\mathbf{x}) = \arg\max_k \sum_{i \in R_m} \mathbb{I}(y_i = k)$$
- **Splitting Criterion:** Gini Impurity reduction:
  $$I_G(m) = 1 - \sum_{k=1}^4 p_{mk}^2, \quad \Delta I_G = I_G(m) - \left(\frac{N_L}{N_m} I_G(L) + \frac{N_R}{N_m} I_G(R)\right)$$
- **Strengths & Limitations:** Invariant to monotonic scaling; captures non-linear interactions automatically. Highly prone to severe overfitting (high variance) without depth regularization or cost-complexity pruning.

### 4. Bagging Ensembles: Random Forest Classifier
- **Working Mechanism:** Bootstrap Aggregating (Bagging) + Random Subspace Method:
  1. Draw $B$ bootstrap samples $\mathcal{D}_b$ of size $N$ with replacement from $\mathcal{D}$.
  2. Grow deep, unpruned decision trees on each $\mathcal{D}_b$, considering only a random subset of $m = \sqrt{D}$ features at each split.
  3. Aggregate predictions via majority vote or averaged class probabilities:
     $$\hat{P}(y = k \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B P_b(y = k \mid \mathbf{x})$$
- **Variance Reduction Theorem:** For $B$ estimators with variance $\sigma^2$ and pairwise correlation $\rho$:
  $$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
  Feature subsampling drastically reduces $\rho$, shrinking overall ensemble variance while preserving low bias.

### 5. Boosting Ensembles: LightGBM & HistGradientBoosting
- **Working Mechanism:** Sequential additive modeling via gradient descent in function space:
  $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \sum_{j=1}^J \gamma_{jm} \mathbb{I}(\mathbf{x} \in R_{jm})$$
  Where negative gradients (pseudo-residuals) guide tree fitting:
  $$r_{ikm} = -\left[\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F_k(\mathbf{x}_i)}\right]_{F=F_{m-1}} = \mathbb{I}(y_i = k) - P(y_i = k \mid \mathbf{x}_i)$$
- **Histogram Binning & GOSS:** Continuous features are discretized into 256 integer bins, reducing split-finding from $O(N \log N)$ to $O(K)$, enabling rapid exploration on large datasets ($280\text{K}$ rows) while handling missing values natively.

### 6. Meta-Learning: Stacking & Soft Voting
- **Soft Voting (Probability Blending):** Combines calibrated posterior probability distributions from diverse model families:
  $$P_{\text{blend}}(y = k \mid \mathbf{x}) = \sum_{m=1}^M w_m P_m(y = k \mid \mathbf{x}), \quad \sum w_m = 1$$
- **Stacking Classifier:** Trains base estimators using out-of-fold cross-validation, generating leak-free meta-features $\mathbf{Z} \in \mathbb{R}^{N \times (M \times K)}$. A meta-estimator (multinomial logistic regression) learns optimal non-linear blending weights.

---

## 3. Validation Design & Hyperparameter Tuning Strategy

### A. Stratified 5-Fold Cross-Validation
- Stratification maintains exact class proportions across all five folds ($23.9\% / 29.1\% / 21.8\% / 25.2\%$).
- Preprocessing steps (scaling parameters $\mu, \sigma$, imputation medians, frequency maps) are fitted strictly on the 4 training folds and applied to the 5th validation fold, eliminating data leakage.

### B. Hyperparameter Search Principles
- **Logistic Regression:** Regularization parameter $C \in [10^{-3}, 10^1]$ and class weight options.
- **Decision Trees:** Max depth $d \in [4, 25]$ and minimum leaf size to arrest overfitting.
- **Random Forest:** Tree count $B \in [50, 200]$, tree depth, and minimum leaf samples.
- **LightGBM:** Leaf count `num_leaves` $\in [31, 127]$, learning rate $\eta \in [0.03, 0.15]$, feature subsampling `colsample_bytree`, and row subsampling `subsample`.
