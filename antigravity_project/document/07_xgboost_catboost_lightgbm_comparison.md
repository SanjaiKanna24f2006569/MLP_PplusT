# The GBDT Holy Trinity: In-Depth Comparison of LightGBM, XGBoost & CatBoost
**IITM BS Degree — Machine Learning Practice (MLP) Project**  
**Student Roll Number:** 24F2006569 | **Term:** 2026 T3  

---

## 1. Executive Summary

In tabular competitive machine learning, **Gradient Boosted Decision Trees (GBDTs)** consistently dominate deep learning models on structured datasets. The three state-of-the-art implementations are:
1. **LightGBM (Microsoft Research, 2017)**
2. **XGBoost (Tianqi Chen & Carlos Guestrin, 2016)**
3. **CatBoost (Yandex Research, 2018)**

While all three minimize loss through gradient descent in function space, their tree construction algorithms, categorical handling, split search strategies, and regularization mechanics differ fundamentally. The highest Kaggle leaderboard scores are almost universally achieved by **ensembling predictions from all three libraries**.

---

## 2. In-Depth Algorithmic & Architectural Comparison

| Dimension | LightGBM | XGBoost | CatBoost |
| :--- | :--- | :--- | :--- |
| **Tree Growth Strategy** | **Leaf-wise (Best-First)**<br>Splits the leaf with maximum loss reduction regardless of level. Achieves lower loss with fewer leaves. | **Depth-wise (Level-wise)**<br>Splits all leaves at the same depth level up to `max_depth`. Prunes backwards using $\gamma$. | **Symmetric (Oblivious Trees)**<br>Applies the exact same feature & threshold split across all nodes at a given depth. |
| **Execution Speed & Memory** | Fastest training speed and lowest memory consumption via 256 histogram bins. | Fast (using `tree_method='hist'`), moderately higher memory for large graphs. | Moderate training speed; **fastest inference speed** due to bitwise tree evaluation. |
| **Categorical Handling** | Sorts categorical histogram bins by gradient statistics ($O(K \log K)$). | Historically required One-Hot or Target Encoding; recently added experimental partition splits. | **State-of-the-Art (Ordered Target Encoding)** with online permutations, eliminating target leakage. |
| **Missing Value Routing** | Automatically assigns missing values to whichever child branch maximizes gain. | Sparsity-aware split: learns default branch direction during split finding. | Imputes with minimum, maximum, or intermediate values; handles missing as separate category. |
| **Loss Approximation** | First-order gradients + diagonal hessians. | **Exact second-order Taylor expansion** ($g_i$ gradient and $h_i$ hessian). | Second-order optimization with **Ordered Boosting** to combat prediction shift. |
| **Key Regularizers** | `reg_alpha` ($\ell_1$), `reg_lambda` ($\ell_2$), `min_child_samples`, `num_leaves`. | `reg_alpha` ($\ell_1$), `reg_lambda` ($\ell_2$), `gamma` (min gain for split), `min_child_weight`. | `l2_leaf_reg`, `random_strength`, `depth`. |

---

## 3. Mathematical Foundations

All three models construct an additive ensemble of $M$ decision trees:

$$F_M(\mathbf{x}) = \sum_{m=1}^M \eta f_m(\mathbf{x})$$

Where $\eta \in (0, 1]$ is the shrinkage factor (learning rate).

### Second-Order Taylor Expansion of the Loss Function:
At step $m$, expanding the loss around the current prediction $\hat{y}_i^{(m-1)}$:

$$\mathcal{L}^{(m)} \approx \sum_{i=1}^N \left[ \ell(y_i, \hat{y}_i^{(m-1)}) + g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i) \right] + \Omega(f_m)$$

Where:
- First-order gradient: $g_i = \frac{\partial \ell(y_i, \hat{y})}{\partial \hat{y}}\Big|_{\hat{y} = \hat{y}_i^{(m-1)}}$
- Second-order Hessian: $h_i = \frac{\partial^2 \ell(y_i, \hat{y})}{\partial \hat{y}^2}\Big|_{\hat{y} = \hat{y}_i^{(m-1)}}$

### Optimal Leaf Weight Solution:
For a tree with $J$ leaves where instance $i$ falls into leaf $j$ ($i \in I_j$):

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

The maximum reduction in loss (split gain) for splitting node $I$ into $I_L$ and $I_R$ is:

$$\text{Gain} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

---

## 4. Why Ensembling All Three Beats Any Single Model

### The Error Diversity Theorem
When blending probability predictions from multiple models:

$$\hat{P}_{\text{blend}}(\mathbf{x}) = \sum_{k=1}^K w_k \hat{P}_k(\mathbf{x})$$

The ensemble variance is governed by:

$$\text{Var}(\hat{P}_{\text{blend}}) = \sum_{k=1}^K w_k^2 \text{Var}(\hat{P}_k) + \sum_{j \ne k} w_j w_k \text{Cov}(\hat{P}_j, \hat{P}_k)$$

Because:
1. **LightGBM** builds asymmetric, deep leaf-wise trees that capture fine-grained high-capacity boundaries.
2. **XGBoost** builds balanced, depth-wise trees with conservative pruning ($\gamma$) and exact hessian curvature.
3. **CatBoost** builds symmetric, oblivious trees with uniform feature splits across all subnodes, heavily suppressing variance on small slices.

Their covariance $\text{Cov}(\hat{P}_j, \hat{P}_k)$ is significantly lower than between models within the same family! When LightGBM misclassifies an edge case due to an aggressive leaf split, CatBoost's conservative oblivious tree and XGBoost's regularized branch pull the blended probability back toward the true class.

---

## 5. Hyperparameter Tuning Cheat Sheet for Competition Success

### 1. LightGBM Key Knobs:
- `num_leaves`: Range $[31, 127]$. Rule of thumb: $\text{num\_leaves} < 2^{\text{max\_depth}}$.
- `min_child_samples`: Range $[20, 100]$. Prevents overfitting on rare leaf combinations.
- `learning_rate`: $[0.03, 0.10]$. Pair with $n_{\text{estimators}} \in [300, 1000]$.
- `colsample_bytree`: $[0.70, 0.85]$. Feature subsampling per tree.
- `subsample`: $[0.70, 0.85]$. Row subsampling (bagging).

### 2. XGBoost Key Knobs:
- `max_depth`: Range $[5, 9]$. (Depth-wise trees; depth 6–7 is standard).
- `min_child_weight`: Range $[1, 10]$. Minimum sum of instance weight (Hessian) in a child.
- `gamma`: Range $[0, 5]$. Minimum loss reduction required to make a further partition.
- `reg_alpha` ($\ell_1$) & `reg_lambda` ($\ell_2$): L1/L2 penalties on leaf weights.
- `tree_method='hist'`: Mandatory for large datasets ($> 100\text{K}$ rows) on CPU.

### 3. CatBoost Key Knobs:
- `depth`: Range $[4, 8]$. (Oblivious trees; depth 6 is optimal on tabular data).
- `l2_leaf_reg`: Range $[1, 10]$. L2 regularization of the cost function.
- `random_strength`: Adds random score to splits to prevent greediness.
- `bagging_temperature`: Bayesian bootstrap temperature (controls sample weight variance).
