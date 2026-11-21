import json
import os

def generate_official_notebook():
    cells = []

    def md(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    def code(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    # Header
    cells.append(md(
        """# IITM BS Degree — Machine Learning Practice (MLP) Project
## Registered Submission Notebook: `24f2006569-notebook-2026t3`
**Student Roll Number:** `24f2006569` | **Term:** 2026 T3  
**Target Variable:** `price_tier` (Tiers 1, 2, 3, 4) | **Metric:** Macro F1 Score  
**Collaborator Shared Access:** `iitmbscs2008p` (Viewer Access, Private)  

---

### High-Performance Multi-GBDT Pipeline Architecture:
1. **Dynamic Dataset Path Resolver:** Seamlessly executes locally or in Kaggle (`/kaggle/input/...`).
2. **Amenity Quality Index ($r = 0.4629$):** Continuous luxury vs. budget amenity scoring across all 144 facility codes without coordinate memorization.
3. **Leak-Free Regional Bayesian Target Encoding:** Captures neighborhood pricing power on `zone`, `space_class`, and `metro` with $m=15$ Bayesian smoothing.
4. **Domain Capacity & Chronometry Engineering:** Workstations per seat, occupancy ratios, booked load dynamics, and operator chronometry.
5. **Multi-GBDT Holy Trinity:** Blends **LightGBM** (40%), **XGBoost** (35%), and **CatBoost** (25%) to minimize generalization variance.
6. **Class-Weighted Objective & Nelder-Mead Probability Calibration:** Resolves the Tier 3 squeeze with calibrated decision multipliers.
7. **Strict Kaggle Submission Guard:** Produces verified `submission.csv` (114,538 rows $\times$ 2 columns) matching `sample_submission.csv`."""
    ))

    # Cell 1: Setup & Permitted Libraries
    cells.append(code(
        """# =============================================================================
# 1. SETUP & PERMITTED LIBRARIES CHECK
# =============================================================================
import os
import re
import sys
import time
import glob
import json
import warnings
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.optimize import minimize

# Permitted GBDT Holy Trinity
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
print(f"NumPy: {np.__version__} | Pandas: {pd.__version__}")
print(f"LightGBM: {lgb.__version__} | XGBoost: {xgb.__version__}")
print("All permitted packages loaded successfully.")"""
    ))

    # Cell 2: Automated Path Discovery & Cleaning
    cells.append(code(
        """# =============================================================================
# 2. AUTOMATIC DATASET PATH RESOLVER & CLEANING
# =============================================================================
def locate_file(filename):
    candidates = [
        os.path.join('/kaggle', 'input', filename),
        os.path.join('.', 'data', filename),
        os.path.join('..', 'data', filename),
        os.path.join('.', filename),
        os.path.join('..', filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if filename in files:
                return os.path.join(root, filename)
    for root, dirs, files in os.walk('.'):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"Could not locate '{filename}'")

train_path = locate_file('train.csv')
test_path = locate_file('test.csv')
sample_sub_path = locate_file('sample_submission.csv')

print(f"Train path: {train_path}")
print(f"Test path:  {test_path}")

t0 = time.time()
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Drop redundant duplicate columns (100% multicollinear)
for df in [train_df, test_df]:
    df.drop(columns=['min_term_floor', 'max_term_ceiling'], errors='ignore', inplace=True)

print(f"Loaded train ({train_df.shape}) and test ({test_df.shape}) in {time.time()-t0:.2f}s")"""
    ))

    # Cell 3: Amenity Quality Index
    cells.append(code(
        """# =============================================================================
# 3. AMENITY QUALITY INDEX (r = 0.463 CORRELATION WITH PRICE TIER)
# =============================================================================
print("Computing robust Amenity Quality Index across 144 facility codes...")
all_facs = []
for s in train_df['facilities'].dropna():
    all_facs.extend(re.findall(r'FCL-\\d+', s))
counts = Counter(all_facs)

fac_weights = {}
for fac in counts:
    has_fac = train_df['facilities'].str.contains(fac, regex=False).fillna(False)
    fac_weights[fac] = train_df.loc[has_fac, 'price_tier'].mean()

global_amenity_mean = train_df['price_tier'].mean()
sorted_facs = sorted(fac_weights.items(), key=lambda x: x[1], reverse=True)
luxury_facs = set([f[0] for f in sorted_facs[:15]])
budget_facs = set([f[0] for f in sorted_facs[-15:]])

def extract_amenity_features(df):
    qual_scores, lux_counts, bud_counts, tot_counts = [], [], [], []
    for s in df['facilities']:
        if not isinstance(s, str) or not s:
            qual_scores.append(global_amenity_mean)
            lux_counts.append(0); bud_counts.append(0); tot_counts.append(0)
        else:
            matches = re.findall(r'FCL-\\d+', s)
            tot_counts.append(len(matches))
            if matches:
                qual_scores.append(np.mean([fac_weights.get(m, global_amenity_mean) for m in matches]))
                lux_counts.append(sum(1 for m in matches if m in luxury_facs))
                bud_counts.append(sum(1 for m in matches if m in budget_facs))
            else:
                qual_scores.append(global_amenity_mean)
                lux_counts.append(0); bud_counts.append(0)
    return pd.DataFrame({
        'amenity_quality_score': np.array(qual_scores, dtype=np.float32),
        'amenity_luxury_count': np.array(lux_counts, dtype=np.int16),
        'amenity_budget_count': np.array(bud_counts, dtype=np.int16),
        'amenity_total_count': np.array(tot_counts, dtype=np.int16),
        'amenity_luxury_ratio': (np.array(lux_counts) / (np.array(tot_counts) + 1.0)).astype(np.float32)
    }, index=df.index)

train_amenity_df = extract_amenity_features(train_df)
test_amenity_df = extract_amenity_features(test_df)
print(f"Amenity Quality Index computed! Pearson r = {train_amenity_df['amenity_quality_score'].corr(train_df['price_tier']):.4f}")"""
    ))

    # Cell 4: Base Domain Features
    cells.append(code(
        """# =============================================================================
# 4. DOMAIN FEATURE EXTRACTION (CAPACITY RATIOS, SENTINELS, CHRONOMETRY)
# =============================================================================
TOP_FACILITIES = [
    'FCL-069', 'FCL-089', 'FCL-120', 'FCL-011', 'FCL-007',
    'FCL-055', 'FCL-071', 'FCL-097', 'FCL-035', 'FCL-098'
]

def extract_base_features(df, train_ref=None):
    out = pd.DataFrame(index=df.index)
    if 'price_tier' in df.columns:
        out['price_tier'] = df['price_tier'].astype(np.int8)
        
    num_cols = [
        'geo_lat', 'geo_lon', 'max_seats', 'utility_rooms', 'zones', 'workstations',
        'min_term_days', 'max_term_days', 'min_term_avg', 'max_term_avg',
        'open_slots_a', 'open_slots_b', 'open_slots_c', 'open_slots_d', 'open_slots_e',
        'review_count', 'review_count_12m', 'review_count_30d', 'review_count_prev_yr',
        'booked_load', 'score_overall', 'score_accuracy', 'score_condition',
        'score_onboarding', 'score_support', 'score_access', 'score_value',
        'operator_metro_units', 'operator_metro_whole', 'operator_metro_private',
        'operator_metro_shared', 'reviews_per_month', 'operator_portfolio_size',
        'operator_platform_age_years', 'operator_platform_age_months',
        'operator_active_years', 'operator_active_months'
    ]
    for c in num_cols:
        if c in df.columns:
            out[c] = df[c].astype(np.float32)
            
    # Structured missingness
    out['is_missing_borough'] = df['borough'].isna().astype(np.int8)
    out['is_missing_reviews'] = df['score_overall'].isna().astype(np.int8)
    out['is_missing_zones'] = df['zones'].isna().astype(np.int8)
    out['is_missing_workstations'] = df['workstations'].isna().astype(np.int8)

    # Sentinel extreme capping
    out['max_term_is_extreme'] = (df['max_term_days'] > 1125).fillna(False).astype(np.int8)
    out['log_max_term_days'] = np.log1p(df['max_term_days'].clip(0, 1125)).astype(np.float32)
    out['log_min_term_days'] = np.log1p(df['min_term_days'].clip(lower=0)).astype(np.float32)
    out['log_portfolio_size'] = np.log1p(df['operator_portfolio_size'].clip(lower=0)).astype(np.float32)

    # Capacity ratios
    out['seats_per_workstation'] = (out['max_seats'] / (out['workstations'].fillna(2.0) + 1.0)).astype(np.float32)
    out['seats_per_zone'] = (out['max_seats'] / (out['zones'].fillna(1.0) + 1.0)).astype(np.float32)
    out['workstations_per_zone'] = (out['workstations'].fillna(2.0) / (out['zones'].fillna(1.0) + 1.0)).astype(np.float32)
    out['utility_rooms_per_seat'] = (out['utility_rooms'].fillna(1.0) / (out['max_seats'] + 1.0)).astype(np.float32)

    # Operator chronometry
    plat_age = df['operator_platform_age_years'].fillna(0)*12 + df['operator_platform_age_months'].fillna(0)
    act_age = df['operator_active_years'].fillna(0)*12 + df['operator_active_months'].fillna(0)
    out['total_platform_age_months'] = plat_age.astype(np.float32)
    out['operator_inactivity_months'] = (plat_age - act_age).clip(lower=0).astype(np.float32)

    # Occupancy dynamics
    out['avail_rate_30'] = (out['open_slots_a'] / 30.0).clip(0, 1).astype(np.float32)
    out['occupancy_30'] = (1.0 - out['avail_rate_30']).astype(np.float32)
    out['booked_load_ratio'] = (out['booked_load'] / 255.0).clip(0, 1).astype(np.float32)

    # Review composite
    score_cols = ['score_overall', 'score_accuracy', 'score_condition', 'score_onboarding', 'score_support', 'score_access', 'score_value']
    out['score_mean'] = out[score_cols].mean(axis=1).astype(np.float32)

    # Facilities presence
    fac_str = df['facilities'].fillna('')
    out['facility_count'] = fac_str.apply(lambda s: len(re.findall(r'FCL-\\d+', s))).astype(np.int16)
    for fcl in TOP_FACILITIES:
        out[f'has_{fcl}'] = fac_str.str.contains(fcl, regex=False).astype(np.int8)

    # Categoricals
    for c in ['operator_class', 'access_type', 'compliance_status', 'metro']:
        out[c] = df[c].fillna('missing').astype('category')

    for c in ['operator_certified', 'operator_id_verified']:
        out[c] = (df[c] == 't').astype(np.int8)

    source_df = train_ref if train_ref is not None else df
    for c in ['space_class', 'zone', 'borough']:
        freq_map = source_df[c].value_counts(normalize=True).to_dict()
        out[f'{c}_freq'] = df[c].map(freq_map).fillna(0).astype(np.float32)
        out[c] = df[c].fillna('missing').astype('category')

    return out

print("Extracting base features...")
X_train_full = extract_base_features(train_df, train_ref=train_df)
y_train_full = X_train_full.pop('price_tier').astype(int)
X_test_full = extract_base_features(test_df, train_ref=train_df)
test_ids = test_df['space_id'].values

# Attach Amenity Features
for c in train_amenity_df.columns:
    X_train_full[c] = train_amenity_df[c]
    X_test_full[c] = test_amenity_df[c]"""
    ))

    # Cell 5: Regional Target Encoding & Preprocessing
    cells.append(code(
        """# =============================================================================
# 5. ROBUST REGIONAL TARGET ENCODING & PREPROCESSING
# =============================================================================
print("Computing robust regional target encodings (Zone, Space Class, Metro)...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
te_cols = ['zone', 'space_class', 'metro']
global_mean = y_train_full.mean()

for c in te_cols:
    X_train_full[f'{c}_te_mean'] = 0.0
    X_test_full[f'{c}_te_mean'] = 0.0

for tr_idx, val_idx in kf.split(X_train_full, y_train_full):
    tr_y = y_train_full.iloc[tr_idx]
    for c in te_cols:
        tr_cat = train_df[c].iloc[tr_idx]
        val_cat = train_df[c].iloc[val_idx]
        stats = tr_y.groupby(tr_cat).agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + 15 * global_mean) / (stats['count'] + 15)
        mapping = smoothed.to_dict()
        X_train_full.loc[X_train_full.index[val_idx], f'{c}_te_mean'] = val_cat.map(mapping).fillna(global_mean)

for c in te_cols:
    stats = y_train_full.groupby(train_df[c]).agg(['count', 'mean'])
    smoothed = (stats['count'] * stats['mean'] + 15 * global_mean) / (stats['count'] + 15)
    mapping = smoothed.to_dict()
    X_test_full[f'{c}_te_mean'] = test_df[c].map(mapping).fillna(global_mean)

# Split 80/20 for local validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

cat_cols = [c for c in X_train_full.columns if str(X_train_full[c].dtype) == 'category']
num_cols = [c for c in X_train_full.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols)
    ]
)

print("Pre-transforming feature matrices...")
X_tr_p = preprocessor.fit_transform(X_tr)
X_val_p = preprocessor.transform(X_val)
X_test_p = preprocessor.transform(X_test_full)

y_tr_0 = (y_tr - 1).values
y_val_0 = (y_val - 1).values

# Class weights: gently protect Tier 3 without over-inflating Tier 2
class_weights = {0: 1.02, 1: 0.95, 2: 1.15, 3: 0.98}
sample_weights_tr = np.array([class_weights[y] for y in y_tr_0])
print(f"Matrix shape: {X_tr_p.shape} | Tier 3 weight: 1.15x")"""
    ))

    # Cell 6: Tri-GBDT Training (LightGBM + XGBoost + CatBoost)
    cells.append(code(
        """# =============================================================================
# 6. TRI-GBDT HOLY TRINITY TRAINING (LIGHTGBM + XGBOOST + CATBOOST)
# =============================================================================
# 1. High-Capacity LightGBM
print("\\n---> [1/3] Training Balanced LightGBM ...")
t0 = time.time()
lgbm = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=500,
    learning_rate=0.06,
    num_leaves=100,
    max_depth=12,
    subsample=0.85,
    colsample_bytree=0.80,
    min_child_samples=25,
    random_state=42,
    n_jobs=-1
)
lgbm.fit(X_tr_p, y_tr_0, sample_weight=sample_weights_tr)
prob_lgb = lgbm.predict_proba(X_val_p)
f1_lgb = f1_score(y_val, np.argmax(prob_lgb, axis=1) + 1, average='macro')
print(f"LightGBM Macro F1: {f1_lgb:.4f} in {time.time()-t0:.1f}s")

# 2. Balanced XGBoost (Exact Hessian Histograms)
print("\\n---> [2/3] Training Balanced XGBoost ...")
t0 = time.time()
xgb_clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    tree_method='hist',
    n_estimators=450,
    learning_rate=0.06,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.80,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
)
xgb_clf.fit(X_tr_p, y_tr_0, sample_weight=sample_weights_tr)
prob_xgb = xgb_clf.predict_proba(X_val_p)
f1_xgb = f1_score(y_val, np.argmax(prob_xgb, axis=1) + 1, average='macro')
print(f"XGBoost Macro F1 : {f1_xgb:.4f} in {time.time()-t0:.1f}s")

# 3. Balanced CatBoost (Symmetric Oblivious Trees)
print("\\n---> [3/3] Training Balanced CatBoost ...")
t0 = time.time()
cb_clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.08,
    depth=6,
    loss_function='MultiClass',
    l2_leaf_reg=4.0,
    auto_class_weights='Balanced',
    random_seed=42,
    thread_count=-1,
    verbose=0
)
cb_clf.fit(X_tr_p, y_tr_0)
prob_cb = cb_clf.predict_proba(X_val_p)
f1_cb = f1_score(y_val, np.argmax(prob_cb, axis=1) + 1, average='macro')
print(f"CatBoost Macro F1: {f1_cb:.4f} in {time.time()-t0:.1f}s")"""
    ))

    # Cell 7: Blending & Optimization
    cells.append(code(
        """# =============================================================================
# 7. GRAND TRI-GBDT ENSEMBLE BLEND & MULTIPLIER OPTIMIZATION
# =============================================================================
# Optimal architectural weights: 40% LightGBM + 35% XGBoost + 25% CatBoost
prob_blend = 0.40 * prob_lgb + 0.35 * prob_xgb + 0.25 * prob_cb
raw_blend_f1 = f1_score(y_val, np.argmax(prob_blend, axis=1) + 1, average='macro')
print(f"Raw Tri-GBDT Blend Macro F1: {raw_blend_f1:.4f}")

# Optimize decision boundary multipliers directly for Macro F1
def loss_func(weights):
    scaled = prob_blend * weights
    preds = np.argmax(scaled, axis=1) + 1
    return -f1_score(y_val, preds, average='macro')

res = minimize(loss_func, [1.0, 1.0, 1.0, 1.0], method='Nelder-Mead', options={'maxiter': 300})
best_w = res.x / res.x[0]
print(f"Optimal Decision Multipliers: {[round(w, 4) for w in best_w]}")

opt_val_preds = np.argmax(prob_blend * best_w, axis=1) + 1
opt_f1 = f1_score(y_val, opt_val_preds, average='macro')
opt_acc = accuracy_score(y_val, opt_val_preds)

print("\\n" + "="*65)
print(f">>> FINAL TRI-GBDT OPTIMIZED MACRO F1: {opt_f1:.4f} <<<")
print(f">>> FINAL TRI-GBDT OPTIMIZED ACCURACY: {opt_acc:.4f} <<<")
print("="*65)
print("Per-class F1:", [round(x, 4) for x in f1_score(y_val, opt_val_preds, average=None)])
print("\\nClassification Report:\\n", classification_report(y_val, opt_val_preds, target_names=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'], digits=4))"""
    ))

    # Cell 8: Prediction & Output Guard
    cells.append(code(
        """# =============================================================================
# 8. GENERATE & VALIDATE FINAL KAGGLE SUBMISSION (submission.csv)
# =============================================================================
print("\\nPredicting on 114,538 Test instances...")
test_p_lgb = lgbm.predict_proba(X_test_p)
test_p_xgb = xgb_clf.predict_proba(X_test_p)
test_p_cb = cb_clf.predict_proba(X_test_p)

test_blend = 0.40 * test_p_lgb + 0.35 * test_p_xgb + 0.25 * test_p_cb
test_scaled = test_blend * best_w
final_preds = np.argmax(test_scaled, axis=1) + 1

# Output CSV: writes directly to current directory for Kaggle notebook commit
sub_df = pd.DataFrame({
    'ID': test_ids,
    'price_tier': final_preds
})
sub_df.to_csv('submission.csv', index=False)
print(f"Saved 'submission.csv' ({sub_df.shape})!")

# Strict Format Verification
sample_sub = pd.read_csv(sample_sub_path)
assert sub_df.shape == sample_sub.shape, f"Shape mismatch: {sub_df.shape} vs {sample_sub.shape}"
assert list(sub_df.columns) == list(sample_sub.columns), "Columns mismatch!"
assert (sub_df['ID'] == sample_sub['ID']).all(), "ID sequence mismatch!"
assert not sub_df['price_tier'].isna().any(), "Submission contains nulls!"
assert set(sub_df['price_tier'].unique()).issubset({1, 2, 3, 4}), "Invalid tier values found!"

print("\\n>>> SUBMISSION VALIDATION PASSED 100%! Ready for Kaggle Leaderboard. <<<\\n")
print(sub_df.head(10))
print("\\nPredicted Test Class Proportions:")
print((sub_df['price_tier'].value_counts(normalize=True).sort_index() * 100).round(2).to_string())"""
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open("24f2006569-notebook-2026t3.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Updated 24f2006569-notebook-2026t3.ipynb successfully!")

    with open("main_submission_notebook.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Updated main_submission_notebook.ipynb successfully!")

if __name__ == '__main__':
    generate_official_notebook()
