import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report, log_loss

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import extract_features

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("=== STARTING ADVANCED GBDT EXPERIMENTATION: XGBOOST & CATBOOST ===")

# 1. Load Data
t0 = time.time()
print("Loading train.csv and test.csv...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# 2. Extract Features
X_train_full = extract_features(train_df, train_ref=train_df, is_train=True)
y_train_full = X_train_full.pop('price_tier')

X_test_full = extract_features(test_df, train_ref=train_df, is_train=False)
test_ids = test_df['space_id'].values

cat_cols = [c for c in X_train_full.columns if str(X_train_full[c].dtype) == 'category']
num_cols = [c for c in X_train_full.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols)
    ]
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

print("Transforming feature matrices...")
X_tr_proc = preprocessor.fit_transform(X_tr)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test_full)

# Convert labels to 0-3 for GBDTs
y_tr_0 = (y_tr - 1).values
y_val_0 = (y_val - 1).values

models_res = {}

# -----------------------------------------------------------------------------
# 1. LightGBM Tuned
# -----------------------------------------------------------------------------
print("\n---> [1/3] Training LightGBM Classifier ...")
t_start = time.time()
lgbm = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=350,
    learning_rate=0.08,
    num_leaves=63,
    max_depth=10,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=30,
    random_state=42,
    n_jobs=-1
)
lgbm.fit(X_tr_proc, y_tr_0)
t_lgb = time.time() - t_start
prob_lgb = lgbm.predict_proba(X_val_proc)
pred_lgb = np.argmax(prob_lgb, axis=1) + 1
f1_lgb = f1_score(y_val, pred_lgb, average='macro')
acc_lgb = accuracy_score(y_val, pred_lgb)
loss_lgb = log_loss(y_val_0, prob_lgb)
print(f"LightGBM: Fit Time={t_lgb:.1f}s | Macro F1={f1_lgb:.4f} | Accuracy={acc_lgb:.4f} | Log Loss={loss_lgb:.4f}")
models_res['LightGBM'] = {'prob': prob_lgb, 'f1': f1_lgb, 'acc': acc_lgb, 'loss': loss_lgb, 'time': t_lgb, 'model': lgbm}

# -----------------------------------------------------------------------------
# 2. XGBoost Tuned (Histogram Method)
# -----------------------------------------------------------------------------
print("\n---> [2/3] Training XGBoost Classifier (tree_method='hist') ...")
t_start = time.time()
xgboost_clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    tree_method='hist',
    n_estimators=350,
    learning_rate=0.08,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
)
xgboost_clf.fit(X_tr_proc, y_tr_0)
t_xgb = time.time() - t_start
prob_xgb = xgboost_clf.predict_proba(X_val_proc)
pred_xgb = np.argmax(prob_xgb, axis=1) + 1
f1_xgb = f1_score(y_val, pred_xgb, average='macro')
acc_xgb = accuracy_score(y_val, pred_xgb)
loss_xgb = log_loss(y_val_0, prob_xgb)
print(f"XGBoost: Fit Time={t_xgb:.1f}s | Macro F1={f1_xgb:.4f} | Accuracy={acc_xgb:.4f} | Log Loss={loss_xgb:.4f}")
models_res['XGBoost'] = {'prob': prob_xgb, 'f1': f1_xgb, 'acc': acc_xgb, 'loss': loss_xgb, 'time': t_xgb, 'model': xgboost_clf}

# -----------------------------------------------------------------------------
# 3. CatBoost Tuned
# -----------------------------------------------------------------------------
print("\n---> [3/3] Training CatBoost Classifier ...")
t_start = time.time()
catboost_clf = CatBoostClassifier(
    iterations=400,
    learning_rate=0.09,
    depth=6,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    l2_leaf_reg=3.0,
    random_seed=42,
    thread_count=-1,
    verbose=100
)
catboost_clf.fit(X_tr_proc, y_tr_0)
t_cat = time.time() - t_start
prob_cat = catboost_clf.predict_proba(X_val_proc)
pred_cat = np.argmax(prob_cat, axis=1) + 1
f1_cat = f1_score(y_val, pred_cat, average='macro')
acc_cat = accuracy_score(y_val, pred_cat)
loss_cat = log_loss(y_val_0, prob_cat)
print(f"CatBoost: Fit Time={t_cat:.1f}s | Macro F1={f1_cat:.4f} | Accuracy={acc_cat:.4f} | Log Loss={loss_cat:.4f}")
models_res['CatBoost'] = {'prob': prob_cat, 'f1': f1_cat, 'acc': acc_cat, 'loss': loss_cat, 'time': t_cat, 'model': catboost_clf}

# -----------------------------------------------------------------------------
# 4. Also fit a fast Random Forest for extra diversity
# -----------------------------------------------------------------------------
print("\n---> Training Random Forest for Diversity ...")
t_start = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_tr_proc, y_tr)
prob_rf = rf.predict_proba(X_val_proc)
pred_rf = rf.predict(X_val_proc)
f1_rf = f1_score(y_val, pred_rf, average='macro')
print(f"Random Forest Macro F1: {f1_rf:.4f}")

# -----------------------------------------------------------------------------
# 5. Grand Tri-GBDT + RF Blend
# -----------------------------------------------------------------------------
print("\n=== EVALUATING MULTI-GBDT ENSEMBLE BLENDS ===")

# Blend A: Equal GBDT Blend (LightGBM + XGBoost + CatBoost)
prob_gbdt3 = (prob_lgb + prob_xgb + prob_cat) / 3.0
pred_gbdt3 = np.argmax(prob_gbdt3, axis=1) + 1
f1_gbdt3 = f1_score(y_val, pred_gbdt3, average='macro')
acc_gbdt3 = accuracy_score(y_val, pred_gbdt3)
loss_gbdt3 = log_loss(y_val_0, prob_gbdt3)
print(f"[Tri-GBDT Blend: LGB + XGB + Cat] Macro F1: {f1_gbdt3:.4f} | Accuracy: {acc_gbdt3:.4f} | Log Loss: {loss_gbdt3:.4f}")

# Blend B: Weighted GBDT + RF Blend
prob_grand = 0.35 * prob_lgb + 0.30 * prob_xgb + 0.25 * prob_cat + 0.10 * prob_rf
pred_grand = np.argmax(prob_grand, axis=1) + 1
f1_grand = f1_score(y_val, pred_grand, average='macro')
acc_grand = accuracy_score(y_val, pred_grand)
loss_grand = log_loss(y_val_0, prob_grand)
print(f"[Grand Ensemble: 35% LGB + 30% XGB + 25% Cat + 10% RF] Macro F1: {f1_grand:.4f} | Accuracy: {acc_grand:.4f} | Log Loss: {loss_grand:.4f}")

per_class_f1 = f1_score(y_val, pred_grand, average=None)
print(f"Grand Ensemble Per-Class F1s (Tiers 1-4): {[round(x, 4) for x in per_class_f1]}")

# -----------------------------------------------------------------------------
# 6. Save Comparison Plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
models_names = ['Random Forest', 'CatBoost', 'HistGB', 'XGBoost', 'LightGBM', 'Tri-GBDT Blend', 'Grand Ensemble']
macro_f1s = [f1_rf, f1_cat, 0.6945, f1_xgb, f1_lgb, f1_gbdt3, f1_grand]
colors = ['#386cb0', '#e6ab02', '#4daf4a', '#ff7f00', '#2ca02c', '#984ea3', '#e41a1c']

bars = ax.bar(models_names, macro_f1s, color=colors, width=0.55)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=9.5, weight='bold')

ax.set_ylim(0.68, 0.735)
ax.set_ylabel('Validation Macro F1 Score', fontsize=11, weight='bold')
plt.title("Comparison of Top Tree Ensembles: LightGBM vs XGBoost vs CatBoost vs Blends", fontsize=13, weight='bold')
plt.xticks(fontsize=10, weight='bold', rotation=10)
plt.tight_layout()
plt.savefig('figures/fig_boosting_triad_comparison.png', dpi=300)
plt.close()
print("Saved figures/fig_boosting_triad_comparison.png")

# -----------------------------------------------------------------------------
# 7. Generate Test Set Predictions from Grand Ensemble
# -----------------------------------------------------------------------------
print("\nGenerating Test Predictions using Grand Ensemble...")
t_test = time.time()
test_prob_lgb = lgbm.predict_proba(X_test_proc)
test_prob_xgb = xgboost_clf.predict_proba(X_test_proc)
test_prob_cat = catboost_clf.predict_proba(X_test_proc)
test_prob_rf = rf.predict_proba(X_test_proc)

test_prob_grand = 0.35 * test_prob_lgb + 0.30 * test_prob_xgb + 0.25 * test_prob_cat + 0.10 * test_prob_rf
final_grand_preds = np.argmax(test_prob_grand, axis=1) + 1

# Save as primary submission and backup
sub_grand_df = pd.DataFrame({'ID': test_ids, 'price_tier': final_grand_preds})
sub_grand_df.to_csv('data/submission.csv', index=False)
sub_grand_df.to_csv('data/submission_grand_ensemble.csv', index=False)
print(f"Saved grand ensemble predictions to data/submission.csv in {time.time()-t_test:.2f}s!")
print(sub_grand_df.head(10))
print("\nGrand Ensemble Test Class Distribution:")
print(sub_grand_df['price_tier'].value_counts(normalize=True).sort_index().round(4) * 100)

# Save results log
results_dict = {
    'LightGBM': {'macro_f1': round(f1_lgb, 4), 'accuracy': round(acc_lgb, 4), 'log_loss': round(loss_lgb, 4), 'fit_time_s': round(t_lgb, 1)},
    'XGBoost': {'macro_f1': round(f1_xgb, 4), 'accuracy': round(acc_xgb, 4), 'log_loss': round(loss_xgb, 4), 'fit_time_s': round(t_xgb, 1)},
    'CatBoost': {'macro_f1': round(f1_cat, 4), 'accuracy': round(acc_cat, 4), 'log_loss': round(loss_cat, 4), 'fit_time_s': round(t_cat, 1)},
    'RandomForest': {'macro_f1': round(f1_rf, 4), 'accuracy': round(accuracy_score(y_val, pred_rf), 4)},
    'TriGBDT_Blend': {'macro_f1': round(f1_gbdt3, 4), 'accuracy': round(acc_gbdt3, 4), 'log_loss': round(loss_gbdt3, 4)},
    'GrandEnsemble': {'macro_f1': round(f1_grand, 4), 'accuracy': round(acc_grand, 4), 'log_loss': round(loss_grand, 4)}
}
with open('document/boosting_triad_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)
print("Saved document/boosting_triad_results.json")

print("\n=== EXPERIMENTATION COMPLETE ===")
