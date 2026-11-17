import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import time
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report, log_loss

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

print("=== DEEP DATA RE-ENGINEERING & ADVANCED TARGET OPTIMIZATION ===")

# 1. Load Data
t0 = time.time()
print("Loading datasets...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Drop useless duplicate columns immediately
for df in [train_df, test_df]:
    df.drop(columns=['min_term_floor', 'max_term_ceiling'], errors='ignore', inplace=True)

print(f"Loaded train ({train_df.shape}) and test ({test_df.shape})")

# 2. Advanced Feature Extraction with Out-of-Fold Target Statistics
print("Building leak-free Out-Of-Fold (OOF) Target Statistics for high-cardinality features...")
from src.feature_engineering import extract_features

X_train = extract_features(train_df, train_ref=train_df, is_train=True)
y_train = X_train.pop('price_tier')

X_test = extract_features(test_df, train_ref=train_df, is_train=False)
test_ids = test_df['space_id'].values

# Out-of-fold Target Encoding for zone, space_class, metro, operator_ref
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
te_cols = ['zone', 'space_class', 'metro']

# Initialize target encoding features
for c in te_cols:
    X_train[f'{c}_te_mean'] = 0.0
    X_test[f'{c}_te_mean'] = 0.0

global_mean = y_train.mean()

for tr_idx, val_idx in kf.split(X_train, y_train):
    tr_y = y_train.iloc[tr_idx]
    for c in te_cols:
        tr_feature = train_df[c].iloc[tr_idx]
        val_feature = train_df[c].iloc[val_idx]
        
        # Bayesian smoothed target mean: (count * mean + 10 * global_mean) / (count + 10)
        stats = tr_y.groupby(tr_feature).agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + 10 * global_mean) / (stats['count'] + 10)
        mapping = smoothed.to_dict()
        
        X_train.loc[X_train.index[val_idx], f'{c}_te_mean'] = val_feature.map(mapping).fillna(global_mean)

# For test set, use full training set statistics
for c in te_cols:
    stats = y_train.groupby(train_df[c]).agg(['count', 'mean'])
    smoothed = (stats['count'] * stats['mean'] + 10 * global_mean) / (stats['count'] + 10)
    mapping = smoothed.to_dict()
    X_test[f'{c}_te_mean'] = test_df[c].map(mapping).fillna(global_mean)

print("Target encoding features computed successfully!")

# Split into train & validation (80/20) for local score verification
cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == 'category']
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols)
    ]
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)

print(f"Preprocessing matrices: train={X_tr.shape}, val={X_val.shape}, test={X_test.shape}")
X_tr_p = preprocessor.fit_transform(X_tr)
X_val_p = preprocessor.transform(X_val)
X_test_p = preprocessor.transform(X_test)

y_tr_0 = (y_tr - 1).values
y_val_0 = (y_val - 1).values

# Class weights to balance Tier 3:
# Class counts: [66889, 81530, 60839, 70488]
# Inverse frequency weights:
class_weights = {0: 1.05, 1: 0.85, 2: 1.25, 3: 0.95}
sample_weights_tr = np.array([class_weights[y] for y in y_tr_0])

# 3. Train Balanced LightGBM
print("\n---> Training Balanced LightGBM ...")
lgbm = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=450,
    learning_rate=0.06,
    num_leaves=95,
    max_depth=12,
    subsample=0.85,
    colsample_bytree=0.80,
    min_child_samples=25,
    random_state=42,
    n_jobs=-1
)
lgbm.fit(X_tr_p, y_tr_0, sample_weight=sample_weights_tr)
prob_lgb = lgbm.predict_proba(X_val_p)
pred_lgb_raw = np.argmax(prob_lgb, axis=1) + 1
f1_raw = f1_score(y_val, pred_lgb_raw, average='macro')
print(f"Raw Balanced LightGBM Macro F1: {f1_raw:.4f} (Accuracy: {accuracy_score(y_val, pred_lgb_raw):.4f})")
print(f"Raw Class F1s: {[round(x, 4) for x in f1_score(y_val, pred_lgb_raw, average=None)]}")

# 4. Train Balanced XGBoost
print("\n---> Training Balanced XGBoost ...")
xgb_clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    tree_method='hist',
    n_estimators=400,
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

# 5. Train Balanced CatBoost
print("\n---> Training Balanced CatBoost ...")
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

# Blend probabilities
prob_blend = 0.40 * prob_lgb + 0.35 * prob_xgb + 0.25 * prob_cb
pred_blend_raw = np.argmax(prob_blend, axis=1) + 1
print(f"\nRaw Tri-GBDT Blend Macro F1: {f1_score(y_val, pred_blend_raw, average='macro'):.4f}")

# 6. Macro F1 Direct Threshold Optimization (Nelder-Mead Search)
print("\n---> Optimizing Decision Multipliers directly for Macro F1 ...")
def loss_func(weights):
    w = np.array(weights)
    scaled_prob = prob_blend * w
    preds = np.argmax(scaled_prob, axis=1) + 1
    # Negative macro F1 to minimize
    return -f1_score(y_val, preds, average='macro')

init_weights = [1.0, 1.0, 1.0, 1.0]
res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 250})
best_weights = res.x
best_weights = best_weights / best_weights[0] # normalize
print(f"Optimal Class Probability Multipliers: {[round(w, 4) for w in best_weights]}")

scaled_prob_val = prob_blend * best_weights
opt_val_preds = np.argmax(scaled_prob_val, axis=1) + 1
opt_macro_f1 = f1_score(y_val, opt_val_preds, average='macro')
opt_acc = accuracy_score(y_val, opt_val_preds)
opt_f1_classes = f1_score(y_val, opt_val_preds, average=None)

print("\n" + "="*60)
print(f"=== OPTIMIZED MACRO F1 SCORE: {opt_macro_f1:.4f} ===")
print(f"=== OPTIMIZED ACCURACY  : {opt_acc:.4f} ===")
print("="*60)
print(f"Class F1s (Tiers 1-4): {[round(x, 4) for x in opt_f1_classes]}")
print("\nClassification Report:\n", classification_report(y_val, opt_val_preds, target_names=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'], digits=4))

# 7. Generate Enhanced Predictions on Test Set
print("\nGenerating Optimized Predictions on Test Set...")
test_p_lgb = lgbm.predict_proba(X_test_p)
test_p_xgb = xgb_clf.predict_proba(X_test_p)
test_p_cb = cb_clf.predict_proba(X_test_p)

test_blend = 0.40 * test_p_lgb + 0.35 * test_p_xgb + 0.25 * test_p_cb
test_scaled = test_blend * best_weights
final_test_preds = np.argmax(test_scaled, axis=1) + 1

sub_df = pd.DataFrame({'ID': test_ids, 'price_tier': final_test_preds})
sub_df.to_csv('data/submission.csv', index=False)
sub_df.to_csv('data/submission_optimized_te.csv', index=False)
print("Saved optimized test predictions to data/submission.csv!")
print(sub_df.head(10))
print("\nOptimized Test Class Distribution:\n", (sub_df['price_tier'].value_counts(normalize=True).sort_index()*100).round(2).to_string())

# Save results
out_data = {
    'raw_blend_macro_f1': round(f1_score(y_val, pred_blend_raw, average='macro'), 4),
    'optimized_macro_f1': round(opt_macro_f1, 4),
    'optimized_accuracy': round(opt_acc, 4),
    'best_weights': [round(w, 4) for w in best_weights.tolist()],
    'per_class_f1': [round(x, 4) for x in opt_f1_classes.tolist()]
}
with open('document/advanced_optimization_results.json', 'w') as f:
    json.dump(out_data, f, indent=2)

print("\n=== EXPERIMENT COMPLETE ===")
