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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

print("=== TESTING HIERARCHICAL SPATIAL-CAPACITY TARGET ENCODING + 5-FOLD ENSEMBLE ===")

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Drop redundant duplicate columns
for df in [train, test]:
    df.drop(columns=['min_term_floor', 'max_term_ceiling'], errors='ignore', inplace=True)

# Import existing feature extraction
from src.feature_engineering import extract_features

print("Extracting base features...")
X_train = extract_features(train, train_ref=train, is_train=True)
y_train = X_train.pop('price_tier').astype(int)

X_test = extract_features(test, train_ref=train, is_train=False)
test_ids = test['space_id'].values

# Create composite group keys as string identifiers
group_keys = [
    ('loc_seats_class', ['geo_lat', 'geo_lon', 'max_seats', 'space_class']),
    ('loc_operator', ['geo_lat', 'geo_lon', 'operator_ref']),
    ('loc_seats', ['geo_lat', 'geo_lon', 'max_seats']),
    ('zone_class', ['zone', 'space_class']),
    ('zone', ['zone']),
    ('metro', ['metro'])
]

print("Constructing group keys...")
for name, cols in group_keys:
    k_tr = train[cols[0]].astype(str)
    k_te = test[cols[0]].astype(str)
    for col in cols[1:]:
        k_tr = k_tr + '_' + train[col].astype(str)
        k_te = k_te + '_' + test[col].astype(str)
    train[name] = k_tr
    test[name] = k_te

# Compute Out-Of-Fold Bayesian Target Statistics for these composite keys
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, _ in group_keys:
    X_train[f'{name}_te_mean'] = 0.0
    X_test[f'{name}_te_mean'] = 0.0

global_mean = y_train.mean()

# Smoothing parameter m depends on cardinality/granularity
smoothing = {
    'loc_seats_class': 3,
    'loc_operator': 3,
    'loc_seats': 5,
    'zone_class': 10,
    'zone': 10,
    'metro': 20
}

print("Computing out-of-fold target encodings...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    tr_y = y_train.iloc[tr_idx]
    for name, _ in group_keys:
        m = smoothing[name]
        tr_key = train[name].iloc[tr_idx]
        val_key = train[name].iloc[val_idx]
        
        stats = tr_y.groupby(tr_key).agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
        mapping = smoothed.to_dict()
        
        X_train.loc[X_train.index[val_idx], f'{name}_te_mean'] = val_key.map(mapping).fillna(global_mean)

# Full train statistics for test set
for name, _ in group_keys:
    m = smoothing[name]
    stats = y_train.groupby(train[name]).agg(['count', 'mean'])
    smoothed = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
    mapping = smoothed.to_dict()
    X_test[f'{name}_te_mean'] = test[name].map(mapping).fillna(global_mean)

print("Target encoding complete!")

# Identify categorical and numeric columns
cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == 'category']
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols)
    ]
)

print("Fitting preprocessor...")
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)
print(f"Feature matrix shape: {X_train_p.shape}")

# Evaluate with 5-Fold Stratified CV on LightGBM
print("\n--- Running 5-Fold Cross Validation with LightGBM ---")
oof_preds_lgb = np.zeros((len(train), 4))
test_preds_lgb = np.zeros((len(test), 4))

y_train_0 = y_train.values - 1
class_weights = {0: 1.05, 1: 0.85, 2: 1.25, 3: 0.95}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_p, y_train_0)):
    X_tr_fold, y_tr_fold = X_train_p[tr_idx], y_train_0[tr_idx]
    X_val_fold, y_val_fold = X_train_p[val_idx], y_train_0[val_idx]
    
    sample_w = np.array([class_weights[y] for y in y_tr_fold])
    
    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,
        n_estimators=450,
        learning_rate=0.06,
        num_leaves=95,
        max_depth=12,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_samples=25,
        random_state=42 + fold,
        n_jobs=-1
    )
    clf.fit(X_tr_fold, y_tr_fold, sample_weight=sample_w)
    
    val_prob = clf.predict_proba(X_val_fold)
    oof_preds_lgb[val_idx] = val_prob
    test_preds_lgb += clf.predict_proba(X_test_p) / 5.0
    
    val_pred_fold = np.argmax(val_prob, axis=1)
    f1_fold = f1_score(y_val_fold, val_pred_fold, average='macro')
    print(f"  Fold {fold+1} Macro F1: {f1_fold:.4f}")

oof_pred_labels = np.argmax(oof_preds_lgb, axis=1) + 1
raw_cv_f1 = f1_score(y_train, oof_pred_labels, average='macro')
print(f"\nOverall OOF LightGBM Macro F1 (Raw Argmax): {raw_cv_f1:.4f}")

# Threshold Optimization on Out-Of-Fold Probabilities
print("\n--- Optimizing Decision Multipliers on 100% OOF Probabilities ---")
def loss_func(weights):
    scaled = oof_preds_lgb * weights
    preds = np.argmax(scaled, axis=1) + 1
    return -f1_score(y_train, preds, average='macro')

res = minimize(loss_func, [1.0, 1.0, 1.0, 1.0], method='Nelder-Mead', options={'maxiter': 300})
best_w = res.x / res.x[0]
print(f"Optimal Multipliers: {[round(w, 4) for w in best_w]}")

opt_oof_preds = np.argmax(oof_preds_lgb * best_w, axis=1) + 1
opt_cv_f1 = f1_score(y_train, opt_oof_preds, average='macro')
print(f"Optimized OOF LightGBM Macro F1: {opt_cv_f1:.4f}")
print("Per-class F1:", [round(x, 4) for x in f1_score(y_train, opt_oof_preds, average=None)])
print("\nOOF Classification Report:\n", classification_report(y_train, opt_oof_preds, target_names=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'], digits=4))

# Check exact match test override impact
exact_key = ['geo_lat', 'geo_lon', 'max_seats', 'operator_ref']
train_mode_map = train.dropna(subset=exact_key).groupby(exact_key)['price_tier'].agg(
    lambda x: x.mode().iloc[0]
).to_dict()

# Count how many test predictions would change with exact match
test_scaled_prob = test_preds_lgb * best_w
test_preds_final = np.argmax(test_scaled_prob, axis=1) + 1

override_count = 0
match_count = 0
test_keys_list = list(zip(test['geo_lat'], test['geo_lon'], test['max_seats'], test['operator_ref']))

test_preds_with_override = test_preds_final.copy()
for i, key_val in enumerate(test_keys_list):
    if key_val in train_mode_map:
        match_count += 1
        hist_mode = train_mode_map[key_val]
        if test_preds_with_override[i] != hist_mode:
            override_count += 1
            test_preds_with_override[i] = hist_mode

print(f"\nExact Match Check:")
print(f"  Test rows matching train exact key: {match_count:,} (out of {len(test):,})")
print(f"  Predictions adjusted by exact match override: {override_count:,} ({override_count/len(test)*100:.2f}%)")

# Save test predictions
sub = pd.DataFrame({'ID': test_ids, 'price_tier': test_preds_with_override})
sub.to_csv('data/submission_oof_lgb_hierarchical.csv', index=False)
print("Saved data/submission_oof_lgb_hierarchical.csv!")
print(sub.head(10))
print("\nSubmission class distribution:\n", (sub['price_tier'].value_counts(normalize=True).sort_index()*100).round(2).to_string())
