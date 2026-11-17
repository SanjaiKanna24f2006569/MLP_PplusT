import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
import lightgbm as lgb
from scipy.optimize import minimize

print("=== EVALUATING GBDT WITH HIERARCHICAL MICRO-SPATIAL TARGET FEATURES ===")

t0 = time.time()
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Drop useless duplicate columns
for df in [train, test]:
    df.drop(columns=['min_term_floor', 'max_term_ceiling'], errors='ignore', inplace=True)

# Feature extraction
from src.feature_engineering import extract_features

print("Extracting base features...")
X_train = extract_features(train, train_ref=train, is_train=True)
y_train = X_train.pop('price_tier').astype(int).values

X_test = extract_features(test, train_ref=train, is_train=False)
test_ids = test['space_id'].values

# Keys
def make_key(df, cols):
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + '__' + df[c].astype(str)
    return k

for df in [train, test]:
    df['k_exact'] = make_key(df, ['geo_lat', 'geo_lon', 'max_seats', 'operator_ref'])
    df['k_seat_class'] = make_key(df, ['geo_lat', 'geo_lon', 'max_seats', 'space_class'])
    df['k_seat'] = make_key(df, ['geo_lat', 'geo_lon', 'max_seats'])
    df['k_operator'] = make_key(df, ['geo_lat', 'geo_lon', 'operator_ref'])
    df['k_loc'] = make_key(df, ['geo_lat', 'geo_lon'])
    df['k_zone_class'] = make_key(df, ['zone', 'space_class'])

# 5-Fold Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# One-hot encode y
y_oh = np.zeros((len(train), 4), dtype=np.float32)
for c in range(1, 5):
    y_oh[:, c-1] = (y_train == c).astype(np.float32)

prior_probs = y_oh.mean(axis=0)

# Initialize spatial prior features in X_train and X_test
for i in range(1, 5):
    X_train[f'spatial_p{i}'] = 0.0
    X_test[f'spatial_p{i}'] = 0.0
X_train['spatial_expected_tier'] = 0.0
X_test['spatial_expected_tier'] = 0.0
X_train['spatial_confidence'] = 0.0
X_test['spatial_confidence'] = 0.0

# Hierarchy definition
levels = [
    ('k_exact', 2.0),
    ('k_seat_class', 3.0),
    ('k_operator', 3.0),
    ('k_seat', 5.0),
    ('k_loc', 10.0),
    ('k_zone_class', 15.0)
]

print("Calculating 5-Fold Out-of-Fold hierarchical spatial features...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(train, y_train)):
    tr_df = train.iloc[tr_idx]
    tr_y_oh = y_oh[tr_idx]
    val_df = train.iloc[val_idx]
    
    val_p = np.tile(prior_probs, (len(val_df), 1))
    
    # Hierarchical lookup from coarse to fine
    for key_col, m in reversed(levels):
        grouped_counts = tr_df.groupby(key_col).size()
        class_sums = pd.DataFrame(tr_y_oh, index=tr_df.index).groupby(tr_df[key_col]).sum()
        smoothed_p = (class_sums.values + m * prior_probs) / (grouped_counts.values[:, None] + m)
        table = pd.DataFrame(smoothed_p, index=grouped_counts.index)
        
        match_idx = val_df[key_col].isin(table.index)
        if match_idx.any():
            matched_keys = val_df.loc[match_idx, key_col]
            val_p[match_idx] = table.loc[matched_keys].values
            
    for i in range(4):
        X_train.loc[X_train.index[val_idx], f'spatial_p{i+1}'] = val_p[:, i]
    exp_tier = val_p @ np.array([1.0, 2.0, 3.0, 4.0])
    X_train.loc[X_train.index[val_idx], 'spatial_expected_tier'] = exp_tier
    X_train.loc[X_train.index[val_idx], 'spatial_confidence'] = val_p.max(axis=1)

# Full train statistics for test set
test_p = np.tile(prior_probs, (len(test), 1))
for key_col, m in reversed(levels):
    grouped_counts = train.groupby(key_col).size()
    class_sums = pd.DataFrame(y_oh, index=train.index).groupby(train[key_col]).sum()
    smoothed_p = (class_sums.values + m * prior_probs) / (grouped_counts.values[:, None] + m)
    table = pd.DataFrame(smoothed_p, index=grouped_counts.index)
    
    match_idx = test[key_col].isin(table.index)
    if match_idx.any():
        matched_keys = test.loc[match_idx, key_col]
        test_p[match_idx] = table.loc[matched_keys].values

for i in range(4):
    X_test[f'spatial_p{i+1}'] = test_p[:, i]
exp_tier_test = test_p @ np.array([1.0, 2.0, 3.0, 4.0])
X_test['spatial_expected_tier'] = exp_tier_test
X_test['spatial_confidence'] = test_p.max(axis=1)

print("Spatial features computed successfully!")

# Preprocessing
cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == 'category']
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols)
    ]
)

print("Transforming features...")
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)
print(f"Matrix shape: {X_train_p.shape}")

# Train 5-Fold LightGBM
print("\n--- Training 5-Fold LightGBM with Spatial Priors ---")
oof_probs = np.zeros((len(train), 4), dtype=np.float32)
test_probs = np.zeros((len(test), 4), dtype=np.float32)

y_train_0 = y_train - 1
class_weights = {0: 1.05, 1: 0.85, 2: 1.25, 3: 0.95}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_p, y_train_0)):
    X_tr_f, y_tr_f = X_train_p[tr_idx], y_train_0[tr_idx]
    X_val_f, y_val_f = X_train_p[val_idx], y_train_0[val_idx]
    
    sample_w = np.array([class_weights[y] for y in y_tr_f])
    
    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=4,
        n_estimators=500,
        learning_rate=0.06,
        num_leaves=100,
        max_depth=12,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_samples=25,
        random_state=42 + fold,
        n_jobs=-1
    )
    clf.fit(X_tr_f, y_tr_f, sample_weight=sample_w)
    
    v_prob = clf.predict_proba(X_val_f)
    oof_probs[val_idx] = v_prob
    test_probs += clf.predict_proba(X_test_p) / 5.0
    
    f1_f = f1_score(y_train[val_idx], np.argmax(v_prob, axis=1) + 1, average='macro')
    print(f"  Fold {fold+1} LightGBM Macro F1: {f1_f:.4f}")

raw_oof_preds = np.argmax(oof_probs, axis=1) + 1
raw_f1 = f1_score(y_train, raw_oof_preds, average='macro')
raw_acc = accuracy_score(y_train, raw_oof_preds)
print(f"\n>>> Raw 5-Fold OOF LightGBM Macro F1: {raw_f1:.4f} (Accuracy: {raw_acc:.4f}) <<<")

# Nelder-Mead Optimization on 100% OOF
print("\n--- Optimizing Decision Multipliers on 100% Out-Of-Fold Probabilities ---")
def loss_func(weights):
    scaled = oof_probs * weights
    preds = np.argmax(scaled, axis=1) + 1
    return -f1_score(y_train, preds, average='macro')

res = minimize(loss_func, [1.0, 1.0, 1.0, 1.0], method='Nelder-Mead', options={'maxiter': 350})
best_w = res.x / res.x[0]
print(f"Optimal Multipliers: {[round(w, 4) for w in best_w]}")

opt_preds = np.argmax(oof_probs * best_w, axis=1) + 1
opt_f1 = f1_score(y_train, opt_preds, average='macro')
opt_acc = accuracy_score(y_train, opt_preds)
print(f"\n=======================================================")
print(f">>> FINAL OPTIMIZED OOF MACRO F1 SCORE: {opt_f1:.4f} <<<")
print(f">>> FINAL OPTIMIZED OOF ACCURACY       : {opt_acc:.4f} <<<")
print(f"=======================================================")
print("Per-class F1:", [round(x, 4) for x in f1_score(y_train, opt_preds, average=None)])
print("\nClassification Report:\n", classification_report(y_train, opt_preds, target_names=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'], digits=4))

# Generate Test Predictions
test_scaled = test_probs * best_w
test_preds = np.argmax(test_scaled, axis=1) + 1

# Check exact key override for 97.85% pure items
exact_map = train.dropna(subset=['k_exact']).groupby('k_exact')['price_tier'].agg(
    lambda x: x.mode().iloc[0]
).to_dict()

overridden = 0
for i, k in enumerate(test['k_exact']):
    if k in exact_map:
        hist_tier = exact_map[k]
        if test_preds[i] != hist_tier:
            overridden += 1
            test_preds[i] = hist_tier

print(f"Total test predictions adjusted with 98% exact historical matches: {overridden:,}")

sub = pd.DataFrame({'ID': test_ids, 'price_tier': test_preds})
sub.to_csv('data/submission_spatial_lgbm_5fold.csv', index=False)
print("Saved data/submission_spatial_lgbm_5fold.csv successfully!")
print(sub.head(10))
print("\nPredicted Class Proportions:\n", (sub['price_tier'].value_counts(normalize=True).sort_index()*100).round(2).to_string())
