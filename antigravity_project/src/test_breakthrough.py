import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
import lightgbm as lgb
from scipy.optimize import minimize

print("=== DEEP FORENSIC INVESTIGATION: PUSHING PAST 0.75+ ===")

t0 = time.time()
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"Loaded train ({train.shape}) and test ({test.shape}) in {time.time()-t0:.2f}s")

# Let's inspect the exact key match
# We will create hierarchical keys:
# Level 1: geo_lat + geo_lon + max_seats + operator_ref (Purity 97.85%, 6.4% test rows)
# Level 2: geo_lat + geo_lon + max_seats + space_class (Purity 91.14%, 74.09% test rows)
# Level 3: geo_lat + geo_lon + max_seats (Purity 85.50%, 89.64% test rows)
# Level 4: geo_lat + geo_lon (Purity 69.57%, 98.40% test rows)

def make_key(df, cols):
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + '__' + df[c].astype(str)
    return k

train['k_exact'] = make_key(train, ['geo_lat', 'geo_lon', 'max_seats', 'operator_ref'])
train['k_seat_class'] = make_key(train, ['geo_lat', 'geo_lon', 'max_seats', 'space_class'])
train['k_seat'] = make_key(train, ['geo_lat', 'geo_lon', 'max_seats'])
train['k_loc'] = make_key(train, ['geo_lat', 'geo_lon'])
train['k_zone_class'] = make_key(train, ['zone', 'space_class'])

test['k_exact'] = make_key(test, ['geo_lat', 'geo_lon', 'max_seats', 'operator_ref'])
test['k_seat_class'] = make_key(test, ['geo_lat', 'geo_lon', 'max_seats', 'space_class'])
test['k_seat'] = make_key(test, ['geo_lat', 'geo_lon', 'max_seats'])
test['k_loc'] = make_key(test, ['geo_lat', 'geo_lon'])
test['k_zone_class'] = make_key(test, ['zone', 'space_class'])

# Let's test out-of-fold cross-validation on train!
# In 5-fold CV, we compute the target statistics strictly using training folds (NO leakage)
# and evaluate how accurately hierarchical target encoding predicts the validation fold!

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train = train['price_tier'].values

# One-hot encode y_train for calculating class probabilities (P1, P2, P3, P4)
y_oh = np.zeros((len(train), 4), dtype=np.float32)
for c in range(1, 5):
    y_oh[:, c-1] = (y_train == c).astype(np.float32)

prior_probs = y_oh.mean(axis=0) # global class prior: [0.2391, 0.2914, 0.2175, 0.2520]
print(f"Global Class Prior: {[round(p, 4) for p in prior_probs]}")

# Let's evaluate a pure hierarchical bayesian lookup on out-of-fold folds!
oof_priors = np.zeros((len(train), 4), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(kf.split(train, y_train)):
    tr_df = train.iloc[tr_idx]
    tr_y_oh = y_oh[tr_idx]
    val_df = train.iloc[val_idx]
    
    # Hierarchical tables with smoothing weights:
    # m is pseudo-counts
    levels = [
        ('k_exact', 2.0),
        ('k_seat_class', 3.0),
        ('k_seat', 5.0),
        ('k_loc', 10.0),
        ('k_zone_class', 15.0)
    ]
    
    fold_priors = {}
    for key_col, m in levels:
        # group sum and count
        grouped_counts = tr_df.groupby(key_col).size()
        # group class counts
        class_sums = pd.DataFrame(tr_y_oh, index=tr_df.index).groupby(tr_df[key_col]).sum()
        # smoothed probabilities
        smoothed_p = (class_sums.values + m * prior_probs) / (grouped_counts.values[:, None] + m)
        fold_priors[key_col] = pd.DataFrame(smoothed_p, index=grouped_counts.index)

    # For validation rows, evaluate hierarchical lookup
    val_probs = np.tile(prior_probs, (len(val_df), 1))
    
    # Step through levels from coarsest to finest
    for key_col, m in reversed(levels):
        table = fold_priors[key_col]
        # find matching indices
        match_idx = val_df[key_col].isin(table.index)
        if match_idx.any():
            matched_keys = val_df.loc[match_idx, key_col]
            val_probs[match_idx] = table.loc[matched_keys].values
            
    oof_priors[val_idx] = val_probs
    val_pred = np.argmax(val_probs, axis=1) + 1
    f1 = f1_score(y_train[val_idx], val_pred, average='macro')
    acc = accuracy_score(y_train[val_idx], val_pred)
    print(f"Fold {fold+1} Pure Spatial-Capacity Prior -> Macro F1: {f1:.4f} | Accuracy: {acc:.4f}")

oof_prior_pred = np.argmax(oof_priors, axis=1) + 1
print(f"\n>>> OVERALL OOF PURE SPATIAL PRIOR MACRO F1: {f1_score(y_train, oof_prior_pred, average='macro'):.4f} <<<")
print(f">>> OVERALL OOF PURE SPATIAL PRIOR ACCURACY: {accuracy_score(y_train, oof_prior_pred):.4f} <<<")
print("Per-class F1:\n", f1_score(y_train, oof_prior_pred, average=None))
