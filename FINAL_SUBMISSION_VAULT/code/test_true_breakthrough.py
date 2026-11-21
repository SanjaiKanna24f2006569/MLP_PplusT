import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import time
import re
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
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

print("=== REBUILDING THE PROVEN TRI-GBDT FOUNDATION (NO MICRO-SPATIAL OVERFITTING) ===")

t0 = time.time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Drop useless duplicate columns
for df in [train_df, test_df]:
    df.drop(columns=['min_term_floor', 'max_term_ceiling'], errors='ignore', inplace=True)

# 1. Amenity Quality Index (144 amenities, high sample size, zero coordinate leakage)
print("Computing robust Amenity Quality Index across 144 facility codes...")
all_facs = []
for s in train_df['facilities'].dropna():
    all_facs.extend(re.findall(r'FCL-\d+', s))
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
            matches = re.findall(r'FCL-\d+', s)
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
print(f"Amenity Quality Index Pearson correlation: {train_amenity_df['amenity_quality_score'].corr(train_df['price_tier']):.4f}")

# 2. Base Domain Features
from src.feature_engineering import extract_features
print("Extracting base features...")
X_train = extract_features(train_df, train_ref=train_df, is_train=True)
y_train = X_train.pop('price_tier').astype(int)

X_test = extract_features(test_df, train_ref=train_df, is_train=False)
test_ids = test_df['space_id'].values

# Attach Amenity Features
for c in train_amenity_df.columns:
    X_train[c] = train_amenity_df[c]
    X_test[c] = test_amenity_df[c]

# 3. Robust Regional Target Encoding (Zone, Space Class, Metro only - NO coordinate memorization!)
print("Computing robust regional target encodings (Zone, Space Class, Metro)...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
te_cols = ['zone', 'space_class', 'metro']
global_mean = y_train.mean()

for c in te_cols:
    X_train[f'{c}_te_mean'] = 0.0
    X_test[f'{c}_te_mean'] = 0.0

for tr_idx, val_idx in kf.split(X_train, y_train):
    tr_y = y_train.iloc[tr_idx]
    for c in te_cols:
        tr_cat = train_df[c].iloc[tr_idx]
        val_cat = train_df[c].iloc[val_idx]
        stats = tr_y.groupby(tr_cat).agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + 15 * global_mean) / (stats['count'] + 15)
        mapping = smoothed.to_dict()
        X_train.loc[X_train.index[val_idx], f'{c}_te_mean'] = val_cat.map(mapping).fillna(global_mean)

for c in te_cols:
    stats = y_train.groupby(train_df[c]).agg(['count', 'mean'])
    smoothed = (stats['count'] * stats['mean'] + 15 * global_mean) / (stats['count'] + 15)
    mapping = smoothed.to_dict()
    X_test[f'{c}_te_mean'] = test_df[c].map(mapping).fillna(global_mean)

# Split 80/20 for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)

cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == 'category']
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols)
    ]
)

print("Transforming matrices...")
X_tr_p = preprocessor.fit_transform(X_tr)
X_val_p = preprocessor.transform(X_val)
X_test_p = preprocessor.transform(X_test)

y_tr_0 = (y_tr - 1).values
y_val_0 = (y_val - 1).values

# Class weights: penalize Tier 3 errors gently without destroying Tier 2
class_weights = {0: 1.02, 1: 0.95, 2: 1.15, 3: 0.98}
sample_weights_tr = np.array([class_weights[y] for y in y_tr_0])

# 4. Train LightGBM
print("\n---> Training High-Capacity LightGBM ...")
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
print(f"LightGBM Macro F1: {f1_lgb:.4f}")

# 5. Train XGBoost
print("\n---> Training XGBoost ...")
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
print(f"XGBoost Macro F1: {f1_xgb:.4f}")

# 6. Train CatBoost
print("\n---> Training CatBoost ...")
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
print(f"CatBoost Macro F1: {f1_cb:.4f}")

# 7. Tri-GBDT Blend
prob_blend = 0.40 * prob_lgb + 0.35 * prob_xgb + 0.25 * prob_cb
raw_blend_f1 = f1_score(y_val, np.argmax(prob_blend, axis=1) + 1, average='macro')
print(f"\n>>> Raw Tri-GBDT Blend Macro F1: {raw_blend_f1:.4f} <<<")

# Multiplier optimization
def loss_func(weights):
    scaled = prob_blend * weights
    preds = np.argmax(scaled, axis=1) + 1
    return -f1_score(y_val, preds, average='macro')

res = minimize(loss_func, [1.0, 1.0, 1.0, 1.0], method='Nelder-Mead', options={'maxiter': 300})
best_w = res.x / res.x[0]
print(f"Optimal Multipliers: {[round(w, 4) for w in best_w]}")

opt_val_preds = np.argmax(prob_blend * best_w, axis=1) + 1
opt_f1 = f1_score(y_val, opt_val_preds, average='macro')
print(f"\n=======================================================")
print(f">>> OPTIMIZED VALIDATION MACRO F1: {opt_f1:.4f} <<<")
print(f">>> OPTIMIZED VALIDATION ACCURACY: {accuracy_score(y_val, opt_val_preds):.4f} <<<")
print(f"=======================================================")
print("Per-class F1:", [round(x, 4) for x in f1_score(y_val, opt_val_preds, average=None)])
print("\nClassification Report:\n", classification_report(y_val, opt_val_preds, target_names=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'], digits=4))

# 8. Predict on Test Set
test_p_lgb = lgbm.predict_proba(X_test_p)
test_p_xgb = xgb_clf.predict_proba(X_test_p)
test_p_cb = cb_clf.predict_proba(X_test_p)

test_blend = 0.40 * test_p_lgb + 0.35 * test_p_xgb + 0.25 * test_p_cb
test_scaled = test_blend * best_w
final_preds = np.argmax(test_scaled, axis=1) + 1

sub = pd.DataFrame({'ID': test_ids, 'price_tier': final_preds})
sub.to_csv('data/submission_true_breakthrough.csv', index=False)
sub.to_csv('data/submission.csv', index=False)
print("Saved data/submission_true_breakthrough.csv and updated data/submission.csv!")
print("\nPredicted Class Proportions:\n", (sub['price_tier'].value_counts(normalize=True).sort_index()*100).round(2).to_string())

# Check agreement with the 0.69639 submission
s_696 = pd.read_csv('data/submission_optimized_te.csv')
agreement = (sub['price_tier'] == s_696['price_tier']).mean() * 100
print(f"\nAgreement with 0.69639 Leaderboard Submission: {agreement:.2f}%")
