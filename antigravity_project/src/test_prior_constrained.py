import os
import sys
import time
import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

print("=== TESTING PRIOR-CONSTRAINED TRI-GBDT WITH UNVEILED FEATURES ===")
t0 = time.time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Drop redundant duplicate columns
for df in [train_df, test_df]:
    df.drop(columns=['min_term_floor', 'max_term_ceiling'], errors='ignore', inplace=True)

# 1. Amenity Quality Index
print("Extracting Amenity Quality Index...")
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

train_amenity = extract_amenity_features(train_df)
test_amenity = extract_amenity_features(test_df)

# 2. Comprehensive Domain & Unfolded Feature Extraction
print("Extracting domain & newly unfolded signals...")
TOP_FACILITIES = ['FCL-069', 'FCL-089', 'FCL-120', 'FCL-011', 'FCL-007', 'FCL-055', 'FCL-071', 'FCL-097', 'FCL-035', 'FCL-098']
SPACE_KEYWORDS = ['Studio', 'Workroom', 'Flex', 'Boutique', 'Premium', 'Bay', 'Ground', 'Suite', 'Loft', 'Niche', 'Alcove']

def extract_all_features(df, train_ref=None):
    out = pd.DataFrame(index=df.index)
    if 'price_tier' in df.columns:
        out['price_tier'] = df['price_tier'].astype(np.int8)

    # Raw numerics
    raw_nums = [
        'geo_lat', 'geo_lon', 'max_seats', 'utility_rooms', 'zones', 'workstations',
        'min_term_days', 'max_term_days', 'min_term_ceiling', 'max_term_floor',
        'min_term_avg', 'max_term_avg',
        'open_slots_a', 'open_slots_b', 'open_slots_c', 'open_slots_d', 'open_slots_e',
        'review_count', 'review_count_12m', 'review_count_30d', 'review_count_prev_yr',
        'booked_load', 'score_overall', 'score_accuracy', 'score_condition',
        'score_onboarding', 'score_support', 'score_access', 'score_value',
        'operator_metro_units', 'operator_metro_whole', 'operator_metro_private',
        'operator_metro_shared', 'reviews_per_month', 'operator_portfolio_size',
        'operator_platform_age_years', 'operator_platform_age_months',
        'operator_active_years', 'operator_active_months'
    ]
    for c in raw_nums:
        if c in df.columns:
            out[c] = df[c].astype(np.float32)

    # Structured missingness
    out['is_missing_borough'] = df['borough'].isna().astype(np.int8)
    out['is_missing_reviews'] = df['score_overall'].isna().astype(np.int8)
    out['is_missing_zones'] = df['zones'].isna().astype(np.int8)
    out['is_missing_workstations'] = df['workstations'].isna().astype(np.int8)
    out['is_missing_utility_rooms'] = df['utility_rooms'].isna().astype(np.int8)

    # Sentinel extreme capping
    out['max_term_is_extreme'] = (df['max_term_days'] > 1125).fillna(False).astype(np.int8)
    out['log_max_term_days'] = np.log1p(df['max_term_days'].clip(0, 1125)).astype(np.float32)
    out['log_min_term_days'] = np.log1p(df['min_term_days'].clip(lower=0)).astype(np.float32)
    out['log_portfolio_size'] = np.log1p(df['operator_portfolio_size'].clip(lower=0)).astype(np.float32)
    out['term_flexibility_ratio'] = (out['log_max_term_days'] - out['log_min_term_days']).astype(np.float32)

    # Capacity interactions & ratios
    cap_vol = np.log1p(df['max_seats'] * (df['zones'].fillna(1.0) + 1.0) * (df['utility_rooms'].fillna(1.0) + 1.0)).astype(np.float32)
    out['capacity_volume'] = cap_vol
    out['seats_workstations_product'] = np.log1p(df['max_seats'] * (df['workstations'].fillna(1.0) + 1.0)).astype(np.float32)
    out['seats_per_utility_room'] = (df['max_seats'] / (df['utility_rooms'].fillna(1.0) + 1.0)).astype(np.float32)
    out['seats_per_workstation'] = (df['max_seats'] / (df['workstations'].fillna(2.0) + 1.0)).astype(np.float32)
    out['seats_per_zone'] = (df['max_seats'] / (df['zones'].fillna(1.0) + 1.0)).astype(np.float32)
    out['workstations_per_zone'] = (df['workstations'].fillna(2.0) / (df['zones'].fillna(1.0) + 1.0)).astype(np.float32)
    out['utility_rooms_per_seat'] = (df['utility_rooms'].fillna(1.0) / (df['max_seats'] + 1.0)).astype(np.float32)

    # Operator specialization & chronometry
    plat_age = df['operator_platform_age_years'].fillna(0)*12 + df['operator_platform_age_months'].fillna(0)
    act_age = df['operator_active_years'].fillna(0)*12 + df['operator_active_months'].fillna(0)
    out['total_platform_age_months'] = plat_age.astype(np.float32)
    out['operator_inactivity_months'] = (plat_age - act_age).clip(lower=0).astype(np.float32)

    op_units = df['operator_metro_units'].fillna(0) + 1.0
    out['op_private_ratio'] = (df['operator_metro_private'].fillna(0) / op_units).astype(np.float32)
    out['op_whole_ratio'] = (df['operator_metro_whole'].fillna(0) / op_units).astype(np.float32)
    out['op_shared_ratio'] = (df['operator_metro_shared'].fillna(0) / op_units).astype(np.float32)
    out['op_whole_to_private_ratio'] = (df['operator_metro_whole'].fillna(0) / (df['operator_metro_private'].fillna(0) + 1.0)).astype(np.float32)

    # Snapshot Dynamics
    s_start = pd.to_datetime(df['snapshot_start_date'], errors='coerce')
    s_end = pd.to_datetime(df['snapshot_end_date'], errors='coerce')
    snap_dur = (s_end - s_start).dt.days.fillna(2).clip(lower=0).astype(np.float32)
    out['snapshot_duration'] = snap_dur
    out['log_snapshot_duration'] = np.log1p(snap_dur).astype(np.float32)

    last_rev = pd.to_datetime(df['last_review_date'], errors='coerce')
    first_rev = pd.to_datetime(df['first_review_date'], errors='coerce')
    days_since_last = (s_start - last_rev).dt.days.fillna(9999).clip(lower=0).astype(np.float32)
    days_since_first = (s_start - first_rev).dt.days.fillna(9999).clip(lower=0).astype(np.float32)
    out['days_since_last_review'] = days_since_last
    out['days_since_first_review'] = days_since_first
    out['has_recent_review_90d'] = (days_since_last <= 90).astype(np.int8)
    out['review_acceleration'] = (df['review_count_30d'] / (df['review_count_12m'].fillna(0)/12.0 + 1.0)).fillna(0).astype(np.float32)

    # Forward Occupancy Dynamics
    occ_30 = (1.0 - (df['open_slots_a'] / 30.0).clip(0, 1)).astype(np.float32)
    occ_60 = (1.0 - (df['open_slots_b'] / 60.0).clip(0, 1)).astype(np.float32)
    occ_90 = (1.0 - (df['open_slots_c'] / 90.0).clip(0, 1)).astype(np.float32)
    occ_365 = (1.0 - (df['open_slots_d'] / 365.0).clip(0, 1)).astype(np.float32)
    out['occupancy_30'] = occ_30
    out['occupancy_60'] = occ_60
    out['occupancy_90'] = occ_90
    out['occupancy_365'] = occ_365
    out['open_slots_e_ratio'] = (df['open_slots_e'] / 201.0).clip(0, 1).astype(np.float32)
    out['occupancy_delta_30_60'] = (occ_30 - occ_60).astype(np.float32)
    out['occupancy_delta_60_90'] = (occ_60 - occ_90).astype(np.float32)
    out['booked_load_ratio'] = (df['booked_load'] / 255.0).clip(0, 1).astype(np.float32)

    # Review value composite
    score_cols = ['score_overall', 'score_accuracy', 'score_condition', 'score_onboarding', 'score_support', 'score_access', 'score_value']
    out['score_mean'] = df[score_cols].mean(axis=1).astype(np.float32)
    out['value_gap'] = (df['score_value'] - df['score_overall']).fillna(0).astype(np.float32)

    # Space Class Keywords
    sc_str = df['space_class'].fillna('')
    for kw in SPACE_KEYWORDS:
        out[f'kw_{kw}'] = sc_str.str.contains(kw, regex=False).astype(np.int8)

    # Specific Facilities Presence
    fac_str = df['facilities'].fillna('')
    out['facility_count'] = fac_str.apply(lambda s: len(re.findall(r'FCL-\d+', s))).astype(np.int16)
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

X_train_full = extract_all_features(train_df, train_ref=train_df)
y_train_full = X_train_full.pop('price_tier').astype(int)
X_test_full = extract_all_features(test_df, train_ref=train_df)
test_ids = test_df['space_id'].values

# Attach Amenity Quality Features & Compound Value
for c in train_amenity.columns:
    X_train_full[c] = train_amenity[c]
    X_test_full[c] = test_amenity[c]

X_train_full['compound_value'] = (X_train_full['capacity_volume'] * X_train_full['amenity_quality_score']).astype(np.float32)
X_test_full['compound_value'] = (X_test_full['capacity_volume'] * X_test_full['amenity_quality_score']).astype(np.float32)

# 3. Robust Target Encodings
print("Computing Bayesian Target Encodings (Zone, Space Class, Metro, and Operator Ref)...")
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
te_cols = ['zone', 'space_class', 'metro']
global_mean = y_train_full.mean()

for c in te_cols:
    X_train_full[f'{c}_te_mean'] = 0.0
    X_test_full[f'{c}_te_mean'] = 0.0
X_train_full['op_te_mean'] = 0.0
X_test_full['op_te_mean'] = 0.0

for tr_idx, val_idx in kf.split(X_train_full, y_train_full):
    tr_y = y_train_full.iloc[tr_idx]
    for c in te_cols:
        tr_cat = train_df[c].iloc[tr_idx]
        val_cat = train_df[c].iloc[val_idx]
        stats = tr_y.groupby(tr_cat).agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + 15 * global_mean) / (stats['count'] + 15)
        mapping = smoothed.to_dict()
        X_train_full.loc[X_train_full.index[val_idx], f'{c}_te_mean'] = val_cat.map(mapping).fillna(global_mean)

    # Operator ref (m=25 smoothing)
    op_tr = train_df['operator_ref'].iloc[tr_idx]
    op_val = train_df['operator_ref'].iloc[val_idx]
    op_stats = tr_y.groupby(op_tr).agg(['count', 'mean'])
    op_smoothed = (op_stats['count'] * op_stats['mean'] + 25.0 * global_mean) / (op_stats['count'] + 25.0)
    op_map = op_smoothed.to_dict()
    X_train_full.loc[X_train_full.index[val_idx], 'op_te_mean'] = op_val.map(op_map).fillna(global_mean)

for c in te_cols:
    stats = y_train_full.groupby(train_df[c]).agg(['count', 'mean'])
    smoothed = (stats['count'] * stats['mean'] + 15 * global_mean) / (stats['count'] + 15)
    mapping = smoothed.to_dict()
    X_test_full[f'{c}_te_mean'] = test_df[c].map(mapping).fillna(global_mean)

op_stats = y_train_full.groupby(train_df['operator_ref']).agg(['count', 'mean'])
op_smoothed = (op_stats['count'] * op_stats['mean'] + 25.0 * global_mean) / (op_stats['count'] + 25.0)
X_test_full['op_te_mean'] = test_df['operator_ref'].map(op_smoothed.to_dict()).fillna(global_mean)

cat_cols = [c for c in X_train_full.columns if str(X_train_full[c].dtype) == 'category']
num_cols = [c for c in X_train_full.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols)
    ]
)

print("Pre-transforming matrices...")
X_train_p = preprocessor.fit_transform(X_train_full)
X_test_p = preprocessor.transform(X_test_full)

y_full_0 = (y_train_full - 1).values

# Stratified 80/20 Holdout
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_p, y_full_0, test_size=0.20, random_state=42, stratify=y_full_0
)

class_weights = {0: 1.02, 1: 0.95, 2: 1.15, 3: 0.98}
sample_weights_tr = np.array([class_weights[y] for y in y_tr])

print("\n---> Training LightGBM ...")
lgbm = lgb.LGBMClassifier(
    objective='multiclass', num_class=4, n_estimators=500, learning_rate=0.06,
    num_leaves=100, max_depth=12, subsample=0.85, colsample_bytree=0.80,
    min_child_samples=25, random_state=42, n_jobs=-1
)
lgbm.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
prob_lgb = lgbm.predict_proba(X_val)
print(f"LGBM Val Macro F1: {f1_score(y_val+1, np.argmax(prob_lgb, axis=1)+1, average='macro'):.4f}")

print("\n---> Training XGBoost ...")
xgb_clf = xgb.XGBClassifier(
    objective='multi:softprob', num_class=4, eval_metric='mlogloss', tree_method='hist',
    n_estimators=450, learning_rate=0.06, max_depth=8, subsample=0.85,
    colsample_bytree=0.80, min_child_weight=3, random_state=42, n_jobs=-1
)
xgb_clf.fit(X_tr, y_tr, sample_weight=sample_weights_tr)
prob_xgb = xgb_clf.predict_proba(X_val)
print(f"XGB Val Macro F1: {f1_score(y_val+1, np.argmax(prob_xgb, axis=1)+1, average='macro'):.4f}")

print("\n---> Training CatBoost ...")
cb_clf = CatBoostClassifier(
    iterations=500, learning_rate=0.08, depth=6, loss_function='MultiClass',
    l2_leaf_reg=4.0, auto_class_weights='Balanced', random_seed=42, thread_count=-1, verbose=0
)
cb_clf.fit(X_tr, y_tr)
prob_cb = cb_clf.predict_proba(X_val)
print(f"CatBoost Val Macro F1: {f1_score(y_val+1, np.argmax(prob_cb, axis=1)+1, average='macro'):.4f}")

prob_blend = 0.40 * prob_lgb + 0.35 * prob_xgb + 0.25 * prob_cb
raw_f1 = f1_score(y_val+1, np.argmax(prob_blend, axis=1)+1, average='macro')
print(f"\nRaw Tri-GBDT Val Macro F1: {raw_f1:.4f}")

# Target priors in validation set
val_priors = np.bincount(y_val) / len(y_val)
print("Validation True Priors:", val_priors.round(4))

# 1. Unconstrained Nelder-Mead
def loss_unconstrained(w):
    preds = np.argmax(prob_blend * w, axis=1)
    return -f1_score(y_val, preds, average='macro')

res_un = minimize(loss_unconstrained, [1.0, 1.0, 1.0, 1.0], method='Nelder-Mead', options={'maxiter': 300})
w_un = res_un.x / res_un.x[0]
pred_un = np.argmax(prob_blend * w_un, axis=1)
prop_un = np.bincount(pred_un) / len(pred_un)
print(f"\n[Unconstrained Nelder-Mead]")
print(f"Weights: {w_un.round(4)}")
print(f"Macro F1: {f1_score(y_val, pred_un, average='macro'):.4f}")
print(f"Predicted Proportions: {prop_un.round(4)}")

# 2. Prior-Regularized Optimization
def loss_regularized(w):
    preds = np.argmax(prob_blend * w, axis=1)
    f1 = f1_score(y_val, preds, average='macro')
    props = np.bincount(preds, minlength=4) / len(preds)
    penalty = 50.0 * np.sum((props - val_priors)**2)
    return -f1 + penalty

res_reg = minimize(loss_regularized, [1.0, 1.0, 1.0, 1.0], method='Nelder-Mead', options={'maxiter': 400})
w_reg = res_reg.x / res_reg.x[0]
pred_reg = np.argmax(prob_blend * w_reg, axis=1)
prop_reg = np.bincount(pred_reg) / len(pred_reg)
print(f"\n[Prior-Regularized Optimization]")
print(f"Weights: {w_reg.round(4)}")
print(f"Macro F1: {f1_score(y_val, pred_reg, average='macro'):.4f}")
print(f"Predicted Proportions: {prop_reg.round(4)}")
print("Per-class F1:", [round(x, 4) for x in f1_score(y_val, pred_reg, average=None)])

# 3. Predict on Test Set with Prior-Regularized Multipliers
print("\nPredicting on Test Set...")
test_p_lgb = lgbm.predict_proba(X_test_p)
test_p_xgb = xgb_clf.predict_proba(X_test_p)
test_p_cb = cb_clf.predict_proba(X_test_p)

test_blend = 0.40 * test_p_lgb + 0.35 * test_p_xgb + 0.25 * test_p_cb

# Test predicted proportions under regularized weights
test_preds_reg = np.argmax(test_blend * w_reg, axis=1) + 1
sub_reg = pd.DataFrame({'ID': test_ids, 'price_tier': test_preds_reg})
sub_reg.to_csv('data/submission_prior_regularized.csv', index=False)

s_698 = pd.read_csv('data/submission_true_breakthrough.csv')
agr = (sub_reg['price_tier'] == s_698['price_tier']).mean() * 100
print(f"\nAgreement with 0.69873: {agr:.2f}%")
print("Test Predicted Proportions:\n", (sub_reg['price_tier'].value_counts(normalize=True).sort_index()*100).round(2).to_dict())
