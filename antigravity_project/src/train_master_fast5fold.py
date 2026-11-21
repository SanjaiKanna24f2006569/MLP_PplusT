import os
import sys
import time
import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

print("=== FAST & ULTRA-ACCURATE 5-FOLD STRATIFIED MASTER PIPELINE ===")
t0 = time.time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Drop redundant duplicate columns
for df in [train_df, test_df]:
    df.drop(columns=['min_term_floor', 'max_term_ceiling'], errors='ignore', inplace=True)

# 1. Amenity Quality Index across 144 facility codes
print("[1/5] Extracting Amenity Quality Index across 144 facility codes...")
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
print("[2/5] Extracting domain signals, snapshot chronometry, and unfolded interactions...")
TOP_FACILITIES = ['FCL-069', 'FCL-089', 'FCL-120', 'FCL-011', 'FCL-007', 'FCL-055', 'FCL-071', 'FCL-097', 'FCL-035', 'FCL-098']
SPACE_KEYWORDS = ['Studio', 'Workroom', 'Flex', 'Boutique', 'Premium', 'Bay', 'Ground', 'Suite', 'Loft', 'Niche', 'Alcove']

def extract_all_features(df, train_ref=None):
    data = {}
    if 'price_tier' in df.columns:
        data['price_tier'] = df['price_tier'].astype(np.int8)

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
            data[c] = df[c].astype(np.float32)

    # Structured missingness
    data['is_missing_borough'] = df['borough'].isna().astype(np.int8)
    data['is_missing_reviews'] = df['score_overall'].isna().astype(np.int8)
    data['is_missing_zones'] = df['zones'].isna().astype(np.int8)
    data['is_missing_workstations'] = df['workstations'].isna().astype(np.int8)
    data['is_missing_utility_rooms'] = df['utility_rooms'].isna().astype(np.int8)

    # Sentinel extreme capping
    data['max_term_is_extreme'] = (df['max_term_days'] > 1125).fillna(False).astype(np.int8)
    data['log_max_term_days'] = np.log1p(df['max_term_days'].clip(0, 1125)).astype(np.float32)
    data['log_min_term_days'] = np.log1p(df['min_term_days'].clip(lower=0)).astype(np.float32)
    data['log_portfolio_size'] = np.log1p(df['operator_portfolio_size'].clip(lower=0)).astype(np.float32)
    data['term_flexibility_ratio'] = (data['log_max_term_days'] - data['log_min_term_days']).astype(np.float32)

    # Capacity interactions & volume
    cap_vol = np.log1p(df['max_seats'] * (df['zones'].fillna(1.0) + 1.0) * (df['utility_rooms'].fillna(1.0) + 1.0)).astype(np.float32)
    data['capacity_volume'] = cap_vol
    data['seats_workstations_product'] = np.log1p(df['max_seats'] * (df['workstations'].fillna(1.0) + 1.0)).astype(np.float32)
    data['seats_per_utility_room'] = (df['max_seats'] / (df['utility_rooms'].fillna(1.0) + 1.0)).astype(np.float32)
    data['seats_per_workstation'] = (df['max_seats'] / (df['workstations'].fillna(2.0) + 1.0)).astype(np.float32)
    data['seats_per_zone'] = (df['max_seats'] / (df['zones'].fillna(1.0) + 1.0)).astype(np.float32)
    data['workstations_per_zone'] = (df['workstations'].fillna(2.0) / (df['zones'].fillna(1.0) + 1.0)).astype(np.float32)
    data['utility_rooms_per_seat'] = (df['utility_rooms'].fillna(1.0) / (df['max_seats'] + 1.0)).astype(np.float32)

    # Operator specialization & chronometry
    plat_age = df['operator_platform_age_years'].fillna(0)*12 + df['operator_platform_age_months'].fillna(0)
    act_age = df['operator_active_years'].fillna(0)*12 + df['operator_active_months'].fillna(0)
    data['total_platform_age_months'] = plat_age.astype(np.float32)
    data['operator_inactivity_months'] = (plat_age - act_age).clip(lower=0).astype(np.float32)

    op_units = df['operator_metro_units'].fillna(0) + 1.0
    data['op_private_ratio'] = (df['operator_metro_private'].fillna(0) / op_units).astype(np.float32)
    data['op_whole_ratio'] = (df['operator_metro_whole'].fillna(0) / op_units).astype(np.float32)
    data['op_shared_ratio'] = (df['operator_metro_shared'].fillna(0) / op_units).astype(np.float32)
    data['op_whole_to_private_ratio'] = (df['operator_metro_whole'].fillna(0) / (df['operator_metro_private'].fillna(0) + 1.0)).astype(np.float32)

    # Temporal & Snapshot Dynamics (UNVEILED SIGNAL r = -0.21)
    s_start = pd.to_datetime(df['snapshot_start_date'], errors='coerce')
    s_end = pd.to_datetime(df['snapshot_end_date'], errors='coerce')
    snap_dur = (s_end - s_start).dt.days.fillna(2).clip(lower=0).astype(np.float32)
    data['snapshot_duration'] = snap_dur
    data['log_snapshot_duration'] = np.log1p(snap_dur).astype(np.float32)

    last_rev = pd.to_datetime(df['last_review_date'], errors='coerce')
    first_rev = pd.to_datetime(df['first_review_date'], errors='coerce')
    days_since_last = (s_start - last_rev).dt.days.fillna(9999).clip(lower=0).astype(np.float32)
    days_since_first = (s_start - first_rev).dt.days.fillna(9999).clip(lower=0).astype(np.float32)
    data['days_since_last_review'] = days_since_last
    data['days_since_first_review'] = days_since_first
    data['has_recent_review_90d'] = (days_since_last <= 90).astype(np.int8)
    data['review_acceleration'] = (df['review_count_30d'] / (df['review_count_12m'].fillna(0)/12.0 + 1.0)).fillna(0).astype(np.float32)

    # Forward Occupancy Dynamics
    occ_30 = (1.0 - (df['open_slots_a'] / 30.0).clip(0, 1)).astype(np.float32)
    occ_60 = (1.0 - (df['open_slots_b'] / 60.0).clip(0, 1)).astype(np.float32)
    occ_90 = (1.0 - (df['open_slots_c'] / 90.0).clip(0, 1)).astype(np.float32)
    occ_365 = (1.0 - (df['open_slots_d'] / 365.0).clip(0, 1)).astype(np.float32)
    data['occupancy_30'] = occ_30
    data['occupancy_60'] = occ_60
    data['occupancy_90'] = occ_90
    data['occupancy_365'] = occ_365
    data['open_slots_e_ratio'] = (df['open_slots_e'] / 201.0).clip(0, 1).astype(np.float32)
    data['occupancy_delta_30_60'] = (occ_30 - occ_60).astype(np.float32)
    data['occupancy_delta_60_90'] = (occ_60 - occ_90).astype(np.float32)
    data['booked_load_ratio'] = (df['booked_load'] / 255.0).clip(0, 1).astype(np.float32)

    # Review value composite
    score_cols = ['score_overall', 'score_accuracy', 'score_condition', 'score_onboarding', 'score_support', 'score_access', 'score_value']
    data['score_mean'] = df[score_cols].mean(axis=1).astype(np.float32)
    data['value_gap'] = (df['score_value'] - df['score_overall']).fillna(0).astype(np.float32)

    # Space Class Keywords (UNVEILED SEMANTIC SIGNALS)
    sc_str = df['space_class'].fillna('')
    for kw in SPACE_KEYWORDS:
        data[f'kw_{kw}'] = sc_str.str.contains(kw, regex=False).astype(np.int8)

    # Facilities Presence
    fac_str = df['facilities'].fillna('')
    data['facility_count'] = fac_str.apply(lambda s: len(re.findall(r'FCL-\d+', s))).astype(np.int16)
    for fcl in TOP_FACILITIES:
        data[f'has_{fcl}'] = fac_str.str.contains(fcl, regex=False).astype(np.int8)

    # Categoricals
    for c in ['operator_class', 'access_type', 'compliance_status', 'metro']:
        data[c] = df[c].fillna('missing').astype('category')

    for c in ['operator_certified', 'operator_id_verified']:
        data[c] = (df[c] == 't').astype(np.int8)

    source_df = train_ref if train_ref is not None else df
    for c in ['space_class', 'zone', 'borough']:
        freq_map = source_df[c].value_counts(normalize=True).to_dict()
        data[f'{c}_freq'] = df[c].map(freq_map).fillna(0).astype(np.float32)
        data[c] = df[c].fillna('missing').astype('category')

    return pd.DataFrame(data, index=df.index)

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

# 3. Robust Target Encodings: Zone, Space Class, Metro AND Operator Ref
print("[3/5] Computing Bayesian Target Encodings (Zone, Space Class, Metro, and Operator Ref)...")
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

print(f"Total features extracted: {X_train_full.shape[1]}")

# Preprocessing
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
X_train_p = preprocessor.fit_transform(X_train_full)
X_test_p = preprocessor.transform(X_test_full)

y_full_0 = (y_train_full - 1).values
class_weights = {0: 1.02, 1: 0.95, 2: 1.15, 3: 0.98}
sample_weights_full = np.array([class_weights[y] for y in y_full_0])

# 4. Run 5-Fold Stratified Cross Validation on Full Dataset
print("\n" + "="*70)
print("[4/5] RUNNING SPEED-OPTIMIZED 5-FOLD STRATIFIED CV ON 100% DATASET...")
print("="*70)

oof_lgb = np.zeros((len(train_df), 4), dtype=np.float32)
oof_xgb = np.zeros((len(train_df), 4), dtype=np.float32)

test_p_lgb = np.zeros((len(test_df), 4), dtype=np.float32)
test_p_xgb = np.zeros((len(test_df), 4), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_p, y_full_0)):
    f_t0 = time.time()
    X_tr_f, y_tr_f = X_train_p[tr_idx], y_full_0[tr_idx]
    X_val_f, y_val_f = X_train_p[val_idx], y_full_0[val_idx]
    sw_tr = sample_weights_full[tr_idx]

    # 1. LightGBM (500 trees, num_leaves=90)
    clf_lgb = lgb.LGBMClassifier(
        objective='multiclass', num_class=4, n_estimators=500, learning_rate=0.07,
        num_leaves=90, max_depth=11, subsample=0.85, colsample_bytree=0.80,
        min_child_samples=30, random_state=42 + fold, n_jobs=-1
    )
    clf_lgb.fit(X_tr_f, y_tr_f, sample_weight=sw_tr)
    v_lgb = clf_lgb.predict_proba(X_val_f)
    oof_lgb[val_idx] = v_lgb
    test_p_lgb += clf_lgb.predict_proba(X_test_p) / 5.0

    # 2. XGBoost (Fast hist, max_depth=6, 320 trees)
    clf_xgb = xgb.XGBClassifier(
        objective='multi:softprob', num_class=4, eval_metric='mlogloss', tree_method='hist',
        n_estimators=320, learning_rate=0.08, max_depth=6, subsample=0.85,
        colsample_bytree=0.80, min_child_weight=4, random_state=42 + fold, n_jobs=-1
    )
    clf_xgb.fit(X_tr_f, y_tr_f, sample_weight=sw_tr)
    v_xgb = clf_xgb.predict_proba(X_val_f)
    oof_xgb[val_idx] = v_xgb
    test_p_xgb += clf_xgb.predict_proba(X_test_p) / 5.0

    v_blend = 0.55 * v_lgb + 0.45 * v_xgb
    f1_fold = f1_score(y_train_full.iloc[val_idx], np.argmax(v_blend, axis=1) + 1, average='macro')
    print(f"  Fold {fold+1}/5 Blend Macro F1: {f1_fold:.4f} (took {time.time()-f_t0:.1f}s)")

# Combined Out-of-Fold Evaluation
oof_blend = 0.55 * oof_lgb + 0.45 * oof_xgb
raw_oof_f1 = f1_score(y_train_full, np.argmax(oof_blend, axis=1) + 1, average='macro')
print(f"\n>>> Raw 5-Fold OOF Macro F1: {raw_oof_f1:.4f} <<<")

# Multiplier optimization
print("\n[5/5] Optimizing decision thresholds via Nelder-Mead simplex search...")
def loss_func(weights):
    scaled = oof_blend * weights
    preds = np.argmax(scaled, axis=1) + 1
    return -f1_score(y_train_full, preds, average='macro')

res = minimize(loss_func, [1.0, 1.0, 1.0, 1.0], method='Nelder-Mead', options={'maxiter': 350})
best_w = res.x / res.x[0]
print(f"Optimal Decision Multipliers: {[round(w, 4) for w in best_w]}")

opt_oof_preds = np.argmax(oof_blend * best_w, axis=1) + 1
opt_oof_f1 = f1_score(y_train_full, opt_oof_preds, average='macro')
opt_acc = accuracy_score(y_train_full, opt_oof_preds)

print("\n" + "="*70)
print(f">>> FINAL 5-FOLD OUT-OF-FOLD MACRO F1 SCORE: {opt_oof_f1:.4f} <<<")
print(f">>> FINAL 5-FOLD OUT-OF-FOLD ACCURACY       : {opt_acc:.4f} <<<")
print("="*70)
print("Per-class F1:", [round(x, 4) for x in f1_score(y_train_full, opt_oof_preds, average=None)])
print("\nClassification Report:\n", classification_report(y_train_full, opt_oof_preds, target_names=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'], digits=4))

# Predict on Test Set
test_blend = 0.55 * test_p_lgb + 0.45 * test_p_xgb
test_scaled = test_blend * best_w
final_preds = np.argmax(test_scaled, axis=1) + 1

sub = pd.DataFrame({'ID': test_ids, 'price_tier': final_preds})
sub.to_csv('data/submission_master_breakthrough.csv', index=False)
sub.to_csv('data/submission.csv', index=False)
print("\nGenerated and saved data/submission_master_breakthrough.csv & data/submission.csv!")
print(sub.head(10))
print("\nPredicted Class Proportions:\n", (sub['price_tier'].value_counts(normalize=True).sort_index()*100).round(2).to_string())

if os.path.exists('data/submission_true_breakthrough.csv'):
    s_698 = pd.read_csv('data/submission_true_breakthrough.csv')
    agr = (sub['price_tier'] == s_698['price_tier']).mean() * 100
    print(f"\nAgreement with 0.69873 Leaderboard Submission: {agr:.2f}%")
print(f"Total Master Pipeline Runtime: {time.time()-t0:.1f}s")
