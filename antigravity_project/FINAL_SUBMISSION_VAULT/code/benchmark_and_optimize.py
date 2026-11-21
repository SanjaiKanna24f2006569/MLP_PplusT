import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
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
from catboost import CatBoostClassifier

print("=== BENCHMARK & MULTI-FEATURE OPTIMIZATION (TARGET: MACRO F1 > 0.72) ===")

t0 = time.time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Drop redundant duplicate columns
for df in [train_df, test_df]:
    df.drop(columns=['min_term_floor', 'max_term_ceiling'], errors='ignore', inplace=True)

# 1. Amenity Quality Weights
print("Extracting Amenity Quality Index across 144 facility codes...")
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

# 2. Extract Base Features & Novel Unfolded Signals
print("Extracting domain features and compound interactions...")
from src.feature_engineering import extract_features
X_train_full = extract_features(train_df, train_ref=train_df, is_train=True)
y_train_full = X_train_full.pop('price_tier').astype(int)
X_test_full = extract_features(test_df, train_ref=train_df, is_train=False)
test_ids = test_df['space_id'].values

# Attach Amenity Features
for c in train_amenity_df.columns:
    X_train_full[c] = train_amenity_df[c]
    X_test_full[c] = test_amenity_df[c]

# Novel Unfolded Signals
for df, X in [(train_df, X_train_full), (test_df, X_test_full)]:
    # Capacity Volume (r = 0.5122)
    cap_vol = np.log1p(df['max_seats'] * (df['zones'].fillna(1.0) + 1.0) * (df['utility_rooms'].fillna(1.0) + 1.0)).astype(np.float32)
    X['capacity_volume'] = cap_vol
    
    # Compound Value = Capacity Volume * Amenity Quality (r = 0.5321)
    X['compound_value'] = (cap_vol * X['amenity_quality_score']).astype(np.float32)
    
    # Seats * Workstations (r = 0.4927)
    X['seats_workstations_product'] = np.log1p(df['max_seats'] * (df['workstations'].fillna(1.0) + 1.0)).astype(np.float32)
    
    # Seats / Utility Rooms (r = 0.3820)
    X['seats_per_utility_room'] = (df['max_seats'] / (df['utility_rooms'].fillna(1.0) + 1.0)).astype(np.float32)
    
    # Operator Specialization Ratios (r = -0.3227 and +0.2704)
    units = df['operator_metro_units'].fillna(0) + 1.0
    X['op_private_ratio'] = (df['operator_metro_private'].fillna(0) / units).astype(np.float32)
    X['op_whole_ratio'] = (df['operator_metro_whole'].fillna(0) / units).astype(np.float32)
    X['op_shared_ratio'] = (df['operator_metro_shared'].fillna(0) / units).astype(np.float32)
    
    # Perceived Value Gap
    X['value_gap'] = (df['score_value'] - df['score_overall']).fillna(0).astype(np.float32)
    
    # Term Flexibility Ratio
    X['term_flexibility_ratio'] = (np.log1p(df['max_term_days'].clip(0, 1125)) - np.log1p(df['min_term_days'].clip(lower=0))).fillna(0).astype(np.float32)
    
    # Review Acceleration
    X['review_acceleration'] = (df['review_count_30d'] / (df['review_count_12m'].fillna(0)/12.0 + 1.0)).fillna(0).astype(np.float32)
    
    # Multi-Horizon Forward Occupancy Dynamics
    occ_30 = (1.0 - (df['open_slots_a'] / 30.0).clip(0, 1)).astype(np.float32)
    occ_60 = (1.0 - (df['open_slots_b'] / 60.0).clip(0, 1)).astype(np.float32)
    occ_90 = (1.0 - (df['open_slots_c'] / 90.0).clip(0, 1)).astype(np.float32)
    occ_365 = (1.0 - (df['open_slots_d'] / 365.0).clip(0, 1)).astype(np.float32)
    X['occupancy_30'] = occ_30
    X['occupancy_60'] = occ_60
    X['occupancy_90'] = occ_90
    X['occupancy_365'] = occ_365
    X['occupancy_delta_30_60'] = (occ_30 - occ_60).astype(np.float32)
    X['occupancy_delta_60_90'] = (occ_60 - occ_90).astype(np.float32)
    
    # Date & Recency Dynamics
    snap_date = pd.to_datetime(df['snapshot_start_date'], errors='coerce')
    last_rev = pd.to_datetime(df['last_review_date'], errors='coerce')
    first_rev = pd.to_datetime(df['first_review_date'], errors='coerce')
    days_since_last = (snap_date - last_rev).dt.days.fillna(9999).clip(lower=0).astype(np.float32)
    days_since_first = (snap_date - first_rev).dt.days.fillna(9999).clip(lower=0).astype(np.float32)
    X['days_since_last_review'] = days_since_last
    X['days_since_first_review'] = days_since_first
    X['has_recent_review_90d'] = (days_since_last <= 90).astype(np.int8)

print(f"Total features extracted: {X_train_full.shape[1]}")

# 3. Regional Bayesian Target Encoding (Zone, Space Class, Metro)
print("Computing Regional Bayesian Target Encoding...")
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
print("RUNNING FULL 5-FOLD STRATIFIED CV TO VERIFY OUT-OF-FOLD MACRO F1...")
print("="*70)

oof_lgb = np.zeros((len(train_df), 4), dtype=np.float32)
oof_xgb = np.zeros((len(train_df), 4), dtype=np.float32)
oof_cb = np.zeros((len(train_df), 4), dtype=np.float32)

test_p_lgb = np.zeros((len(test_df), 4), dtype=np.float32)
test_p_xgb = np.zeros((len(test_df), 4), dtype=np.float32)
test_p_cb = np.zeros((len(test_df), 4), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_p, y_full_0)):
    f_t0 = time.time()
    X_tr_f, y_tr_f = X_train_p[tr_idx], y_full_0[tr_idx]
    X_val_f, y_val_f = X_train_p[val_idx], y_full_0[val_idx]
    sw_tr = sample_weights_full[tr_idx]
    
    # 1. LightGBM
    clf_lgb = lgb.LGBMClassifier(
        objective='multiclass', num_class=4, n_estimators=550, learning_rate=0.06,
        num_leaves=100, max_depth=12, subsample=0.85, colsample_bytree=0.80,
        min_child_samples=25, random_state=42 + fold, n_jobs=-1
    )
    clf_lgb.fit(X_tr_f, y_tr_f, sample_weight=sw_tr)
    v_lgb = clf_lgb.predict_proba(X_val_f)
    oof_lgb[val_idx] = v_lgb
    test_p_lgb += clf_lgb.predict_proba(X_test_p) / 5.0
    
    # 2. XGBoost
    clf_xgb = xgb.XGBClassifier(
        objective='multi:softprob', num_class=4, eval_metric='mlogloss', tree_method='hist',
        n_estimators=500, learning_rate=0.06, max_depth=8, subsample=0.85,
        colsample_bytree=0.80, min_child_weight=3, random_state=42 + fold, n_jobs=-1
    )
    clf_xgb.fit(X_tr_f, y_tr_f, sample_weight=sw_tr)
    v_xgb = clf_xgb.predict_proba(X_val_f)
    oof_xgb[val_idx] = v_xgb
    test_p_xgb += clf_xgb.predict_proba(X_test_p) / 5.0
    
    # 3. CatBoost
    clf_cb = CatBoostClassifier(
        iterations=500, learning_rate=0.08, depth=6, loss_function='MultiClass',
        l2_leaf_reg=4.0, auto_class_weights='Balanced', random_seed=42 + fold,
        thread_count=-1, verbose=0
    )
    clf_cb.fit(X_tr_f, y_tr_f)
    v_cb = clf_cb.predict_proba(X_val_f)
    oof_cb[val_idx] = v_cb
    test_p_cb += clf_cb.predict_proba(X_test_p) / 5.0
    
    v_blend = 0.40 * v_lgb + 0.35 * v_xgb + 0.25 * v_cb
    f1_fold = f1_score(y_train_full[val_idx], np.argmax(v_blend, axis=1) + 1, average='macro')
    print(f"  Fold {fold+1} Tri-GBDT Macro F1: {f1_fold:.4f} (took {time.time()-f_t0:.1f}s)")

# Combined OOF
oof_blend = 0.40 * oof_lgb + 0.35 * oof_xgb + 0.25 * oof_cb
raw_oof_f1 = f1_score(y_train_full, np.argmax(oof_blend, axis=1) + 1, average='macro')
print(f"\n>>> Raw 5-Fold OOF Macro F1 Score: {raw_oof_f1:.4f} <<<")

# Multiplier optimization
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
print(f">>> FINAL OPTIMIZED 5-FOLD OOF MACRO F1 SCORE: {opt_oof_f1:.4f} <<<")
print(f">>> FINAL OPTIMIZED 5-FOLD OOF ACCURACY       : {opt_acc:.4f} <<<")
print("="*70)
print("Per-class F1:", [round(x, 4) for x in f1_score(y_train_full, opt_oof_preds, average=None)])
print("\nClassification Report:\n", classification_report(y_train_full, opt_oof_preds, target_names=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'], digits=4))

# 5. Predict on Test Set
test_blend = 0.40 * test_p_lgb + 0.35 * test_p_xgb + 0.25 * test_p_cb
test_scaled = test_blend * best_w
final_preds = np.argmax(test_scaled, axis=1) + 1

sub = pd.DataFrame({'ID': test_ids, 'price_tier': final_preds})
sub.to_csv('data/submission_optimized_72plus.csv', index=False)
sub.to_csv('data/submission.csv', index=False)
print("Saved data/submission_optimized_72plus.csv and updated data/submission.csv!")
print(sub.head(10))
print("\nPredicted Class Proportions:\n", (sub['price_tier'].value_counts(normalize=True).sort_index()*100).round(2).to_string())

s_698 = pd.read_csv('data/submission_true_breakthrough.csv')
agr = (sub['price_tier'] == s_698['price_tier']).mean() * 100
print(f"\nAgreement with 0.69873 Leaderboard Submission: {agr:.2f}%")
