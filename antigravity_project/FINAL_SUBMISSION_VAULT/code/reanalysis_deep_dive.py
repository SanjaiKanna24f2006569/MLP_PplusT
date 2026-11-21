import pandas as pd
import numpy as np

print("Loading train and test...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

print("\n=== 1. OPERATOR OVERLAP ===")
train_ops = set(train['operator_ref'].dropna().unique())
test_ops = set(test['operator_ref'].dropna().unique())
overlap_ops = train_ops.intersection(test_ops)
print(f"Train unique operators: {len(train_ops):,}")
print(f"Test unique operators:  {len(test_ops):,}")
print(f"Overlap operators:      {len(overlap_ops):,} ({len(overlap_ops)/len(test_ops)*100:.2f}% of test ops)")
test_in_train = test['operator_ref'].isin(train_ops).mean() * 100
print(f"Test rows whose operator is in train: {test_in_train:.2f}%")

print("\n=== 2. ZONE OVERLAP ===")
train_zones = set(train['zone'].dropna().unique())
test_zones = set(test['zone'].dropna().unique())
overlap_zones = train_zones.intersection(test_zones)
print(f"Train unique zones: {len(train_zones):,}")
print(f"Test unique zones:  {len(test_zones):,}")
print(f"Overlap zones:      {len(overlap_zones):,} ({len(overlap_zones)/len(test_zones)*100:.2f}%)")
print(f"Test rows whose zone is in train: {test['zone'].isin(train_zones).mean()*100:.2f}%")

print("\n=== 3. SPACE_CLASS OVERLAP ===")
train_sc = set(train['space_class'].dropna().unique())
test_sc = set(test['space_class'].dropna().unique())
print(f"Test space_classes in train: {len(test_sc.intersection(train_sc))}/{len(test_sc)} ({test['space_class'].isin(train_sc).mean()*100:.2f}%)")

print("\n=== 4. EXACT / NEAR DUPLICATE CHECK ===")
# Check identical rows ignoring space_id, target, dates
cols_to_check = [c for c in test.columns if c not in ['space_id', 'snapshot_start_date', 'snapshot_end_date']]
print(f"Checking duplicates across {len(cols_to_check)} features...")
train_dups = train.duplicated(subset=cols_to_check).sum()
test_dups = test.duplicated(subset=cols_to_check).sum()
print(f"Train near-duplicates (same physical listing, different dates): {train_dups:,} ({train_dups/len(train)*100:.2f}%)")
print(f"Test near-duplicates: {test_dups:,} ({test_dups/len(test)*100:.2f}%)")

# Check train listings appearing in test
train_tuples = set(train[cols_to_check].itertuples(index=False, name=None))
test_matches = sum(1 for row in test[cols_to_check].itertuples(index=False, name=None) if row in train_tuples)
print(f"Test rows that have an IDENTICAL match in train: {test_matches:,} ({test_matches/len(test)*100:.2f}%)")

print("\n=== 5. CHECK LOCATION + OPERATOR MATCHES ===")
loc_cols = ['geo_lat', 'geo_lon', 'max_seats', 'operator_ref']
train_locs = set(train[loc_cols].itertuples(index=False, name=None))
test_loc_matches = sum(1 for row in test[loc_cols].itertuples(index=False, name=None) if row in train_locs)
print(f"Test rows matching (lat, lon, seats, operator) in train: {test_loc_matches:,} ({test_loc_matches/len(test)*100:.2f}%)")
