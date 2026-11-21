import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

keys = [
    ['geo_lat', 'geo_lon', 'max_seats', 'operator_ref'],
    ['geo_lat', 'geo_lon', 'max_seats', 'space_class'],
    ['geo_lat', 'geo_lon', 'max_seats'],
    ['geo_lat', 'geo_lon', 'operator_ref'],
    ['geo_lat', 'geo_lon']
]

for key in keys:
    tr = train.dropna(subset=key)
    te = test.dropna(subset=key)
    
    # Mode price_tier per key and consistency
    stats = tr.groupby(key)['price_tier'].agg(
        count='count',
        nunique='nunique',
        mode_purity=lambda x: x.value_counts().iloc[0] / len(x)
    )
    
    # Check test matches
    te_match = te.merge(tr[key].drop_duplicates(), on=key, how='inner')
    
    print(f"Key: {key}")
    print(f"  Unique keys in train: {len(stats):,}")
    print(f"  Average mode purity: {stats['mode_purity'].mean()*100:.2f}%")
    print(f"  Keys with 100% purity: {(stats['nunique'] == 1).mean()*100:.2f}%")
    print(f"  Test rows matched: {len(te_match):,} / {len(test):,} ({len(te_match)/len(test)*100:.2f}%)")
    print("-" * 50)
