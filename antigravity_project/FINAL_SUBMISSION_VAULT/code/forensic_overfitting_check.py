import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score

print("=== FORENSIC DISCREPANCY ANALYSIS ===")
s_opt = pd.read_csv('data/submission_optimized_te.csv') # Scored 0.69639 on Kaggle
s_new = pd.read_csv('data/submission_breakthrough_75.csv') # Scored 0.67955 on Kaggle

print(f"Total test rows: {len(s_opt):,}")
disagreements = (s_opt['price_tier'] != s_new['price_tier']).sum()
print(f"Number of disagreements: {disagreements:,} ({disagreements/len(s_opt)*100:.2f}%)")

print("\nConfusion Matrix between 0.69639 (rows) and 0.67955 (cols):")
ct = pd.crosstab(s_opt['price_tier'], s_new['price_tier'], rownames=['0.69639'], colnames=['0.67955'])
print(ct)

# Check which classes shifted the most
print("\nClass Distribution Comparison:")
df_comp = pd.DataFrame({
    '0.69639 (Opt)': s_opt['price_tier'].value_counts(normalize=True).sort_index() * 100,
    '0.67955 (New)': s_new['price_tier'].value_counts(normalize=True).sort_index() * 100
})
df_comp['Diff (%)'] = df_comp['0.67955 (New)'] - df_comp['0.69639 (Opt)']
print(df_comp.round(2))

# Load train and test to see what kind of samples shifted
train = pd.read_csv('data/train.csv', nrows=1000)
test = pd.read_csv('data/test.csv')

# Let's inspect test instances where s_opt and s_new disagree
diff_idx = s_opt['price_tier'] != s_new['price_tier']
diff_test = test.loc[diff_idx]

print("\nDisagreement samples characteristics:")
print("Missing geo_lat in diff:", diff_test['geo_lat'].isna().mean()*100)
print("Missing borough in diff:", diff_test['borough'].isna().mean()*100)
print("Top zones in diff:", diff_test['zone'].value_counts().head(5).to_dict())
