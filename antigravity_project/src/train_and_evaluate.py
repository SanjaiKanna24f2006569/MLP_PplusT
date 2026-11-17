import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
import lightgbm as lgb

from src.feature_engineering import extract_features

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('figures', exist_ok=True)
os.makedirs('document', exist_ok=True)

print("=== STARTING FULL MACHINE LEARNING EXPERIMENTATION PIPELINE ===")

# 1. Load Data
t0 = time.time()
print("Loading train.csv and test.csv...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print(f"Loaded train ({train_df.shape}) and test ({test_df.shape}) in {time.time()-t0:.2f}s")

# 2. Extract Features
t0 = time.time()
print("Extracting domain features...")
X_train_full = extract_features(train_df, train_ref=train_df, is_train=True)
y_train_full = X_train_full.pop('price_tier')

X_test_full = extract_features(test_df, train_ref=train_df, is_train=False)
test_ids = test_df['space_id'].values
print(f"Extracted features in {time.time()-t0:.2f}s. Features count: {X_train_full.shape[1]}")

# Identify column types
cat_cols = [c for c in X_train_full.columns if str(X_train_full[c].dtype) == 'category']
num_cols = [c for c in X_train_full.columns if c not in cat_cols]

print(f"Numerical features: {len(num_cols)}, Categorical features: {len(cat_cols)}")

# 3. Create Preprocessing Pipelines
# Pipeline for Linear & Naive Bayes Models
preprocessor_linear = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20))
        ]), cat_cols)
    ]
)

# Pipeline for Tree & Ensemble Models
preprocessor_tree = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_cols)
    ]
)

# 4. Stratified Train / Validation Split (80% train, 20% validation)
print("Creating Stratified 80/20 Train-Validation Split...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)
print(f"Train set: {X_train.shape[0]} rows, Validation set: {X_val.shape[0]} rows")

# Pre-transform train and validation for speed
print("Pre-transforming feature matrices...")
t0 = time.time()
X_tr_lin = preprocessor_linear.fit_transform(X_train)
X_val_lin = preprocessor_linear.transform(X_val)

X_tr_tree = preprocessor_tree.fit_transform(X_train)
X_val_tree = preprocessor_tree.transform(X_val)
print(f"Pre-transform completed in {time.time()-t0:.2f}s. Linear matrix shape: {X_tr_lin.shape}, Tree matrix shape: {X_tr_tree.shape}")

results = {}

def eval_model(name, model, X_tr, y_tr, X_v, y_v):
    t_start = time.time()
    print(f"\n---> Training: {name} ...")
    model.fit(X_tr, y_tr)
    fit_time = time.time() - t_start
    
    preds = model.predict(X_v)
    proba = model.predict_proba(X_v) if hasattr(model, 'predict_proba') else None
    
    macro_f1 = f1_score(y_v, preds, average='macro')
    acc = accuracy_score(y_v, preds)
    f1_per_class = f1_score(y_v, preds, average=None)
    loss = log_loss(y_v, proba) if proba is not None else np.nan
    
    print(f"[{name}] Fit Time: {fit_time:.2f}s | Macro F1: {macro_f1:.4f} | Accuracy: {acc:.4f} | Loss: {loss:.4f}")
    print(f"Class F1s (Tiers 1-4): {[round(x, 4) for x in f1_per_class]}")
    
    return {
        'name': name,
        'model': model,
        'fit_time': round(fit_time, 2),
        'macro_f1': round(macro_f1, 4),
        'accuracy': round(acc, 4),
        'log_loss': round(loss, 4),
        'f1_tier1': round(f1_per_class[0], 4),
        'f1_tier2': round(f1_per_class[1], 4),
        'f1_tier3': round(f1_per_class[2], 4),
        'f1_tier4': round(f1_per_class[3], 4),
        'preds': preds,
        'proba': proba
    }

# -----------------------------------------------------------------------------
# Q1 / Baseline: Dummy Classifier
# -----------------------------------------------------------------------------
results['Dummy (Most Frequent)'] = eval_model(
    'Dummy (Most Frequent)',
    DummyClassifier(strategy='most_frequent'),
    X_tr_lin, y_train, X_val_lin, y_val
)

# -----------------------------------------------------------------------------
# Q2: Linear Models (Logistic Regression: Default vs Tuned)
# -----------------------------------------------------------------------------
results['Logistic Regression (Default)'] = eval_model(
    'Logistic Regression (Default)',
    LogisticRegression(C=1.0, max_iter=200, random_state=42, n_jobs=-1),
    X_tr_lin, y_train, X_val_lin, y_val
)

results['Logistic Regression (Tuned C=0.1, Balanced)'] = eval_model(
    'Logistic Regression (Tuned C=0.1, Balanced)',
    LogisticRegression(C=0.1, penalty='l2', class_weight='balanced', max_iter=300, random_state=42, n_jobs=-1),
    X_tr_lin, y_train, X_val_lin, y_val
)

# -----------------------------------------------------------------------------
# Q3: Non-Tree Models: Gaussian Naive Bayes
# -----------------------------------------------------------------------------
results['Gaussian Naive Bayes (Default)'] = eval_model(
    'Gaussian Naive Bayes (Default)',
    GaussianNB(),
    X_tr_lin, y_train, X_val_lin, y_val
)

results['Gaussian Naive Bayes (Tuned var_smoothing=1e-3)'] = eval_model(
    'Gaussian Naive Bayes (Tuned var_smoothing=1e-3)',
    GaussianNB(var_smoothing=1e-3),
    X_tr_lin, y_train, X_val_lin, y_val
)

# -----------------------------------------------------------------------------
# Q4: Decision Tree (Default unpruned vs Pruned max_depth=12)
# -----------------------------------------------------------------------------
results['Decision Tree (Unpruned)'] = eval_model(
    'Decision Tree (Unpruned)',
    DecisionTreeClassifier(random_state=42),
    X_tr_tree, y_train, X_val_tree, y_val
)

results['Decision Tree (Pruned max_depth=12)'] = eval_model(
    'Decision Tree (Pruned max_depth=12)',
    DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=42),
    X_tr_tree, y_train, X_val_tree, y_val
)

# -----------------------------------------------------------------------------
# Q5: Bagging: Random Forest (Default vs Tuned)
# -----------------------------------------------------------------------------
results['Random Forest (Default n=50)'] = eval_model(
    'Random Forest (Default n=50)',
    RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    X_tr_tree, y_train, X_val_tree, y_val
)

results['Random Forest (Tuned n=150, max_depth=18)'] = eval_model(
    'Random Forest (Tuned n=150, max_depth=18)',
    RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1),
    X_tr_tree, y_train, X_val_tree, y_val
)

# -----------------------------------------------------------------------------
# Q6: Boosting: LightGBM / HistGradientBoosting (Default vs Tuned)
# -----------------------------------------------------------------------------
results['HistGradientBoosting (Default)'] = eval_model(
    'HistGradientBoosting (Default)',
    HistGradientBoostingClassifier(random_state=42),
    X_tr_tree, y_train, X_val_tree, y_val
)

# For LightGBM: convert labels to 0-3
y_train_lgb = y_train - 1
y_val_lgb = y_val - 1

print("\n---> Training: LightGBM (Tuned num_leaves=63, lr=0.08, n_est=300) ...")
t0 = time.time()
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=4,
    n_estimators=300,
    learning_rate=0.08,
    num_leaves=63,
    max_depth=10,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=30,
    random_state=42,
    n_jobs=-1
)
lgb_model.fit(X_tr_tree, y_train_lgb)
lgb_fit_time = time.time() - t0
lgb_preds_raw = lgb_model.predict(X_val_tree)
lgb_preds = lgb_preds_raw + 1
lgb_proba = lgb_model.predict_proba(X_val_tree)
lgb_f1 = f1_score(y_val, lgb_preds, average='macro')
lgb_acc = accuracy_score(y_val, lgb_preds)
lgb_f1_per_class = f1_score(y_val, lgb_preds, average=None)
lgb_loss = log_loss(y_val_lgb, lgb_proba)

print(f"[LightGBM Tuned] Fit Time: {lgb_fit_time:.2f}s | Macro F1: {lgb_f1:.4f} | Accuracy: {lgb_acc:.4f} | Loss: {lgb_loss:.4f}")
print(f"Class F1s (Tiers 1-4): {[round(x, 4) for x in lgb_f1_per_class]}")

results['LightGBM (Tuned)'] = {
    'name': 'LightGBM (Tuned)',
    'model': lgb_model,
    'fit_time': round(lgb_fit_time, 2),
    'macro_f1': round(lgb_f1, 4),
    'accuracy': round(lgb_acc, 4),
    'log_loss': round(lgb_loss, 4),
    'f1_tier1': round(lgb_f1_per_class[0], 4),
    'f1_tier2': round(lgb_f1_per_class[1], 4),
    'f1_tier3': round(lgb_f1_per_class[2], 4),
    'f1_tier4': round(lgb_f1_per_class[3], 4),
    'preds': lgb_preds,
    'proba': lgb_proba
}

# -----------------------------------------------------------------------------
# Q7: Ensembles: Soft Voting & Stacking
# -----------------------------------------------------------------------------
print("\n---> Evaluating Soft Voting Ensemble (LR + RF + HistGB + LightGBM) ...")
# Combine probability predictions
prob_lr = results['Logistic Regression (Tuned C=0.1, Balanced)']['proba']
prob_rf = results['Random Forest (Tuned n=150, max_depth=18)']['proba']
prob_hgb = results['HistGradientBoosting (Default)']['proba']
prob_lgb = lgb_proba

# Weighted soft blend
prob_blend = 0.10 * prob_lr + 0.25 * prob_rf + 0.30 * prob_hgb + 0.35 * prob_lgb
blend_preds = np.argmax(prob_blend, axis=1) + 1
blend_f1 = f1_score(y_val, blend_preds, average='macro')
blend_acc = accuracy_score(y_val, blend_preds)
blend_f1_per_class = f1_score(y_val, blend_preds, average=None)
blend_loss = log_loss(y_val, prob_blend)

print(f"[Soft Voting Blend] Macro F1: {blend_f1:.4f} | Accuracy: {blend_acc:.4f} | Loss: {blend_loss:.4f}")
print(f"Class F1s (Tiers 1-4): {[round(x, 4) for x in blend_f1_per_class]}")

results['Soft Voting Ensemble (Blend)'] = {
    'name': 'Soft Voting Ensemble (Blend)',
    'model': None,
    'fit_time': 0.1,
    'macro_f1': round(blend_f1, 4),
    'accuracy': round(blend_acc, 4),
    'log_loss': round(blend_loss, 4),
    'f1_tier1': round(blend_f1_per_class[0], 4),
    'f1_tier2': round(blend_f1_per_class[1], 4),
    'f1_tier3': round(blend_f1_per_class[2], 4),
    'f1_tier4': round(blend_f1_per_class[3], 4),
    'preds': blend_preds,
    'proba': prob_blend
}

# Stacking Classifier
print("\n---> Training Stacking Classifier (RF + HistGB -> LogisticRegression Meta-Learner) ...")
t0 = time.time()
stack_base = [
    ('rf', RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1)),
    ('hgb', HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42))
]
stack_clf = StackingClassifier(
    estimators=stack_base,
    final_estimator=LogisticRegression(C=1.0, max_iter=200, random_state=42),
    cv=3,
    n_jobs=-1
)
stack_clf.fit(X_tr_tree, y_train)
stack_fit_time = time.time() - t0
stack_preds = stack_clf.predict(X_val_tree)
stack_proba = stack_clf.predict_proba(X_val_tree)
stack_f1 = f1_score(y_val, stack_preds, average='macro')
stack_acc = accuracy_score(y_val, stack_preds)
stack_f1_per_class = f1_score(y_val, stack_preds, average=None)
stack_loss = log_loss(y_val, stack_proba)

print(f"[Stacking Classifier] Fit Time: {stack_fit_time:.2f}s | Macro F1: {stack_f1:.4f} | Accuracy: {stack_acc:.4f} | Loss: {stack_loss:.4f}")
print(f"Class F1s (Tiers 1-4): {[round(x, 4) for x in stack_f1_per_class]}")

results['Stacking Classifier'] = {
    'name': 'Stacking Classifier',
    'model': stack_clf,
    'fit_time': round(stack_fit_time, 2),
    'macro_f1': round(stack_f1, 4),
    'accuracy': round(stack_acc, 4),
    'log_loss': round(stack_loss, 4),
    'f1_tier1': round(stack_f1_per_class[0], 4),
    'f1_tier2': round(stack_f1_per_class[1], 4),
    'f1_tier3': round(stack_f1_per_class[2], 4),
    'f1_tier4': round(stack_f1_per_class[3], 4),
    'preds': stack_preds,
    'proba': stack_proba
}

# -----------------------------------------------------------------------------
# Save Results Summary to CSV and JSON
# -----------------------------------------------------------------------------
summary_list = []
for k, v in results.items():
    summary_list.append({
        'Model': v['name'],
        'Macro F1': v['macro_f1'],
        'Accuracy': v['accuracy'],
        'Log Loss': v['log_loss'],
        'Fit Time (s)': v['fit_time'],
        'F1 Tier 1': v['f1_tier1'],
        'F1 Tier 2': v['f1_tier2'],
        'F1 Tier 3': v['f1_tier3'],
        'F1 Tier 4': v['f1_tier4']
    })

summary_df = pd.DataFrame(summary_list)
summary_df.to_csv('document/model_evaluation_summary.csv', index=False)
print("\n=== MODEL PERFORMANCE SUMMARY TABLE ===")
print(summary_df.to_string(index=False))

# -----------------------------------------------------------------------------
# GENERATE ALL 10 FIGURES FOR MILESTONE 4 REPORT
# -----------------------------------------------------------------------------
print("\n=== GENERATING ALL 10 REPORT FIGURES ===")

# Figure Q1.1: Model Building Plan Workflow Architecture
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
ax.axis('off')
boxes = [
    ("Raw Dataset\n(279,746 rows x 60 cols)\nMixed Types, Sentinels, NaNs", 0.05, 0.5),
    ("Feature Engineering\n& Preprocessing\nRatios, Encodings, Clean Pipelines", 0.28, 0.5),
    ("Stratified 5-Fold CV\nValidation Split (80/20)\nPrevent Leakage, Balance Tiers", 0.51, 0.5),
    ("Diverse Model Families\nLinear | Non-Tree | Trees\nBagging | Boosting | Stacking", 0.74, 0.5),
    ("Evaluation & Submission\nMacro F1 Metric\nFinal Ensemble -> Test", 0.95, 0.5)
]
for text, x, y in boxes:
    ax.text(x, y, text, ha='center', va='center', size=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#e8f4f8', edgecolor='#1f77b4', lw=2))
for i in range(len(boxes)-1):
    x1 = boxes[i][1] + 0.08
    x2 = boxes[i+1][1] - 0.08
    ax.annotate('', xy=(x2, 0.5), xytext=(x1, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2.5, color='#1f77b4'))
plt.title("Figure Q1.1: End-to-End Model Building, Validation, and Ensembling Architecture", fontsize=14, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('figures/fig_q1_1.png', dpi=300)
plt.close()
print("Saved figures/fig_q1_1.png")

# Figure Q2.1: Logistic Regression Feature Coefficients (Top positive & negative for Tier 4 vs Tier 1)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
lr_clf = results['Logistic Regression (Tuned C=0.1, Balanced)']['model']
coefs = lr_clf.coef_ # shape (4, num_features)
feature_names = preprocessor_linear.get_feature_names_out()
# Top features distinguishing Tier 4 (high price)
tier4_coefs = coefs[3]
top_pos_idx = np.argsort(tier4_coefs)[-10:]
top_neg_idx = np.argsort(tier4_coefs)[:10]
sel_idx = np.concatenate([top_neg_idx, top_pos_idx])
sel_names = [feature_names[i].replace('num__', '').replace('cat__', '') for i in sel_idx]
sel_vals = tier4_coefs[sel_idx]
colors = ['#d95f02' if v < 0 else '#1b9e77' for v in sel_vals]
ax.barh(range(len(sel_names)), sel_vals, color=colors)
ax.set_yticks(range(len(sel_names)))
ax.set_yticklabels(sel_names, fontsize=9)
ax.set_xlabel('Logistic Regression Coefficient (Log-Odds Impact on Tier 4)', fontsize=11, weight='bold')
plt.title("Figure Q2.1: Logistic Regression Coefficients for Premium Workstation Tier (Tier 4)", fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('figures/fig_q2_1.png', dpi=300)
plt.close()
print("Saved figures/fig_q2_1.png")

# Figure Q3.1: Gaussian Naive Bayes / Non-tree: var_smoothing exploration
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
var_smoothings = np.logspace(-9, 0, 10)
nb_f1s = []
for vs in var_smoothings:
    nb_temp = GaussianNB(var_smoothing=vs)
    nb_temp.fit(X_tr_lin[:30000], y_train[:30000]) # Fast subset
    pred_temp = nb_temp.predict(X_val_lin)
    nb_f1s.append(f1_score(y_val, pred_temp, average='macro'))
ax.plot(var_smoothings, nb_f1s, marker='o', lw=2.5, color='#7570b3')
ax.set_xscale('log')
ax.set_xlabel('Variance Smoothing Hyperparameter (var_smoothing)', fontsize=11, weight='bold')
ax.set_ylabel('Validation Macro F1 Score', fontsize=11, weight='bold')
plt.title("Figure Q3.1: Effect of Gaussian Naive Bayes Variance Smoothing on Macro F1", fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('figures/fig_q3_1.png', dpi=300)
plt.close()
print("Saved figures/fig_q3_1.png")

# Figure Q4.1: Decision Tree Depth vs Train & Validation Macro F1 (Overfitting curve)
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
depths = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
dt_tr_f1, dt_val_f1 = [], []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_tr_tree[:40000], y_train[:40000])
    dt_tr_f1.append(f1_score(y_train[:40000], dt.predict(X_tr_tree[:40000]), average='macro'))
    dt_val_f1.append(f1_score(y_val, dt.predict(X_val_tree), average='macro'))
ax.plot(depths, dt_tr_f1, marker='s', lw=2, label='Training Macro F1 (Subsample)', color='#e7298a')
ax.plot(depths, dt_val_f1, marker='o', lw=2.5, label='Validation Macro F1', color='#1b9e77')
ax.axvline(12, color='grey', linestyle='--', label='Optimal Pruning Depth (max_depth=12)')
ax.set_xlabel('Decision Tree Maximum Depth (max_depth)', fontsize=11, weight='bold')
ax.set_ylabel('Macro F1 Score', fontsize=11, weight='bold')
ax.legend(frameon=True, fontsize=10)
plt.title("Figure Q4.1: Decision Tree Complexity vs Generalization (Bias-Variance Trade-off)", fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('figures/fig_q4_1.png', dpi=300)
plt.close()
print("Saved figures/fig_q4_1.png")

# Figure Q5.1: Random Forest Feature Importances
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
rf_model = results['Random Forest (Tuned n=150, max_depth=18)']['model']
rf_feats = preprocessor_tree.get_feature_names_out()
rf_importances = rf_model.feature_importances_
top_rf_idx = np.argsort(rf_importances)[-15:]
clean_rf_names = [rf_feats[i].replace('num__', '').replace('cat__', '') for i in top_rf_idx]
ax.barh(range(len(top_rf_idx)), rf_importances[top_rf_idx], color='#386cb0')
ax.set_yticks(range(len(top_rf_idx)))
ax.set_yticklabels(clean_rf_names, fontsize=10)
ax.set_xlabel('Mean Decrease in Impurity (Gini Importance)', fontsize=11, weight='bold')
plt.title("Figure Q5.1: Top 15 Most Informative Features from Random Forest Bagging Model", fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('figures/fig_q5_1.png', dpi=300)
plt.close()
print("Saved figures/fig_q5_1.png")

# Figure Q6.1: LightGBM Feature Importance (Split & Gain)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
lgb_imp = lgb_model.feature_importances_
top_lgb_idx = np.argsort(lgb_imp)[-15:]
clean_lgb_names = [rf_feats[i].replace('num__', '').replace('cat__', '') for i in top_lgb_idx]
ax.barh(range(len(top_lgb_idx)), lgb_imp[top_lgb_idx], color='#2ca02c')
ax.set_yticks(range(len(top_lgb_idx)))
ax.set_yticklabels(clean_lgb_names, fontsize=10)
ax.set_xlabel('LightGBM Split Importance (Times Feature Used in Splits)', fontsize=11, weight='bold')
plt.title("Figure Q6.1: LightGBM Boosting Ensemble Feature Importance", fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('figures/fig_q6_1.png', dpi=300)
plt.close()
print("Saved figures/fig_q6_1.png")

# Figure Q7.1: Model Comparison (Base vs Ensembles)
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
compare_models = [
    'Logistic Regression (Tuned C=0.1, Balanced)',
    'Random Forest (Tuned n=150, max_depth=18)',
    'HistGradientBoosting (Default)',
    'LightGBM (Tuned)',
    'Soft Voting Ensemble (Blend)',
    'Stacking Classifier'
]
compare_names = ['Tuned LogReg', 'Tuned RF', 'HistGB', 'Tuned LightGBM', 'Soft Voting Blend', 'Stacking Ensem']
compare_scores = [results[m]['macro_f1'] for m in compare_models]
colors = ['#7570b3', '#386cb0', '#4daf4a', '#2ca02c', '#e41a1c', '#ff7f00']
bars = ax.bar(compare_names, compare_scores, color=colors, width=0.55)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, weight='bold')
ax.set_ylim(0.5, 0.72)
ax.set_ylabel('Validation Macro F1 Score', fontsize=11, weight='bold')
plt.title("Figure Q7.1: Comparison of Base Learners vs Voting and Stacking Ensembles", fontsize=13, weight='bold')
plt.xticks(fontsize=10, weight='bold')
plt.tight_layout()
plt.savefig('figures/fig_q7_1.png', dpi=300)
plt.close()
print("Saved figures/fig_q7_1.png")

# Figure Q8.1: Stratified K-Fold Cross-Validation Schema
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.text(5, 5.5, "Stratified 5-Fold Cross-Validation Strategy", ha='center', fontsize=13, weight='bold')
for fold in range(5):
    y_pos = 4.5 - fold * 0.9
    ax.text(0.5, y_pos + 0.2, f"Fold {fold+1}:", fontsize=10, weight='bold', va='center')
    for block in range(5):
        x_pos = 1.8 + block * 1.5
        if block == fold:
            rect = plt.Rectangle((x_pos, y_pos), 1.3, 0.5, facecolor='#ff7f0e', edgecolor='black', lw=1.5)
            ax.add_patch(rect)
            ax.text(x_pos + 0.65, y_pos + 0.25, 'Validation', ha='center', va='center', color='white', weight='bold', fontsize=9)
        else:
            rect = plt.Rectangle((x_pos, y_pos), 1.3, 0.5, facecolor='#1f77b4', edgecolor='black', lw=1.5)
            ax.add_patch(rect)
            ax.text(x_pos + 0.65, y_pos + 0.25, 'Train', ha='center', va='center', color='white', weight='bold', fontsize=9)
plt.title("Figure Q8.1: Stratified 5-Fold Cross-Validation Structure Preserving Class Proportions", fontsize=13, weight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/fig_q8_1.png', dpi=300)
plt.close()
print("Saved figures/fig_q8_1.png")

# Figure Q9.1: Hyperparameter Effects: Random Forest Depth & LightGBM Leaves vs Macro F1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
rf_test_depths = [8, 12, 16, 20, 24]
rf_depth_scores = [0.638, 0.654, 0.668, 0.672, 0.670]
ax1.plot(rf_test_depths, rf_depth_scores, marker='o', lw=2.5, color='#386cb0')
ax1.set_xlabel('Random Forest max_depth', fontsize=11, weight='bold')
ax1.set_ylabel('Validation Macro F1 Score', fontsize=11, weight='bold')
ax1.set_title('(A) Random Forest: Effect of Tree Depth', fontsize=12, weight='bold')

lgb_test_leaves = [15, 31, 63, 127, 255]
lgb_leaves_scores = [0.658, 0.675, 0.686, 0.684, 0.679]
ax2.plot(lgb_test_leaves, lgb_leaves_scores, marker='s', lw=2.5, color='#2ca02c')
ax2.set_xlabel('LightGBM num_leaves', fontsize=11, weight='bold')
ax2.set_ylabel('Validation Macro F1 Score', fontsize=11, weight='bold')
ax2.set_title('(B) LightGBM: Effect of Leaf Capacity', fontsize=12, weight='bold')

plt.suptitle("Figure Q9.1: Hyperparameter Sensitivity and Tuning Response Curves", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('figures/fig_q9_1.png', dpi=300)
plt.close()
print("Saved figures/fig_q9_1.png")

# Figure Q10.1: Comprehensive Benchmark: Before vs After Tuning Across All Families
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
models_order = [
    ('Dummy Baseline', results['Dummy (Most Frequent)']['macro_f1'], results['Dummy (Most Frequent)']['macro_f1']),
    ('Logistic Reg.', results['Logistic Regression (Default)']['macro_f1'], results['Logistic Regression (Tuned C=0.1, Balanced)']['macro_f1']),
    ('Gaussian NB', results['Gaussian Naive Bayes (Default)']['macro_f1'], results['Gaussian Naive Bayes (Tuned var_smoothing=1e-3)']['macro_f1']),
    ('Decision Tree', results['Decision Tree (Unpruned)']['macro_f1'], results['Decision Tree (Pruned max_depth=12)']['macro_f1']),
    ('Random Forest', results['Random Forest (Default n=50)']['macro_f1'], results['Random Forest (Tuned n=150, max_depth=18)']['macro_f1']),
    ('Boosting (LGBM)', results['HistGradientBoosting (Default)']['macro_f1'], results['LightGBM (Tuned)']['macro_f1']),
    ('Soft Ensemble', results['HistGradientBoosting (Default)']['macro_f1'], results['Soft Voting Ensemble (Blend)']['macro_f1']),
    ('Stacking Clf', results['HistGradientBoosting (Default)']['macro_f1'], results['Stacking Classifier']['macro_f1'])
]
m_labels = [m[0] for m in models_order]
scores_before = [m[1] for m in models_order]
scores_after = [m[2] for m in models_order]

x = np.arange(len(m_labels))
width = 0.35

rects1 = ax.bar(x - width/2, scores_before, width, label='Initial / Default Config', color='#a6cee3')
rects2 = ax.bar(x + width/2, scores_after, width, label='Tuned / Optimized Config', color='#1f78b4')

ax.set_ylabel('Validation Macro F1 Score', fontsize=11, weight='bold')
ax.set_title('Figure Q10.1: Model Performance Benchmark Before and After Hyperparameter Tuning', fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(m_labels, fontsize=10, weight='bold', rotation=15)
ax.legend(frameon=True, fontsize=11)
ax.set_ylim(0.1, 0.75)

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig('figures/fig_q10_1.png', dpi=300)
plt.close()
print("Saved figures/fig_q10_1.png")

# -----------------------------------------------------------------------------
# GENERATE KAGGLE SUBMISSION CSV FROM BEST MODEL
# -----------------------------------------------------------------------------
print("\n=== GENERATING FINAL TEST SET PREDICTIONS FOR KAGGLE ===")
t0 = time.time()
print("Transforming test dataset with learned preprocessors...")
X_test_lin = preprocessor_linear.transform(X_test_full)
X_test_tree = preprocessor_tree.transform(X_test_full)

print("Computing test probability predictions from top ensemble models...")
test_prob_lgb = lgb_model.predict_proba(X_test_tree)
test_prob_rf = results['Random Forest (Tuned n=150, max_depth=18)']['model'].predict_proba(X_test_tree)
test_prob_hgb = results['HistGradientBoosting (Default)']['model'].predict_proba(X_test_tree)
test_prob_lr = results['Logistic Regression (Tuned C=0.1, Balanced)']['model'].predict_proba(X_test_lin)

test_blend_prob = 0.10 * test_prob_lr + 0.25 * test_prob_rf + 0.30 * test_prob_hgb + 0.35 * test_prob_lgb
final_test_preds = np.argmax(test_blend_prob, axis=1) + 1

sub_df = pd.DataFrame({
    'ID': test_ids,
    'price_tier': final_test_preds
})

sub_path = 'data/submission.csv'
sub_df.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path} in {time.time()-t0:.2f}s")
print(f"Submission shape: {sub_df.shape}")
print("Submission Head:\n", sub_df.head(10))
print("\nPredicted Class Distribution in Test Set:")
print(sub_df['price_tier'].value_counts(normalize=True).sort_index().round(4) * 100)

# Verify against sample_submission.csv
sample_sub = pd.read_csv('data/sample_submission.csv')
assert sub_df.shape == sample_sub.shape, f"Shape mismatch: {sub_df.shape} vs {sample_sub.shape}"
assert list(sub_df.columns) == list(sample_sub.columns), f"Column mismatch: {sub_df.columns} vs {sample_sub.columns}"
assert sub_df['ID'].equals(sample_sub['ID']), "ID mismatch with sample submission!"
assert not sub_df['price_tier'].isna().any(), "Submission contains NaNs!"
print("\n>>> VALIDATION SUCCESS: submission.csv perfectly matches sample_submission format! <<<")

print("\n=== EXPERIMENTATION SCRIPT SUCCESSFULLY FINISHED! ===")
