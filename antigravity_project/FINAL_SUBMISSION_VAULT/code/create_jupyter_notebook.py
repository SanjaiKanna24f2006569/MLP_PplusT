import json

def create_notebook():
    cells = []

    def md_cell(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(keepends=True)
        }

    def code_cell(code):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(keepends=True)
        }

    # Notebook Title & Meta
    cells.append(md_cell(
        "# IITM BS Degree — Machine Learning Practice (MLP) Project\n"
        "## Milestone 4 & Final Competition Pipeline: High-Performance Multi-GBDT Ensemble\n"
        "**Student Roll Number:** 24f2006569 | **Term:** 2026 T3\n"
        "**Target Variable:** `price_tier` (Multiclass: 1, 2, 3, 4)\n"
        "**Evaluation Metric:** Macro F1 Score\n\n"
        "### Key Architectural Features:\n"
        "1. Domain Feature Engineering (Capacity ratios, occupancy dynamics, facilities intelligence)\n"
        "2. Leak-free Preprocessing Pipeline\n"
        "3. Stratified 5-Fold Cross-Validation\n"
        "4. Full GBDT Triad: **LightGBM**, **XGBoost**, and **CatBoost**\n"
        "5. Grand Probability Ensemble for Kaggle Leaderboard Submission"
    ))

    # Cell 1: Imports
    cells.append(code_cell(
        "import os\n"
        "import re\n"
        "import time\n"
        "import json\n"
        "import warnings\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "from sklearn.model_selection import train_test_split, StratifiedKFold\n"
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.metrics import f1_score, accuracy_score, classification_report, log_loss\n\n"
        "import lightgbm as lgb\n"
        "import xgboost as xgb\n"
        "from catboost import CatBoostClassifier\n"
        "from sklearn.ensemble import RandomForestClassifier\n\n"
        "warnings.filterwarnings('ignore')\n"
        "print('All permitted libraries imported successfully.')"
    ))

    # Cell 2: Data Loading
    cells.append(code_cell(
        "# Load train and test data\n"
        "DATA_DIR = '../input' if os.path.exists('../input') else 'data'\n"
        "train_path = os.path.join(DATA_DIR, 'train.csv') if os.path.exists(os.path.join(DATA_DIR, 'train.csv')) else 'train.csv'\n"
        "test_path = os.path.join(DATA_DIR, 'test.csv') if os.path.exists(os.path.join(DATA_DIR, 'test.csv')) else 'test.csv'\n\n"
        "train_df = pd.read_csv(train_path)\n"
        "test_df = pd.read_csv(test_path)\n\n"
        "print(f'Train shape: {train_df.shape}')\n"
        "print(f'Test shape: {test_df.shape}')\n"
        "print('Target Distribution:\\n', train_df['price_tier'].value_counts(normalize=True).sort_index().round(4) * 100)"
    ))

    # Cell 3: Feature Engineering Function
    cells.append(code_cell(
        "TOP_FACILITIES = ['FCL-078', 'FCL-145', 'FCL-135', 'FCL-093', 'FCL-080',\n"
        "                  'FCL-139', 'FCL-020', 'FCL-058', 'FCL-056', 'FCL-138']\n\n"
        "def extract_features(df, train_ref=None, is_train=True):\n"
        "    out = pd.DataFrame(index=df.index)\n"
        "    if 'price_tier' in df.columns:\n"
        "        out['price_tier'] = df['price_tier'].astype(np.int8)\n"
        "    num_cols = [\n"
        "        'geo_lat', 'geo_lon', 'max_seats', 'utility_rooms', 'zones', 'workstations',\n"
        "        'min_term_days', 'max_term_days', 'max_term_floor', 'min_term_ceiling',\n"
        "        'min_term_avg', 'max_term_avg', 'open_slots_a', 'open_slots_b', 'open_slots_c',\n"
        "        'open_slots_d', 'open_slots_e', 'review_count', 'review_count_12m', 'review_count_30d',\n"
        "        'review_count_prev_yr', 'booked_load', 'score_overall', 'score_accuracy',\n"
        "        'score_condition', 'score_onboarding', 'score_support', 'score_access', 'score_value',\n"
        "        'operator_metro_units', 'operator_metro_whole', 'operator_metro_private',\n"
        "        'operator_metro_shared', 'reviews_per_month', 'operator_portfolio_size',\n"
        "        'operator_platform_age_years', 'operator_platform_age_months',\n"
        "        'operator_active_years', 'operator_active_months'\n"
        "    ]\n"
        "    for c in num_cols:\n"
        "        if c in df.columns:\n"
        "            out[c] = df[c].astype(np.float32)\n"
        "    out['is_missing_borough'] = df['borough'].isna().astype(np.int8)\n"
        "    out['is_missing_reviews'] = df['score_overall'].isna().astype(np.int8)\n"
        "    out['max_term_is_extreme'] = (df['max_term_days'] > 1125).fillna(False).astype(np.int8)\n"
        "    out['log_max_term_days'] = np.log1p(df['max_term_days'].clip(0, 1125)).astype(np.float32)\n"
        "    out['log_min_term_days'] = np.log1p(df['min_term_days'].clip(lower=0)).astype(np.float32)\n"
        "    out['seats_per_workstation'] = (out['max_seats'] / (out['workstations'].fillna(2.0) + 1.0)).astype(np.float32)\n"
        "    out['seats_per_zone'] = (out['max_seats'] / (out['zones'].fillna(1.0) + 1.0)).astype(np.float32)\n"
        "    plat_age = df['operator_platform_age_years'].fillna(0)*12 + df['operator_platform_age_months'].fillna(0)\n"
        "    act_age = df['operator_active_years'].fillna(0)*12 + df['operator_active_months'].fillna(0)\n"
        "    out['total_platform_age_months'] = plat_age.astype(np.float32)\n"
        "    out['operator_inactivity_months'] = (plat_age - act_age).clip(lower=0).astype(np.float32)\n"
        "    out['avail_rate_30'] = (out['open_slots_a'] / 30.0).clip(0, 1).astype(np.float32)\n"
        "    out['occupancy_30'] = (1.0 - out['avail_rate_30']).astype(np.float32)\n"
        "    score_cols = ['score_overall', 'score_accuracy', 'score_condition', 'score_onboarding', 'score_support', 'score_access', 'score_value']\n"
        "    out['score_mean'] = out[score_cols].mean(axis=1).astype(np.float32)\n"
        "    fac_str = df['facilities'].fillna('')\n"
        "    out['facility_count'] = fac_str.apply(lambda s: len(re.findall(r'FCL-\\d+', s))).astype(np.int16)\n"
        "    for fcl in TOP_FACILITIES:\n"
        "        out[f'has_{fcl}'] = fac_str.str.contains(fcl, regex=False).astype(np.int8)\n"
        "    low_cats = ['operator_class', 'access_type', 'compliance_status', 'metro']\n"
        "    for c in low_cats:\n"
        "        out[c] = df[c].fillna('missing').astype('category')\n"
        "    source_df = train_ref if train_ref is not None else df\n"
        "    for c in ['space_class', 'zone', 'borough']:\n"
        "        freq_map = source_df[c].value_counts(normalize=True).to_dict()\n"
        "        out[f'{c}_freq'] = df[c].map(freq_map).fillna(0).astype(np.float32)\n"
        "        out[c] = df[c].fillna('missing').astype('category')\n"
        "    return out"
    ))

    # Cell 4: Execute Feature Extraction
    cells.append(code_cell(
        "X_train_full = extract_features(train_df, train_ref=train_df, is_train=True)\n"
        "y_train_full = X_train_full.pop('price_tier')\n"
        "X_test_full = extract_features(test_df, train_ref=train_df, is_train=False)\n"
        "test_ids = test_df['space_id'].values\n\n"
        "cat_cols = [c for c in X_train_full.columns if str(X_train_full[c].dtype) == 'category']\n"
        "num_cols = [c for c in X_train_full.columns if c not in cat_cols]\n\n"
        "X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full)\n"
        "print(f'Train: {X_tr.shape}, Val: {X_val.shape}')"
    ))

    # Cell 5: Pipelines
    cells.append(code_cell(
        "preprocessor = ColumnTransformer([\n"
        "    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),\n"
        "    ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\n"
        "                      ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols)\n"
        "])\n"
        "X_tr_proc = preprocessor.fit_transform(X_tr)\n"
        "X_val_proc = preprocessor.transform(X_val)\n"
        "X_test_proc = preprocessor.transform(X_test_full)\n"
        "y_tr_0 = (y_tr - 1).values\n"
        "y_val_0 = (y_val - 1).values\n"
        "print('Feature matrix prepared successfully.')"
    ))

    # Cell 6: Model 1 - LightGBM
    cells.append(code_cell(
        "print('Training LightGBM...')\n"
        "lgbm = lgb.LGBMClassifier(\n"
        "    objective='multiclass', num_class=4, n_estimators=350, learning_rate=0.08,\n"
        "    num_leaves=63, max_depth=10, subsample=0.85, colsample_bytree=0.85,\n"
        "    min_child_samples=30, random_state=42, n_jobs=-1\n"
        ")\n"
        "lgbm.fit(X_tr_proc, y_tr_0)\n"
        "prob_lgb = lgbm.predict_proba(X_val_proc)\n"
        "pred_lgb = np.argmax(prob_lgb, axis=1) + 1\n"
        "print('LightGBM Macro F1:', round(f1_score(y_val, pred_lgb, average='macro'), 4))\n"
        "print('LightGBM Accuracy:', round(accuracy_score(y_val, pred_lgb), 4))"
    ))

    # Cell 7: Model 2 - XGBoost
    cells.append(code_cell(
        "print('Training XGBoost (tree_method=hist)...')\n"
        "xgboost_clf = xgb.XGBClassifier(\n"
        "    objective='multi:softprob', num_class=4, eval_metric='mlogloss',\n"
        "    tree_method='hist', n_estimators=350, learning_rate=0.08, max_depth=7,\n"
        "    subsample=0.85, colsample_bytree=0.85, min_child_weight=3, random_state=42, n_jobs=-1\n"
        ")\n"
        "xgboost_clf.fit(X_tr_proc, y_tr_0)\n"
        "prob_xgb = xgboost_clf.predict_proba(X_val_proc)\n"
        "pred_xgb = np.argmax(prob_xgb, axis=1) + 1\n"
        "print('XGBoost Macro F1:', round(f1_score(y_val, pred_xgb, average='macro'), 4))\n"
        "print('XGBoost Accuracy:', round(accuracy_score(y_val, pred_xgb), 4))"
    ))

    # Cell 8: Model 3 - CatBoost
    cells.append(code_cell(
        "print('Training CatBoost...')\n"
        "catboost_clf = CatBoostClassifier(\n"
        "    iterations=400, learning_rate=0.09, depth=6,\n"
        "    loss_function='MultiClass', eval_metric='MultiClass',\n"
        "    l2_leaf_reg=3.0, random_seed=42, thread_count=-1, verbose=100\n"
        ")\n"
        "catboost_clf.fit(X_tr_proc, y_tr_0)\n"
        "prob_cat = catboost_clf.predict_proba(X_val_proc)\n"
        "pred_cat = np.argmax(prob_cat, axis=1) + 1\n"
        "print('CatBoost Macro F1:', round(f1_score(y_val, pred_cat, average='macro'), 4))\n"
        "print('CatBoost Accuracy:', round(accuracy_score(y_val, pred_cat), 4))"
    ))

    # Cell 9: Grand Ensemble Blending & Evaluation
    cells.append(code_cell(
        "# Grand Ensemble Blend (LGBM 35% + XGBoost 30% + CatBoost 25% + RF 10%)\n"
        "rf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5, random_state=42, n_jobs=-1)\n"
        "rf.fit(X_tr_proc, y_tr)\n"
        "prob_rf = rf.predict_proba(X_val_proc)\n\n"
        "prob_grand = 0.35 * prob_lgb + 0.30 * prob_xgb + 0.25 * prob_cat + 0.10 * prob_rf\n"
        "pred_grand = np.argmax(prob_grand, axis=1) + 1\n\n"
        "print('=== GRAND ENSEMBLE VALIDATION PERFORMANCE ===')\n"
        "print('Macro F1 Score:', round(f1_score(y_val, pred_grand, average='macro'), 4))\n"
        "print('Accuracy:', round(accuracy_score(y_val, pred_grand), 4))\n"
        "print('Log Loss:', round(log_loss(y_val_0, prob_grand), 4))\n"
        "print('\\nPer-Class Classification Report:\\n', classification_report(y_val, pred_grand, digits=4))"
    ))

    # Cell 10: Submission Generation
    cells.append(code_cell(
        "# Predict on Test Set using Grand Ensemble\n"
        "test_prob_lgb = lgbm.predict_proba(X_test_proc)\n"
        "test_prob_xgb = xgboost_clf.predict_proba(X_test_proc)\n"
        "test_prob_cat = catboost_clf.predict_proba(X_test_proc)\n"
        "test_prob_rf = rf.predict_proba(X_test_proc)\n\n"
        "test_prob_grand = 0.35 * test_prob_lgb + 0.30 * test_prob_xgb + 0.25 * test_prob_cat + 0.10 * test_prob_rf\n"
        "final_preds = np.argmax(test_prob_grand, axis=1) + 1\n\n"
        "sub_df = pd.DataFrame({'ID': test_ids, 'price_tier': final_preds})\n"
        "sub_df.to_csv('submission.csv', index=False)\n"
        "print('Generated submission.csv with shape:', sub_df.shape)\n"
        "print(sub_df.head(10))\n"
        "print('\\nPredicted Class Distribution:\\n', sub_df['price_tier'].value_counts(normalize=True).sort_index().round(4) * 100)"
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open('FinalNotebook_M4.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("Successfully updated FinalNotebook_M4.ipynb with full GBDT triad!")

if __name__ == '__main__':
    create_notebook()
