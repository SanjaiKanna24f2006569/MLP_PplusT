import re
import numpy as np
import pandas as pd

TOP_FACILITIES = [
    'FCL-078', 'FCL-145', 'FCL-135', 'FCL-093', 'FCL-080',
    'FCL-139', 'FCL-020', 'FCL-058', 'FCL-056', 'FCL-138'
]

def extract_features(df, train_ref=None, is_train=True):
    """
    Extract robust, leak-free domain features from the dataset.
    If train_ref is provided, frequency maps and reference medians are learned from train_ref.
    """
    out = pd.DataFrame(index=df.index)
    
    # 1. Target (if present)
    if 'price_tier' in df.columns:
        out['price_tier'] = df['price_tier'].astype(np.int8)
        
    # 2. Key raw numerical features
    num_cols = [
        'geo_lat', 'geo_lon', 'max_seats', 'utility_rooms', 'zones', 'workstations',
        'min_term_days', 'max_term_days', 'max_term_floor', 'min_term_ceiling',
        'min_term_avg', 'max_term_avg', 'open_slots_a', 'open_slots_b', 'open_slots_c',
        'open_slots_d', 'open_slots_e', 'review_count', 'review_count_12m', 'review_count_30d',
        'review_count_prev_yr', 'booked_load', 'score_overall', 'score_accuracy',
        'score_condition', 'score_onboarding', 'score_support', 'score_access', 'score_value',
        'operator_metro_units', 'operator_metro_whole', 'operator_metro_private',
        'operator_metro_shared', 'reviews_per_month', 'operator_portfolio_size',
        'operator_platform_age_years', 'operator_platform_age_months',
        'operator_active_years', 'operator_active_months'
    ]
    for c in num_cols:
        if c in df.columns:
            out[c] = df[c].astype(np.float32)
            
    # 3. Missingness indicators
    out['is_missing_borough'] = df['borough'].isna().astype(np.int8)
    out['is_missing_reviews'] = df['score_overall'].isna().astype(np.int8)
    out['is_missing_zones'] = df['zones'].isna().astype(np.int8)
    out['is_missing_workstations'] = df['workstations'].isna().astype(np.int8)
    out['is_missing_utility_rooms'] = df['utility_rooms'].isna().astype(np.int8)

    # 4. Extreme values treatment and flags
    out['max_term_is_extreme'] = (df['max_term_days'] > 1125).fillna(False).astype(np.int8)
    out['log_max_term_days'] = np.log1p(df['max_term_days'].clip(0, 1125)).astype(np.float32)
    out['log_max_term_avg'] = np.log1p(df['max_term_avg'].clip(0, 1125)).astype(np.float32)
    out['log_min_term_days'] = np.log1p(df['min_term_days'].clip(lower=0)).astype(np.float32)
    out['log_portfolio_size'] = np.log1p(df['operator_portfolio_size'].clip(lower=0)).astype(np.float32)

    # 5. Capacity ratios
    out['seats_per_workstation'] = (out['max_seats'] / (out['workstations'].fillna(2.0) + 1.0)).astype(np.float32)
    out['seats_per_zone'] = (out['max_seats'] / (out['zones'].fillna(1.0) + 1.0)).astype(np.float32)
    out['workstations_per_zone'] = (out['workstations'].fillna(2.0) / (out['zones'].fillna(1.0) + 1.0)).astype(np.float32)
    out['utility_rooms_per_seat'] = (out['utility_rooms'].fillna(1.0) / (out['max_seats'] + 1.0)).astype(np.float32)
    out['seats_per_portfolio'] = (out['max_seats'] / (out['operator_portfolio_size'].fillna(4.0) + 1.0)).astype(np.float32)

    # 6. Operator platform timeline
    plat_age = df['operator_platform_age_years'].fillna(0) * 12 + df['operator_platform_age_months'].fillna(0)
    act_age = df['operator_active_years'].fillna(0) * 12 + df['operator_active_months'].fillna(0)
    out['total_platform_age_months'] = plat_age.astype(np.float32)
    out['total_active_age_months'] = act_age.astype(np.float32)
    out['operator_inactivity_months'] = (plat_age - act_age).clip(lower=0).astype(np.float32)

    # 7. Operator portfolio composition ratios
    tot_units = out['operator_metro_units'].fillna(0) + 1.0
    out['operator_metro_whole_ratio'] = (out['operator_metro_whole'].fillna(0) / tot_units).astype(np.float32)
    out['operator_metro_private_ratio'] = (out['operator_metro_private'].fillna(0) / tot_units).astype(np.float32)
    out['operator_metro_shared_ratio'] = (out['operator_metro_shared'].fillna(0) / tot_units).astype(np.float32)

    # 8. Slot availability & occupancy rates
    out['avail_rate_30'] = (out['open_slots_a'] / 30.0).clip(0, 1).astype(np.float32)
    out['avail_rate_60'] = (out['open_slots_b'] / 60.0).clip(0, 1).astype(np.float32)
    out['avail_rate_90'] = (out['open_slots_c'] / 90.0).clip(0, 1).astype(np.float32)
    out['avail_rate_365'] = (out['open_slots_d'] / 365.0).clip(0, 1).astype(np.float32)
    out['occupancy_30'] = (1.0 - out['avail_rate_30']).astype(np.float32)
    out['occupancy_365'] = (1.0 - out['avail_rate_365']).astype(np.float32)
    out['booked_load_ratio'] = (out['booked_load'] / 255.0).clip(0, 1).astype(np.float32)

    # 9. Review scores composite
    score_cols = ['score_overall', 'score_accuracy', 'score_condition', 'score_onboarding', 'score_support', 'score_access', 'score_value']
    out['score_mean'] = out[score_cols].mean(axis=1).astype(np.float32)

    # 10. Date features
    ref_date = pd.to_datetime('2024-08-30')
    last_rev = pd.to_datetime(df['last_review_date'], errors='coerce')
    first_rev = pd.to_datetime(df['first_review_date'], errors='coerce')
    out['review_span_days'] = (last_rev - first_rev).dt.days.fillna(0).clip(lower=0).astype(np.float32)
    out['days_since_last_review'] = (ref_date - last_rev).dt.days.fillna(9999).clip(lower=0).astype(np.float32)

    # 11. Facilities extraction
    fac_str = df['facilities'].fillna('')
    out['facility_count'] = fac_str.apply(lambda s: len(re.findall(r'FCL-\d+', s))).astype(np.int16)
    for fcl in TOP_FACILITIES:
        out[f'has_{fcl}'] = fac_str.str.contains(fcl, regex=False).astype(np.int8)

    # 12. Categorical Encodings
    # Low-cardinality nominals -> Categorical codes / one-hot
    low_cats = ['operator_class', 'access_type', 'compliance_status', 'metro']
    for c in low_cats:
        out[c] = df[c].fillna('missing').astype('category')
        
    bool_cats = ['operator_certified', 'operator_has_avatar', 'operator_id_verified', 'bookable']
    for c in bool_cats:
        out[c] = (df[c] == 't').astype(np.int8)

    # Frequency encoding for high cardinality features
    source_df = train_ref if train_ref is not None else df
    for c in ['space_class', 'zone', 'borough']:
        freq_map = source_df[c].value_counts(normalize=True).to_dict()
        out[f'{c}_freq'] = df[c].map(freq_map).fillna(0).astype(np.float32)
        out[c] = df[c].fillna('missing').astype('category')

    return out
