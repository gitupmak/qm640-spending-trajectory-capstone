# Generates full_features_4w.csv and full_features_12w.csv

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import compute_rfm, compute_trajectory_features

# ── Load raw data ─────────────────────────────────────────────────────────────
print("Loading raw data...")

dtypes = {
    'household_key': 'int32',
    'BASKET_ID': 'int64',
    'DAY': 'int16',
    'PRODUCT_ID': 'int32',
    'QUANTITY': 'int16',
    'SALES_VALUE': 'float32',
    'STORE_ID': 'int16',
    'RETAIL_DISC': 'float32',
    'TRANS_TIME': 'int16',
    'WEEK_NO': 'int16',
    'COUPON_DISC': 'float32',
    'COUPON_MATCH_DISC': 'float32'
}

transactions = pd.read_csv(
    r'data\dunnhumby_The-Complete-Journey\transaction_data.csv',
    dtype=dtypes
)
campaign_table = pd.read_csv(r'data\dunnhumby_The-Complete-Journey\campaign_table.csv')
campaign_desc = pd.read_csv(r'data\dunnhumby_The-Complete-Journey\campaign_desc.csv')
coupon_redempt = pd.read_csv(r'data\dunnhumby_The-Complete-Journey\coupon_redempt.csv')
product = pd.read_csv(r'data\dunnhumby_The-Complete-Journey\product.csv')
hh_demo = pd.read_csv(r'data\dunnhumby_The-Complete-Journey\hh_demographic.csv')

print(f"Transactions: {len(transactions):,}")

transactions_clean = transactions[
    (transactions['SALES_VALUE'] > 0) &
    (transactions['QUANTITY'] > 0)
].copy()
print(f"After cleaning: {len(transactions_clean):,}")

# ── Build hh_campaigns with redemption label ──────────────────────────────────
hh_campaigns = campaign_table.merge(
    campaign_desc[['CAMPAIGN','START_DAY','END_DAY']],
    on='CAMPAIGN', how='left'
)

redeemed_hh_camp = coupon_redempt[['household_key','CAMPAIGN']].drop_duplicates()
redeemed_hh_camp['redeemed'] = 1

hh_campaigns = hh_campaigns.merge(redeemed_hh_camp, on=['household_key','CAMPAIGN'], how='left')
hh_campaigns['redeemed'] = hh_campaigns['redeemed'].fillna(0).astype(int)

print(f"hh_campaigns: {len(hh_campaigns):,} pairs")

# ── Compute RFM (same for all windows) ───────────────────────────────────────
print("\nComputing RFM features (shared across all windows)...")
rfm_df = compute_rfm(hh_campaigns, transactions_clean)
print(f"RFM computed: {len(rfm_df):,} rows")

# ── Function to build full feature file for a given window ───────────────────
def build_full_features(window_weeks, rfm_df, hh_campaigns, transactions_clean, product):
    print(f"\n{'='*50}")
    print(f"Computing trajectory features: {window_weeks}w window...")

    traj_df = compute_trajectory_features(
        hh_campaigns, transactions_clean, product,
        window_weeks=window_weeks
    )
    print(f"Trajectory computed: {len(traj_df):,} rows")

    # Merge RFM + trajectory + redemption label
    full = rfm_df.merge(traj_df, on=['household_key','CAMPAIGN'], how='inner')
    full = full.merge(
        hh_campaigns[['household_key','CAMPAIGN','redeemed']],
        on=['household_key','CAMPAIGN'], how='left'
    )

    # Drop rows with any NaN in features
    feature_cols = ['recency','frequency','monetary','prior_redeem_rate',
                    'spend_slope','spend_acceleration','category_mix_shift',
                    'basket_size_trend','promo_engage_velocity','visit_freq_trend']
    before = len(full)
    full = full.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = before - len(full)
    print(f"Dropped {dropped} rows with NaN (insufficient window data)")
    print(f"Final shape: {full.shape}")
    print(f"Redemption rate: {full['redeemed'].mean():.4f}")

    # Save
    out_path = f'outputs/full_features_{window_weeks}w.csv'
    full.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return full

# ── Generate 4w and 12w ───────────────────────────────────────────────────────
df_4w  = build_full_features(4,  rfm_df, hh_campaigns, transactions_clean, product)
df_12w = build_full_features(12, rfm_df, hh_campaigns, transactions_clean, product)

print("\nDone. Files saved:")
print("   outputs/full_features_4w.csv")
print("   outputs/full_features_12w.csv")
print("\nSummary:")
print(f"  4w:  {len(df_4w):,} pairs | redemption {df_4w['redeemed'].mean():.4f}")
print(f"  12w: {len(df_12w):,} pairs | redemption {df_12w['redeemed'].mean():.4f}")