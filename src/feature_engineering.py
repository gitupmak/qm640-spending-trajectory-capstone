import pandas as pd
import numpy as np

def compute_rfm(hh_campaigns, transactions_clean):
    """
    Compute static RFM features per household-campaign pair.
    All features computed strictly from transactions before campaign START_DAY.
    
    Features:
        recency       : days between last purchase and campaign START_DAY
        frequency     : number of shopping trips in 26 weeks before START_DAY
        monetary      : mean basket value in 26 weeks before START_DAY
        prior_redeem_rate : fraction of prior campaigns redeemed before this one
    
    Returns DataFrame with household_key, CAMPAIGN, and 4 RFM features.
    """
    records = []

    for _, row in hh_campaigns.iterrows():
        hh = row['household_key']
        campaign = row['CAMPAIGN']
        start_day = row['START_DAY']
        window_start = start_day - (26 * 7)  # 26-week lookback

        # Transactions strictly before campaign start
        hh_tx = transactions_clean[
            (transactions_clean['household_key'] == hh) &
            (transactions_clean['DAY'] < start_day) &
            (transactions_clean['DAY'] >= window_start)
        ]

        if len(hh_tx) == 0:
            records.append({
                'household_key': hh, 'CAMPAIGN': campaign,
                'recency': np.nan, 'frequency': np.nan,
                'monetary': np.nan, 'prior_redeem_rate': np.nan
            })
            continue

        # Recency: days since last purchase before campaign start
        recency = start_day - hh_tx['DAY'].max()

        # Frequency: unique shopping trips (baskets) in window
        frequency = hh_tx['BASKET_ID'].nunique()

        # Monetary: mean basket value in window
        basket_values = hh_tx.groupby('BASKET_ID')['SALES_VALUE'].sum()
        monetary = basket_values.mean()

        # Prior redemption rate: campaigns redeemed before this one
        prior = hh_campaigns[
            (hh_campaigns['household_key'] == hh) &
            (hh_campaigns['START_DAY'] < start_day)
        ]
        prior_redeem_rate = prior['redeemed'].mean() if len(prior) > 0 else 0.0

        records.append({
            'household_key': hh, 'CAMPAIGN': campaign,
            'recency': recency, 'frequency': frequency,
            'monetary': monetary, 'prior_redeem_rate': prior_redeem_rate
        })

    return pd.DataFrame(records)


def compute_trajectory_features(hh_campaigns, transactions_clean, product, window_weeks=8):
    """
    Compute 6 trajectory features per household-campaign pair.
    All features computed strictly from transactions in [START_DAY - window_weeks*7, START_DAY).
    
    Features:
        spend_slope         : OLS slope of weekly spend over window
        spend_acceleration  : second-order OLS coefficient of weekly spend
        category_mix_shift  : Jaccard dissimilarity of category distribution
                              (first half vs second half of window)
        basket_size_trend   : OLS slope of items-per-trip over window
        promo_engage_velocity: OLS slope of promotional item proportion over window
        visit_freq_trend    : OLS slope of weekly trip count over window
    
    Returns DataFrame with household_key, CAMPAIGN, and 6 trajectory features.
    """
    from numpy.polynomial import polynomial as P

    window_days = window_weeks * 7
    records = []

    # Pre-merge product for category info
    tx_product = transactions_clean.merge(
        product[['PRODUCT_ID', 'DEPARTMENT']],
        on='PRODUCT_ID', how='left'
    )

    for _, row in hh_campaigns.iterrows():
        hh = row['household_key']
        campaign = row['CAMPAIGN']
        start_day = row['START_DAY']
        w_start = start_day - window_days
        w_end = start_day  # strictly before

        hh_tx = tx_product[
            (tx_product['household_key'] == hh) &
            (tx_product['DAY'] >= w_start) &
            (tx_product['DAY'] < w_end)
        ].copy()

        # Default NaN record
        base = {'household_key': hh, 'CAMPAIGN': campaign,
                'spend_slope': np.nan, 'spend_acceleration': np.nan,
                'category_mix_shift': np.nan, 'basket_size_trend': np.nan,
                'promo_engage_velocity': np.nan, 'visit_freq_trend': np.nan}

        if len(hh_tx) < 2:
            records.append(base)
            continue

        # Assign relative week number (0-indexed)
        hh_tx['rel_week'] = ((hh_tx['DAY'] - w_start) // 7).clip(
            upper=window_weeks - 1
        )

        # --- Weekly aggregations ---
        weekly_spend = hh_tx.groupby('rel_week')['SALES_VALUE'].sum()
        weekly_trips = hh_tx.groupby('rel_week')['BASKET_ID'].nunique()
        weekly_items = hh_tx.groupby('rel_week')['QUANTITY'].sum()
        weekly_promo = hh_tx.groupby('rel_week').apply(
            lambda x: (x['COUPON_DISC'] > 0).mean()
        )

        weeks = np.arange(window_weeks)

        def ols_slope(series):
            """Fit OLS line, return slope. Fills missing weeks with 0."""
            y = series.reindex(weeks, fill_value=0).values.astype(float)
            if y.std() == 0:
                return 0.0
            coeffs = np.polyfit(weeks, y, 1)
            return float(coeffs[0])

        def ols_acceleration(series):
            """Fit 2nd order polynomial, return quadratic coefficient."""
            y = series.reindex(weeks, fill_value=0).values.astype(float)
            if y.std() == 0:
                return 0.0
            coeffs = np.polyfit(weeks, y, 2)
            return float(coeffs[0])

        # --- Feature 1 & 2: Spend slope and acceleration ---
        spend_slope = ols_slope(weekly_spend)
        spend_accel = ols_acceleration(weekly_spend)

        # --- Feature 3: Category mix shift (Jaccard dissimilarity) ---
        mid = w_start + window_days // 2
        cats_first = set(hh_tx[hh_tx['DAY'] < mid]['DEPARTMENT'].dropna().unique())
        cats_second = set(hh_tx[hh_tx['DAY'] >= mid]['DEPARTMENT'].dropna().unique())
        if cats_first | cats_second:
            jaccard = 1 - len(cats_first & cats_second) / len(cats_first | cats_second)
        else:
            jaccard = 0.0

        # --- Feature 4: Basket size trend ---
        # items_per_trip = hh_tx.groupby(['rel_week', 'BASKET_ID'])['QUANTITY'].sum()
        # items_per_trip = items_per_trip.groupby('rel_week').mean()
        # basket_trend = ols_slope(items_per_trip)

        # --- Feature 4: Basket size trend ---
        # Use number of distinct product line items per basket (more stable than quantity)
        items_per_trip = hh_tx.groupby(['rel_week', 'BASKET_ID'])['PRODUCT_ID'].nunique()
        items_per_trip = items_per_trip.groupby('rel_week').mean()
        basket_trend = ols_slope(items_per_trip)

        # --- Feature 5: Promo engagement velocity ---
        # Use RETAIL_DISC > 0 OR COUPON_DISC > 0 as promo flag
        hh_tx['is_promo'] = (
            (hh_tx['RETAIL_DISC'].abs() > 0) | 
            (hh_tx['COUPON_DISC'].abs() > 0)
        ).astype(float)
        weekly_promo = hh_tx.groupby('rel_week')['is_promo'].mean()
        promo_slope = ols_slope(weekly_promo)

        # --- Feature 6: Visit frequency trend ---
        visit_slope = ols_slope(weekly_trips)

        records.append({
            'household_key': hh, 'CAMPAIGN': campaign,
            'spend_slope': spend_slope,
            'spend_acceleration': spend_accel,
            'category_mix_shift': jaccard,
            'basket_size_trend': basket_trend,
            'promo_engage_velocity': promo_slope,
            'visit_freq_trend': visit_slope
        })

    return pd.DataFrame(records)