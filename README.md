# From Purchase History to Purchase Trajectory: An AI-Enhanced Framework for Grocery Promotional Response Prediction

**Course:** QM 640 - Data Analytics Capstone
**Program:** DBA (AI & ML), Walsh College
**Student:** Mohd Aslam Khan

---

## Research Overview

This study tests whether pre-campaign spending trajectory features improve grocery coupon redemption prediction beyond static RFM, and whether an LLM agent selecting household-specific observation windows provides further predictive lift.

Traditional targeting relies on static Recency, Frequency, and Monetary (RFM) models that reduce each customer to a single-point snapshot - unable to distinguish a household whose grocery engagement is accelerating from one in steady decline. This study addresses two gaps: (1) no published study has applied multi-dimensional trajectory features to promotional coupon redemption prediction in grocery retail, and (2) all existing studies apply a uniform observation window regardless of household purchase regularity.

**Domain:** Retail & Consumer Packaged Goods (CPG)

---

## Key Results

| Hypothesis | Result | Key Evidence |
|---|---|---|
| H1: Distinct trajectory archetypes exist | **Supported** | k=3 archetypes (silhouette=0.198): Stable 44.9% → 14.4% redemption, Declining 28.2% → 10.7%, Growing 26.9% → 10.5% |
| H2: Gradient boosting outperforms LR | Not supported | LR best (AUC=0.7378) > RF (0.7311) > XGBoost (0.7272) > LightGBM (0.7242) |
| H3: Trajectory improves AUC by ≥2pp | Not supported | ΔAUC=−0.69pp, DeLong z=−1.24, p=0.216 |
| H4: LLM agent beats fixed + heuristic | Not supported | LLM (0.7456) vs Fixed 12w (0.7435) vs Heuristic (0.7446), p=0.181 |

**Central finding:** Prior redemption behaviour (`prior_redeem_rate`, mean |SHAP|=0.415) dominates all other signals. A lightweight 4-feature Logistic Regression pipeline achieves **2.78× targeting precision** over random selection at K=20%, deployable as a Python microservice with no trajectory feature engineering required.

---

## Research Questions

| RQ | Question | Method | Result |
|----|----------|--------|--------|
| RQ1 | What distinct spending trajectory archetypes emerge prior to a promotional campaign? | K-Means clustering, silhouette score optimisation | k=3 archetypes identified |
| RQ2 | Which ML model best predicts coupon redemption using trajectory + RFM features? | 4-model comparison, temporal 70/30 split, AUC-ROC | Logistic Regression (AUC=0.7378) |
| RQ3 | Do trajectory features improve AUC by ≥2pp over static RFM alone? | DeLong's test, SHAP feature importance | Not supported (p=0.216) |
| RQ4 | Does LLM-assisted household-specific window selection improve prediction? | Three-way comparison (Fixed / Heuristic / LLM Agent), DeLong's test | Not supported (p=0.181) |

---

## Repository Structure

```
qm640-spending-trajectory-capstone/
│
├── README.md                          ← This file
│
├── data/
│   └── dunnhumby_The-Complete-Journey/
│       └── *.csv                      ← Raw Dunnhumby data (download separately)
│   └── README.md                      ← Download instructions + full data dictionary
│
├── notebooks/
│   ├── eda.ipynb                      ← Full EDA: 10 figures, campaign analysis,
│   │                                     RFM + trajectory distributions, correlation matrices
│   └── modelling.ipynb                ← Complete modelling pipeline:
│                                         Cell 1: Setup + temporal split
│                                         Cell 2: LR baseline (AUC=0.7378)
│                                         Cell 3: RF, XGBoost, LightGBM (RQ2)
│                                         Cell 4: DeLong's test RQ3 (Condition A/B/C)
│                                         Cell 5: SHAP analysis
│                                         Cell 6: Agent input metrics + heuristic
│                                         Cell 7: Azure OpenAI LLM agent (1,560 calls)
│                                         Cell 8: RQ4 three-way comparison + DeLong's test
│
├── src/
│   ├── feature_engineering.py         ← RFM + trajectory feature functions
│   │                                     compute_rfm(hh_campaigns, transactions_clean)
│   │                                     compute_trajectory_features(..., window_weeks=8)
│   └── generate_window_features.py    ← Generates full_features_4w.csv and
│                                         full_features_12w.csv from raw data (~25 min)
│
└── outputs/
    ├── full_features.csv              ← Analytical sample: 7,123 pairs × 15 cols (8w window)
    ├── full_features_4w.csv           ← 6,930 pairs (4-week window)
    ├── full_features_12w.csv          ← 7,167 pairs (12-week window)
    ├── hh_campaigns.csv               ← Household-campaign pairs with redemption label
    ├── agent_metrics.csv              ← 4 agent input metrics per household (1,560 rows)
    ├── agent_log.csv                  ← All 1,560 LLM API calls: metrics, window, justification
    ├── results_rq2.csv                ← Four-model AUC comparison
    ├── results_rq3_ablation.csv       ← Feature condition comparison (A, B, C)
    ├── results_rq4.csv                ← Three-way window selector comparison
    ├── shap_summary.csv               ← Mean absolute SHAP values, all 10 features
    └── fig_01 to fig_12_*.png         ← All figures referenced in the final report
```

---

## Dataset

**Dunnhumby Complete Journey**
Source: https://www.dunnhumby.com/source-files/
*See `data/README.md` for download instructions and full data dictionary*

| Statistic | Value |
|---|---|
| Raw transactions | 2,595,732 |
| After cleaning | 2,576,788 |
| Households (total) | 2,500 |
| Campaign-exposed households | 1,560 |
| Household-campaign pairs (8w) | 7,123 |
| Pair-level redemption rate | 12.48% |
| Campaigns | 30 (weeks 32–101) |
| Study period | 102 consecutive weeks |

---

## Analytical Pipeline

| Phase | Method | Output | RQ |
|-------|--------|--------|----|
| Data cleaning | Zero-value removal, winsorisation | 2,576,788 clean records | All |
| Feature engineering | 4 static RFM + 6 OLS trajectory features at 4w / 8w / 12w | `full_features_*.csv` | All |
| Archetype discovery | K-Means clustering (k=2–8), silhouette score | k=3: Stable / Declining / Growing | RQ1 |
| Model comparison | LR, RF, XGBoost, LightGBM — temporal 70/30 split, SMOTE | AUC, F1, Precision@K | RQ2 |
| Incremental AUC test | Conditions A / B / C, DeLong's test, SHAP | ΔAUC, z-stat, p-value | RQ3 |
| Window sweep | Fixed 4w / 8w / 12w | Best window = 12w (AUC=0.7435) | RQ3/RQ4 |
| Adaptive window agent | Azure OpenAI GPT-4.1-mini, temp=0.0, Pydantic structured output | `agent_log.csv` | RQ4 |
| Statistical testing | DeLong's nonparametric AUC comparison | p-values for H3 and H4 | RQ3, RQ4 |

**Evaluation design:** Temporal campaign split - earliest 21 campaigns (70%) = train (4,629 pairs), latest 9 campaigns (30%) = test (2,494 pairs). SMOTE applied to training set only.

---

## Feature Architecture

### Static RFM (4 features — 26-week lookback)

| Feature | Definition |
|---|---|
| `recency` | Days since last purchase before campaign START_DAY |
| `frequency` | Unique shopping trips (baskets) in 26-week window |
| `monetary` | Mean basket value across all trips in 26-week window |
| `prior_redeem_rate` | Fraction of prior campaigns redeemed (0.0 if none) |

### Trajectory (6 features — computed at window w ∈ {4, 8, 12} weeks)

| Feature | Definition |
|---|---|
| `spend_slope` | 1st-order OLS coefficient of weekly spend |
| `spend_acceleration` | 2nd-order OLS coefficient of weekly spend |
| `category_mix_shift` | Jaccard dissimilarity of product departments (first vs second half of window) |
| `basket_size_trend` | OLS slope of mean distinct product line items per basket per week |
| `promo_engage_velocity` | OLS slope of weekly proportion of discounted transactions |
| `visit_freq_trend` | OLS slope of weekly trip count |

All trajectory features are computed strictly from transactions **before** each campaign's `START_DAY` to prevent data leakage, enforced programmatically in `src/feature_engineering.py`.

### Agent Input Metrics (RQ4 only - not model features)

| Metric | Definition |
|---|---|
| `purchase_regularity` | Mean frequency normalised to 0–1 |
| `data_density` | 26w frequency scaled to 12w equivalent (max 12.0) |
| `trajectory_volatility` | Mean absolute spend acceleration |
| `signal_recency_ratio` | max(0, 1 − mean_recency / 90) |

---

## LLM Agent Design (RQ4)

- **Model:** Azure OpenAI GPT-4.1-mini
- **Temperature:** 0.0 (fully deterministic)
- **Output format:** Structured via Pydantic — `window: Literal[4, 8, 12]` + `reason: str`
- **API calls:** 1,560 (zero errors)
- **Window distribution:** 4w → 1,511 (96.9%), 8w → 36 (2.3%), 12w → 13 (0.8%)
- **Agreement with heuristic:** 63.65%
- **Primary decision driver (content analysis):** `data_density` (100% of justifications), `signal_recency_ratio` (98.7%)

Full system prompt and Pydantic schema are documented in the final report (Appendix B).

---

## References

- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Biometrics, 44(3), 837–845.
- Dunnhumby. (2014). The Complete Journey. https://www.dunnhumby.com/source-files/
- Hughes, A. M. (1994). Strategic database marketing. Probus Publishing.
- Langen, H., & Huber, M. (2023). PLoS ONE, 18(1), e0278937.
- Lin, J. (2025). PLoS ONE, 20(5), e0321854.
- Mena, G., et al. (2024). Annals of Operations Research, 339(1), 765–787.
- Pérez, A. S., et al. (2025). Proceedings of NAACL 2025 (Vol. 1, pp. 562–582).
- Smaili, M. Y., & Hachimi, H. (2023). Ain Shams Engineering Journal, 14(12), 102254.
- Wan, M., McAuley, J., & Leskovec, J. (2017). Proceedings of WWW 2017 (pp. 1103–1112).
- Wang, S., Sun, L., & Yu, Y. (2024). Scientific Reports, 14, Article 17491.
