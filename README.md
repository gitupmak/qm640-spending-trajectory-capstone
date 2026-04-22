# From Purchase History to Purchase Trajectory: An AI-Enhanced Framework for Grocery Promotional Response Prediction

**Course:** QM 640 - Data Analytics Capstone  
**Program:** MS (AI & ML), Walsh College  
**Student:** Mohd Aslam Khan

## Research Overview

This study investigates whether pre-campaign spending trajectory features - the direction, velocity, and category composition of a household's purchasing behaviour in the weeks before a campaign - predict promotional coupon redemption better than static RFM features alone.

A secondary contribution is an LLM-assisted adaptive window selection agent that selects the optimal lookback window (4, 8, or 12 weeks) per household based on its individual data characteristics, testing whether household-specific window selection further improves prediction over any fixed window strategy.

**Domain:** Retail & Consumer Packaged Goods (CPG)

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | What distinct spending trajectory archetypes emerge from household grocery transaction data in the weeks prior to a promotional campaign? |
| RQ2 | Which predictive modelling approach best identifies households likely to redeem promotional coupons when pre-campaign spending trajectory features are combined with static purchase history features? |
| RQ3 | Do pre-campaign spending trajectory features improve campaign response prediction accuracy beyond what static RFM features alone can achieve, and which trajectory dimensions contribute most to any improvement? |
| RQ4 | Does using a household-specific observation window, selected based on each household's individual purchase data characteristics improve promotional campaign response prediction compared to applying a uniform observation window across all households, and which household data characteristics most influence the window selection decision? |

## Repository Structure

```
qm640-spending-trajectory-capstone/
│
├── README.md
│
├── data/
│   └── dunnhumby_The-Complete-Journey/
│       └── Data in CSV files
│   └── README.md
│
├── notebooks/
│   └── eda.ipynb 
│
├── src/
│   └── feature_engineering.py
│
└── outputs/
    └── (plots and results saved here)
```

## Dataset

**Dunnhumby Complete Journey**  
Source: https://www.dunnhumby.com/source-files/  
*See `data/README.md` for download instructions and details*


## Analytical Pipeline

| Phase | Method | RQ |
|-------|--------|----|
| Feature Engineering | Rolling window trajectory features (4/8/12 weeks) + static RFM | All |
| Archetype Discovery | K-Means clustering, silhouette score optimisation | RQ1 |
| Model Comparison | Logistic Regression, Random Forest, XGBoost, LightGBM | RQ2 |
| Incremental AUC Test | DeLong's test, SHAP feature importance | RQ3 |
| Adaptive Window Agent | OpenAI GPT-4.1 API, temperature=0.0, content analysis | RQ4 |

**Evaluation design:** Temporal campaign split (earliest ~21 campaigns = train, latest ~9 = test). Random split run as robustness check.

