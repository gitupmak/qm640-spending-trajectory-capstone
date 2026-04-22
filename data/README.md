# Data

## Download Instructions

1. Go to https://www.dunnhumby.com/source-files/
2. Register for free access
3. Download **The Complete Journey** dataset

## Dataset Overview

**Source:** Dunnhumby Source Files (official academic release)  
**Households:** 2,500 frequent shoppers  
**Period:** 2 years (102 weeks)  
**Description:** Household-level grocery transaction data including all purchases, direct marketing campaign history, and coupon redemption records from a major US grocery retailer.

## Tables Used in This Study

### 1. `transaction_data.csv` — 2,595,732 records

Primary source for all trajectory feature engineering and RFM computation. Each row is one product line from a store receipt.

| Field | Type | Description |
|-------|------|-------------|
| household_key | int | Unique household identifier |
| BASKET_ID | int | Unique purchase occasion (basket) identifier |
| DAY | int | Day of transaction (1–711) |
| PRODUCT_ID | int | Unique product identifier |
| QUANTITY | int | Number of units purchased |
| SALES_VALUE | float | Amount received by retailer from sale |
| STORE_ID | int | Unique store identifier |
| RETAIL_DISC | float | Discount applied via retailer loyalty card (negative value) |
| TRANS_TIME | int | Time of day transaction occurred |
| WEEK_NO | int | Week of transaction (1–102) |
| COUPON_DISC | float | Discount applied via manufacturer coupon |
| COUPON_MATCH_DISC | float | Discount applied via retailer coupon match |

**Note:** `SALES_VALUE` is the amount received by the retailer, not necessarily the price paid by the customer. To compute actual customer price: `(SALES_VALUE - (RETAIL_DISC + COUPON_MATCH_DISC)) / QUANTITY`

**Sample record:**
```
household_key: 2375 | BASKET_ID: 26984851472 | DAY: 1 | PRODUCT_ID: 1004906
QUANTITY: 1 | SALES_VALUE: 1.39 | STORE_ID: 364 | RETAIL_DISC: -0.6
TRANS_TIME: 1631 | WEEK_NO: 1 | COUPON_DISC: 0 | COUPON_MATCH_DISC: 0
```

### 2. `campaign_table.csv` — ~7,200 records

Maps households to campaigns received. Each household may receive a different set of campaigns.

| Field | Type | Description |
|-------|------|-------------|
| DESCRIPTION | str | Campaign type: TypeA, TypeB, or TypeC |
| household_key | int | Unique household identifier |
| CAMPAIGN | int | Campaign identifier (1–30) |

**Campaign type logic:**
- **TypeA** — Personalised. Each household receives 16 coupons selected from a pool based on prior purchase behaviour
- **TypeB / TypeC** — Broadcast. All households in the campaign receive all coupons

**Sample record:**
```
DESCRIPTION: TypeA | household_key: 17 | CAMPAIGN: 26
```

### 3. `campaign_desc.csv` — 30 records

Campaign timing boundaries. Used to define the pre-campaign feature window endpoint.

| Field | Type | Description |
|-------|------|-------------|
| DESCRIPTION | str | Campaign type: TypeA, TypeB, or TypeC |
| CAMPAIGN | int | Campaign identifier (1–30) |
| START_DAY | int | First day of campaign validity |
| END_DAY | int | Last day of campaign validity |

**Sample record:**
```
DESCRIPTION: TypeB | CAMPAIGN: 24 | START_DAY: 659 | END_DAY: 719
```

### 4. `coupon_redempt.csv` — ~2,318 records

**This table is the source of the binary dependent variable.** A household-campaign pair is coded 1 (redeemed) if it appears here; 0 otherwise.

| Field | Type | Description |
|-------|------|-------------|
| household_key | int | Unique household identifier |
| DAY | int | Day the coupon was redeemed |
| COUPON_UPC | int | Unique coupon identifier |
| CAMPAIGN | int | Campaign the coupon belongs to |

**Sample record:**
```
household_key: 1 | DAY: 421 | COUPON_UPC: 10000085364 | CAMPAIGN: 8
```

### 5. `hh_demographic.csv` — ~801 records

Demographic information for approximately one-third of households. Fields use anonymised generic names with ordinal values.

| Field | Type | Description |
|-------|------|-------------|
| household_key | int | Unique household identifier |
| classification_1 | str | Age group (Age Group1–Group6, ordered) |
| classification_2 | str | Marital status (X, Y, Z) |
| classification_3 | str | Income level (Level1–Level12, ordered) |
| HOMEOWNER_DESC | str | Homeowner status (e.g., Homeowner, Renter) |
| classification_4 | str | Household size proxy (1–5+, ordered) |
| classification_5 | str | Demographic group (Group1–Group5, ordered) |
| KID_CATEGORY_DESC | str | Number of children (None/Unknown, 1, 2, 3+) |

**Note:** Demographic data is available for only ~801 of 2,500 households. Used for descriptive profiling of trajectory archetypes only — not included as prediction features due to incomplete coverage.

**Sample record:**
```
household_key: 1 | classification_1: Age Group6 | classification_2: X
classification_3: Level4 | HOMEOWNER_DESC: Homeowner | classification_4: 2
classification_5: Group5 | KID_CATEGORY_DESC: None/Unknown
```

### 6. `product.csv` — ~92,354 records

Three-level product category hierarchy. Used to compute category mix shift trajectory features.

| Field | Type | Description |
|-------|------|-------------|
| PRODUCT_ID | int | Unique product identifier |
| MANUFACTURER | int | Manufacturer code |
| DEPARTMENT | str | Top-level category (e.g., GROCERY, MEAT, PRODUCE) |
| BRAND | str | National or Private label |
| COMMODITY_DESC | str | Mid-level category (e.g., FRZN ICE) |
| SUB_COMMODITY_DESC | str | Lowest-level category (e.g., ICE - CRUSHED/CUBED) |
| CURR_SIZE_OF_PRODUCT | str | Package size (not available for all products) |

**Sample record:**
```
PRODUCT_ID: 25671 | MANUFACTURER: 2 | DEPARTMENT: GROCERY
BRAND: National | COMMODITY_DESC: FRZN ICE
SUB_COMMODITY_DESC: ICE - CRUSHED/CUBED | CURR_SIZE_OF_PRODUCT: 22 LB
```

