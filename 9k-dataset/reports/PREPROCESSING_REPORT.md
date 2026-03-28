# Preprocessing Report

Date: 2026-03-23

## Overview

This report summarizes the preprocessing work completed for the dataset in [data.csv](c:/Users/SAKETH/Documents/Project/9k-dataset/data.csv).

Primary output:
- [processed/data_preprocessed.csv](c:/Users/SAKETH/Documents/Project/9k-dataset/processed/data_preprocessed.csv)

Supporting summary:
- [processed/preprocessing_summary.json](c:/Users/SAKETH/Documents/Project/9k-dataset/processed/preprocessing_summary.json)

## What Was Cleaned

- Removed 21 trailing empty export columns.
- Retained the 24 real annotation columns.
- Normalized fraud type naming where variants existed.
- Standardized multi-value categorical fields such as payment method, fraud channel, victim action, and request type.
- Converted fraud-label fields into clean binary values.
- Standardized numeric score fields such as urgency and psychological tactics.
- Parsed text amounts into a numeric amount column when possible.
- Added a binary flag indicating whether a numeric amount was successfully extracted.

## Dataset Shape

- Raw dataset shape: 9,630 rows x 45 columns
- Processed dataset shape: 9,630 rows x 26 columns

## Key Output Columns Added

- `annotation.key_features.amount_normalized`
- `annotation.key_features.has_amount`

## Class Distribution

`annotation.is_fraud`

- `1`: 5,434
- `0`: 294
- `-1`: 3,902

## Top Fraud Types

- `none`: 4,199
- `transaction`: 3,719
- `commerce`: 795
- `social_engineering`: 767
- `meta`: 45
- `credential`: 35

## Currency Snapshot

- `unknown`: 7,611
- `USD`: 1,168
- `INR`: 586
- `EUR`: 78
- `GBP`: 45

## Amount Parsing

- Numeric amount values extracted: 2,513
- Text values like `$20`, `5 lakhs`, and `1000 USD` were converted where possible.
- Non-parsable or ambiguous values remain empty in `annotation.key_features.amount_normalized`.

## Notes For Modeling

- The processed file is cleaner, but it is not yet a final XGBoost feature matrix.
- Several columns are still categorical text fields and need encoding before training.
- `annotation.is_fraud` includes `-1`, so target handling should be defined before model training.

## Current Recommendation

Use [processed/data_preprocessed.csv](c:/Users/SAKETH/Documents/Project/9k-dataset/processed/data_preprocessed.csv) as the base dataset for feature engineering and model preparation.
