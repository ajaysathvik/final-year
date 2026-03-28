# XGBoost Imbalanced Test Report (3K)

Date: 2026-03-26

## Overview

This report summarizes an XGBoost baseline trained on a balanced training subset and evaluated on an imbalanced held-out test set from `data_binary_only_first_3000.csv`.

## Dataset Design

- Rows with `annotation.is_fraud = -1` were dropped.
- The training portion was balanced by downsampling `fraud` to match the available `not_fraud` rows.

Train/test sizes:
- Balanced train rows: 242
- Imbalanced test rows: 600
- Train class counts: {'0': 121, '1': 121}
- Test class counts: {'0': 30, '1': 570}

## Results

- Accuracy: 0.9300
- Precision: 0.9944
- Recall: 0.9316
- F1: 0.9620
- ROC-AUC: 0.9788

Confusion matrix:
- TN: 27
- FP: 3
- FN: 39
- TP: 531
