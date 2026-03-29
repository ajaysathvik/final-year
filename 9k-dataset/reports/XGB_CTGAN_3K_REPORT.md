# XGBoost + CTGAN 3K Report

Date: 2026-03-26

## Setup

- Source data: `data_binary_only_first_3000.csv`.
- 3k dataset class counts: `2849 fraud`, `151 non_fraud`.
- Train split: `2279 fraud`, `121 non_fraud`.
- Test split: `570 fraud`, `30 non_fraud`.
- CTGAN trains only on the real minority rows from the train split.
- Scraped batch counts: `0 fraud`, `0 non_fraud`.
- Target balance: `1:18` in `non_fraud:fraud` terms.
- Synthetic non-fraud rows requested from scraped batch delta: `0`.

## Results

| Run | Scale Pos Weight | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Macro F1 | Non Fraud F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0531 | 0.9383 | 0.9981 | 0.9368 | 0.9665 | 0.9800 | 0.9990 | 0.7885 | 0.6105 |
| augmented | 0.0531 | 0.9383 | 0.9981 | 0.9368 | 0.9665 | 0.9800 | 0.9990 | 0.7885 | 0.6105 |

## Interpretation

- Baseline scale_pos_weight: `0.0531`.
- Augmented scale_pos_weight: `0.0531`.
- Synthetic minority rows added: `0`.
- Macro F1 change: `+0.0000`.
- Non fraud F1 change: `+0.0000`.