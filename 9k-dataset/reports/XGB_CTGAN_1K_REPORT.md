# XGBoost + CTGAN 1K Report

Date: 2026-03-26

## Setup

- Source subset: first `1000` rows from `processed/data_binary_only.csv`.
- 1k subset class counts: `958 fraud`, `42 non_fraud`.
- Train split: `766 fraud`, `34 non_fraud`.
- Test split: `192 fraud`, `8 non_fraud`.
- CTGAN trains only on the real minority rows from the 800-row train split.
- Augmentation rule: add one more minority block equal to the real minority count in training.

## Results

| Run | Scale Pos Weight | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Macro F1 | Non Fraud F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0444 | 0.9300 | 1.0000 | 0.9271 | 0.9622 | 0.9834 | 0.9993 | 0.7477 | 0.5333 |
| augmented | 0.0888 | 0.9500 | 1.0000 | 0.9479 | 0.9733 | 0.9951 | 0.9998 | 0.7943 | 0.6154 |

## Interpretation

- Baseline scale_pos_weight: `0.0444`.
- Augmented scale_pos_weight: `0.0888`.
- Synthetic minority rows added: `34`.
- Macro F1 change: `+0.0466`.
- Non fraud F1 change: `+0.0821`.