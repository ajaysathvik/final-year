# XGBoost + CTGAN 3K Report

Date: 2026-03-26

## Setup

- Source data: `data_binary_only_first_3000.csv`.
- 3k dataset class counts: `2849 fraud`, `151 non_fraud`.
- Train split: `2279 fraud`, `121 non_fraud`.
- Test split: `570 fraud`, `30 non_fraud`.
- CTGAN trains only on the real minority rows from the train split.
- Augmentation rule: add one more minority block equal to the real minority count in training.

## Results

| Run | Scale Pos Weight | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Macro F1 | Non Fraud F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0531 | 0.9367 | 0.9963 | 0.9368 | 0.9656 | 0.9798 | 0.9989 | 0.7807 | 0.5957 |
| augmented | 0.1062 | 0.9467 | 0.9963 | 0.9474 | 0.9712 | 0.9818 | 0.9990 | 0.8038 | 0.6364 |

## Interpretation

- Baseline scale_pos_weight: `0.0531`.
- Augmented scale_pos_weight: `0.1062`.
- Synthetic minority rows added: `121`.
- Macro F1 change: `+0.0231`.
- Non fraud F1 change: `+0.0406`.