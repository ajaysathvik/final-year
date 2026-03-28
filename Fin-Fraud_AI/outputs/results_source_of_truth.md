# Verified Results Source of Truth

This file is the final paper-facing summary derived directly from the generated CSV outputs in `outputs/`.

## Files Used

- `outputs/classifier_comparison.csv`
- `outputs/robustness_results.csv`
- `outputs/ood_results.csv`
- `outputs/synthetic_quality_metrics.csv`
- `outputs/learning_curve_results.csv`
- `outputs/agentic_metric_results.csv`

## 1. Raw vs CTGAN vs Adv-CTGAN

Source: `outputs/classifier_comparison.csv`

- Best clean model overall: `Original + Random Forest`
  - Accuracy: `0.9962`
  - F1-score: `0.9980`
  - AUC-ROC: `0.9997`
  - Log Loss: `0.0137`
- Best CTGAN clean result:
  - `CTGAN + Neural Network (MLP)` and `CTGAN + Random Forest` both reach F1-score `0.9951`
- Best Adv-CTGAN clean result:
  - `Adv-CTGAN + Gradient Boosting` reaches F1-score `0.9941`
- Main clean-performance takeaway:
  - The original regime remains strongest overall on clean metrics.
  - CTGAN and Adv-CTGAN remain competitive but do not beat the original regime on the strongest clean score.

## 2. Robustness

Source: `outputs/robustness_results.csv`

- Original (Imbalanced)
  - Clean Accuracy: `0.9868`
  - FGSM Accuracy: `0.7726`
  - PGD Accuracy: `0.7726`
  - Robustness Score: `0.2143`
- Standard CTGAN
  - Clean Accuracy: `0.9850`
  - FGSM Accuracy: `0.6692`
  - PGD Accuracy: `0.6692`
  - Robustness Score: `0.3158`
- Adv-CTGAN (Custom)
  - Clean Accuracy: `0.9850`
  - FGSM Accuracy: `0.7613`
  - PGD Accuracy: `0.7613`
  - Robustness Score: `0.2237`
- Main robustness takeaway:
  - Adv-CTGAN clearly improves robustness over standard CTGAN.
  - The original regime remains slightly stronger than Adv-CTGAN on the aggregate robustness score.

## 3. OOD / Shift Robustness

Source: `outputs/ood_results.csv`

- Best OOD accuracy overall: `Original + Gradient Boosting = 0.9850`
- Smallest OOD accuracy drop overall: `Original + Gradient Boosting = 0.0056`
- Best OOD result in CTGAN regime: `CTGAN + Neural Network (MLP) = 0.9455`
- Best OOD result in Adv-CTGAN regime: `Adv-CTGAN + Neural Network (MLP) = 0.9643`
- Main OOD takeaway:
  - Original-data models are most stable overall.
  - Neural-network models are the most stable within augmented regimes.

## 4. Learning Curve

Source: `outputs/learning_curve_results.csv`

- Primary classifier used: `Random Forest`
- Sample fractions tested: `20%`, `40%`, `60%`, `80%`, `100%`
- Final F1-score by regime at full data:
  - Original: `0.9980`
  - CTGAN: `0.9951`
  - Adv-CTGAN: `0.9931`
- Low-data result at 20%:
  - Original: `0.9855`
  - CTGAN: `0.9922`
  - Adv-CTGAN: `0.9922`
- Main learning-curve takeaway:
  - Synthetic augmentation helps most in the low-data setting.
  - As more real data becomes available, the original regime overtakes both augmented regimes.

## 5. Agentic Metric

Source: `outputs/agentic_metric_results.csv`

### Definition

The agentic workflow uses a lightweight Evaluation Agent and Decision Agent:

- Evaluation Agent computes:
  - Clean Utility = `0.5 * F1 + 0.3 * AUC + 0.2 * Accuracy`
  - OOD Utility = `1 - Accuracy Drop`
  - Calibration Utility = normalized inverse log loss
  - Robustness Utility = normalized inverse robustness score
  - Synthetic Utility = `0.6 * normalized diversity + 0.4 * normalized inverse FID`
- Decision Agent score:
  - Agentic Score = `0.35 * Clean Utility + 0.30 * Robustness Utility + 0.20 * OOD Utility + 0.10 * Calibration Utility + 0.05 * Synthetic Utility`

### Ranking

- Original
  - Selected classifier: `Random Forest`
  - Agentic Score: `0.9456`
  - Decision: `Deploy now`
- Adv-CTGAN
  - Selected classifier: `Random Forest`
  - Agentic Score: `0.8687`
  - Decision: `Use as robustness-aware backup`
- CTGAN
  - Selected classifier: `Neural Network (MLP)`
  - Agentic Score: `0.6276`
  - Decision: `Keep as baseline only`

### Main agentic takeaway

- The original regime is still the best deployment choice overall.
- Adv-CTGAN is the strongest augmentation-based fallback when robustness is prioritized.
