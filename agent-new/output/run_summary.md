# Agentic Fraud Pipeline — Run Summary

**Status:** COMPLETED  
**Total iterations:** 5  
**Generated at:** 2026-03-29 10:57:53 UTC  

## 1. Scraping (Ingestion Agent)

| Metric | Value |
|--------|-------|
| Posts before | 3699 |
| Posts after | 3703 |
| New posts detected | 4 |
| Comments after | 22941 |
| Smoke test | False |
| Return code | 0 |

## 2. Labeling & Post-Processing

| Metric | Value |
|--------|-------|
| Labeling skipped | False |
| New posts to label | 10 |
| New labels created | 0 |
| Post-processing skipped | True |
| Processed rows | 0 |
| Label return code | 0 |

### Dataset Profile

| Metric | Value |
|--------|-------|
| Train rows | 4549 |
| Test rows | 1177 |
| Label noise score | 0.0697 |
| Slang drift score | 0.0 |
| Label review recommendation | keep |
| Ambiguous ratio | 0.04 |

## 3. Balancing (CTGAN)

| Metric | Value |
|--------|-------|
| Best ratio | 5 |
| CTGAN return code | 0 |
| Synthetic rows | 793 |
| Mean JSD | 0.1291 |
| Quality accepted | True |
| Balance accepted | True |

## 4. Classifier Training

| Metric | Value |
|--------|-------|
| Return code | 0 |
| Best F1 | 0.9665 |
| Passed | True |

## 5. Adversarial Training (Strategy Agent)

| Metric | Value |
|--------|-------|
| Focus | generic |
| Recommended focus | transactional short-form fraud |
| Training return code | 0 |
| Robustness return code | 0 |
| Baseline FGSM F1 | 0.9851 |
| Adversarial FGSM F1 | 0.9878 |
| Adversarial Clean F1 | 0.9878 |
| Robustness gain (Δ) | 0.0027 |
| Accepted | True |

### Attack Surface

| Fraud Channel | Count |
|---------------|-------|
| website | 286 |
| email | 222 |
| sms | 220 |
| phone | 185 |
| social_media | 104 |
| unknown | 44 |
| telegram | 35 |
| none | 20 |
| app | 16 |
| sms | email | phone | social_media | website | app | unknown | none | 11 |

## 6. Final Evaluation

| Metric | Value |
|--------|-------|
| Eval return code | 0 |
| Robustness return code | 0 |
| Best F1 | 0.9665 |
| Robustness score | 0.9878 |
| Passed | True |
| Correction target | complete |

## Agent Decisions Timeline

| # | Agent | Action | Confidence | Summary |
|---|-------|--------|------------|---------|
| 1 | ingestion_agent | ready_for_balancing | 0.95 | Ingestion completed successfully with 4 new posts scraped. Labeling phase encountered critical failures for 3 out of 10  |
| 2 | balance_agent | ready_for_strategy | 0.6 | BalanceAgent accepted CTGAN output. Best ratio=5, mean JSD=0.1291, threshold=0.2. |
| 3 | training_agent | ready_for_strategy | 0.6 | TrainingAgent passed base-model validation. best_f1=0.9665, required>=0.738. |
| 4 | strategy_agent | ready_for_evaluation | 0.99 | StrategyAgent accepted adversarial training. focus=generic, robustness_gain=0.0027, adversarial_clean_f1=0.9878. |
| 5 | evaluation_agent | deploy | 0.99 | EvaluationAgent passed final validation. best_f1=0.9665 vs threshold=0.82, non_fraud_f1=0.6105 vs threshold=0.5, robustn |

## Adversarial Training Charts

Charts copied to `output/adversarial/charts/`:

- `chart1_all_metrics.png`
- `chart2_f1_robustness.png`
- `chart3_feature_perturbation.png`
- `chart4_feature_importance.png`
- `robustness_curve.png`
