# Agentic Fraud Pipeline — Run Summary

**Status:** STOPPED  
**Total iterations:** 8  
**Generated at:** 2026-03-30 09:21:18 UTC  

## 1. Scraping (Ingestion Agent)

| Metric | Value |
|--------|-------|
| Posts before | 3867 |
| Posts after | 3982 |
| New posts detected | 115 |
| Comments after | 24203 |
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

### Drift Review

| Metric | Value |
|--------|-------|
| Accepted | True |
| Retries exhausted | False |

## 3. Balancing (CTGAN)

| Metric | Value |
|--------|-------|
| Best ratio | 15 |
| CTGAN return code | 1 |
| Synthetic rows | N/A |
| Mean JSD | 1.0 |
| Quality accepted | False |
| Balance accepted | False |

## 4. Classifier Training

| Metric | Value |
|--------|-------|
| Return code | N/A |
| Best F1 | N/A |
| Passed | N/A |

## 5. Supervisor / Investigation / Policy

| Metric | Value |
|--------|-------|
| Route lane | N/A |
| Escalation score | N/A |
| Recommended action | N/A |
| Policy mode | N/A |
| Policy approved | N/A |

## 6. Adversarial Training (Strategy Agent)

| Metric | Value |
|--------|-------|
| Focus | N/A |
| Recommended focus | N/A |
| Training return code | N/A |
| Robustness return code | N/A |
| Baseline FGSM F1 | N/A |
| Adversarial FGSM F1 | N/A |
| Adversarial Clean F1 | N/A |
| Robustness gain (Δ) | N/A |
| Accepted | N/A |

## 7. Final Evaluation

| Metric | Value |
|--------|-------|
| Eval return code | N/A |
| Robustness return code | N/A |
| Best F1 | N/A |
| Robustness score | N/A |
| Passed | N/A |
| Correction target | N/A |

## 8. Simulation Gate

| Metric | Value |
|--------|-------|
| Proposal mode | N/A |
| Simulation score | N/A |
| Guardrail status | N/A |
| Approved | N/A |

## Agent Decisions Timeline

| # | Agent | Action | Confidence | Summary |
|---|-------|--------|------------|---------|
| 1 | ingestion_agent | ready_for_balancing | 0.95 | Ingestion completed successfully with 115 new Reddit posts scraped. Labeling phase encountered critical failures for 3 o |
| 2 | drift_agent | continue_to_balance | 0.96 | DriftAgent accepted dataset health. label_noise_score=0.0697, slang_drift_score=0.0, recommendation=keep. |
| 3 | balance_agent | retry_balancing | 0.99 | BalanceAgent rejected balancing output. Best ratio=15. mean JSD=1.0 exceeded threshold=0.2. |
| 4 | balance_agent | retry_balancing | 0.99 | BalanceAgent rejected balancing output. Best ratio=15. mean JSD=1.0 exceeded threshold=0.2. |
| 5 | balance_agent | retry_balancing | 0.99 | BalanceAgent rejected balancing output. Best ratio=15. mean JSD=1.0 exceeded threshold=0.2. |
| 6 | balance_agent | retry_balancing | 0.99 | BalanceAgent rejected balancing output. Best ratio=15. mean JSD=1.0 exceeded threshold=0.2. |
| 7 | balance_agent | retry_balancing | 0.99 | BalanceAgent rejected balancing output. Best ratio=15. mean JSD=1.0 exceeded threshold=0.2. |
| 8 | balance_agent | retry_balancing | 0.98 | BalanceAgent rejected balancing output. Best ratio=15. mean JSD=1.0 exceeded threshold=0.2. |

## Adversarial Training Charts

Charts copied to `output/adversarial/charts/`:

- `chart1_all_metrics.png`
- `chart2_f1_robustness.png`
- `chart3_feature_perturbation.png`
- `chart4_feature_importance.png`
- `robustness_curve.png`
