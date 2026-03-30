# Agentic Fraud Pipeline — Run Summary

**Status:** STOPPED  
**Total iterations:** 7  
**Generated at:** 2026-03-30 08:50:48 UTC  

## 1. Scraping (Ingestion Agent)

| Metric | Value |
|--------|-------|
| Posts before | N/A |
| Posts after | N/A |
| New posts detected | 0 |
| Comments after | N/A |
| Smoke test | N/A |
| Return code | N/A |

## 2. Labeling & Post-Processing

| Metric | Value |
|--------|-------|
| Labeling skipped | False |
| New posts to label | 0 |
| New labels created | 0 |
| Post-processing skipped | False |
| Processed rows | N/A |
| Label return code | N/A |

### Dataset Profile

| Metric | Value |
|--------|-------|
| Train rows | 3000 |
| Test rows | 600 |
| Label noise score | 0.04 |
| Slang drift score | 0.03 |
| Label review recommendation | continue |
| Ambiguous ratio | 0.06 |

### Drift Review

| Metric | Value |
|--------|-------|
| Accepted | True |
| Retries exhausted | False |

## 3. Balancing (CTGAN)

| Metric | Value |
|--------|-------|
| Best ratio | 10 |
| CTGAN return code | 0 |
| Synthetic rows | 100 |
| Mean JSD | 0.05 |
| Quality accepted | True |
| Balance accepted | True |

## 4. Classifier Training

| Metric | Value |
|--------|-------|
| Return code | 0 |
| Best F1 | 0.93 |
| Passed | True |

## 5. Supervisor / Investigation / Policy

| Metric | Value |
|--------|-------|
| Route lane | policy_agent |
| Escalation score | N/A |
| Recommended action | N/A |
| Policy mode | standard |
| Policy approved | True |

## 6. Adversarial Training (Strategy Agent)

| Metric | Value |
|--------|-------|
| Focus | generic |
| Recommended focus | transactional short-form fraud |
| Training return code | 0 |
| Robustness return code | 0 |
| Baseline FGSM F1 | 0.91 |
| Adversarial FGSM F1 | 0.94 |
| Adversarial Clean F1 | 0.95 |
| Robustness gain (Δ) | 0.03 |
| Accepted | True |

### Attack Surface

| Fraud Channel | Count |
|---------------|-------|
| website | 12 |
| email | 8 |

## 7. Final Evaluation

| Metric | Value |
|--------|-------|
| Eval return code | 0 |
| Robustness return code | 0 |
| Best F1 | 0.93 |
| Robustness score | 0.94 |
| Passed | True |
| Correction target | complete |

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
| 1 | drift_agent | continue_to_balance | 0.94 | DriftAgent accepted dataset health. label_noise_score=0.04, slang_drift_score=0.03, recommendation=continue. |
| 2 | balance_agent | ready_for_strategy | 0.85 | BalanceAgent accepted the batch without CTGAN because no synthetic non-fraud rows were required. Best ratio=10. |
| 3 | training_agent | ready_for_strategy | 0.95 | TrainingAgent passed base-model validation. best_f1=0.93, required>=0.738. |
| 4 | supervisor_agent | policy_agent | 0.98 | SupervisorAgent routed the workflow to policy_agent. base_model_f1=0.93, drift=0.03, synthetic_jsd=0.05. |
| 5 | policy_agent | strategy_agent | 0.98 | PolicyAgent approved proposal. action=advance_pipeline, mode=standard, needs_adversarial_training=False. |
| 6 | strategy_agent | ready_for_evaluation | 0.95 | StrategyAgent accepted adversarial training. focus=generic, robustness_gain=0.03, adversarial_clean_f1=0.95. |
| 7 | evaluation_agent | complete | 0.65 | EvaluationAgent passed final validation. best_f1=0.93 vs threshold=0.82, non_fraud_f1=0.72 vs threshold=0.5, robustness_ |

## Adversarial Training Charts

Charts copied to `output/adversarial/charts/`:

- `chart1_all_metrics.png`
- `chart2_f1_robustness.png`
- `chart3_feature_perturbation.png`
- `chart4_feature_importance.png`
- `robustness_curve.png`
