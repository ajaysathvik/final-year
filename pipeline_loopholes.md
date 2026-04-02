# MAPE-K Fraud Detection Pipeline — Loopholes & Issues

## Critical Bugs (Red)

### Bug 1 — Drift window too small (Monitor)
- **Where:** `drift_agent`, scraped data window
- **Issue:** PSI/KS statistics computed on only **24 rows** vs. 7,702 training rows. This sample size is statistically insufficient — PSI and KS tests are unreliable below ~500 samples. Drift conclusions drawn here are effectively noise.
- **Fix:** Buffer scraped data until a minimum threshold (e.g. 500 rows) before triggering drift analysis, or use an online drift detector (e.g. `river.drift.ADWIN`) that is designed for small-batch streaming.

---

### Bug 2 — Static test set never refreshed (Analyze)
- **Where:** `evaluation_agent`
- **Issue:** The held-out test set is fixed at the original 1,926 rows from the initial train/test split. After drift is detected and new scraped data is appended, the test set still reflects the old distribution. Evaluation metrics no longer measure real-world performance.
- **Fix:** Rebuild the test set each run by sampling from the most recent data window, or maintain a rolling held-out set that tracks the current distribution.

---

### Bug 5 — Severe class imbalance after augmentation (Execute)
- **Where:** `training_agent`, post-augmentation data
- **Issue:** After adversarial augmentation, training data is `fraud=34,856 / non-fraud=3,369` — a **~10:1 ratio**. Yet `balance_agent` reports no action needed (ratio=1.29), which means it checked the *pre-augmentation* data. The model is trained on a heavily skewed set, which inflates recall but suppresses precision (precision=0.79, FPR=0.34).
- **Fix:** Re-run balance check *after* augmentation, not before. The balance gate must see the final training set that goes into `training_agent`.

---

### Bug 7 — L1 loop retrain ignores degraded F1 (L1 loop)
- **Where:** `policy_agent` inside the L1 feedback loop
- **Issue:** After the first retrain, F1 drops from 0.8982 → 0.8724. The policy agent detects this degradation ("⬇ degrading") but still triggers another retrain with **identical settings** — same model, same hyperparameters, same augmentation strategy. The second model scores F1=0.8705, slightly worse. The system then promotes this model anyway.
- **Fix:** When the L1 loop detects degradation, the policy must escalate — either halt and fall back to the previously promoted model, or signal the strategy agent to change hyperparameters / model type before retrying.

---

### Bug 9 — Noise stress test F1 is identical at all epsilon levels (Simulate)
- **Where:** `simulation_agent`, noise stress test
- **Issue:** `eps_0.01 = eps_0.03 = eps_0.05 = eps_0.1 = 0.7215` — all four epsilon levels produce the exact same F1. This is a strong indicator that the noise perturbation is not actually being applied to the test features, or the model's decision boundary is completely insensitive to the noise range tested. The robustness test is giving false assurance.
- **Fix:** Log intermediate perturbed samples to confirm noise is being applied. Try wider epsilon values (e.g. 0.2, 0.5). Also test feature-targeted perturbation rather than uniform Gaussian noise.

---

## Design Weaknesses (Amber)

### Bug 3 — Drift always triggers retrain regardless of F1 (Plan)
- **Where:** `policy_agent`
- **Issue:** The sole retrain trigger is `drift_detected=True`. There is no gate on whether the current model's F1 is still acceptable. If F1=0.95 and only minor covariate drift is present, a retrain is wasteful and risks introducing a worse model — which is exactly what happened this run.
- **Fix:** Add a combined condition: retrain only if `drift_detected AND (current_f1 < threshold OR f1_trend == degrading)`. Drift alone should trigger monitoring escalation, not automatic retraining.

---

### Bug 4 — Strategy agent is fully static (Plan)
- **Where:** `strategy_agent`
- **Issue:** The strategy is hardcoded: always `XGBoost n_estimators=200`, always `noise_std=0.05`, always `boundary_k=5`. It reads from the KB but never adapts based on what it finds there. Even when the KB shows a degrading F1 trend, the strategy is unchanged.
- **Fix:** Implement basic adaptive rules in the strategy agent — e.g. increase `n_estimators` after degradation, try a different model type (RandomForest) after two consecutive F1 drops, or reduce adversarial noise if FPR is rising.

---

### Bug 6 — Adversarial samples only use noise perturbation (Execute)
- **Where:** `adversarial_agent`
- **Issue:** Despite the spec requiring noise perturbation + boundary attack + evasion-style mutation, the log shows only `noise_std=0.05` and `boundary_k=5`. There is no evidence of evasion-style mutation in the output. The adversarial strengthening may be less diverse than expected, reducing its effectiveness against real evasion attacks.
- **Fix:** Confirm all three adversarial strategies are generating distinct sample sets. Log the count of samples from each method separately (e.g. `noise=10000, boundary=10000, evasion=10499`).

---

### Bug 8 — L1 loop retrains with zero hyperparameter change (L1 loop)
- **Where:** `training_agent` on second iteration
- **Issue:** The L1 loop's purpose is to adapt. Both iterations train the exact same XGBoost model with identical settings, producing nearly identical results (F1: 0.8724 → 0.8705). The loop burned compute without improving anything.
- **Fix:** The loop must modify at least one variable between iterations — hyperparameters, augmentation volume, feature selection, or model type. Otherwise it should exit early rather than repeat.

---

## Summary Table

| # | Severity | Stage | Issue |
|---|----------|-------|-------|
| 1 | Critical | Monitor | Drift computed on 24 rows — statistically unreliable |
| 2 | Critical | Analyze | Test set frozen at original distribution |
| 3 | Amber | Plan | Drift alone triggers retrain with no F1 gate |
| 4 | Amber | Plan | Strategy agent never adapts despite KB history |
| 5 | Critical | Execute | Balance check runs pre-augmentation; 10:1 imbalance ignored |
| 6 | Amber | Execute | Adversarial only confirmed for noise; evasion unverified |
| 7 | Critical | L1 Loop | Degrading F1 triggers identical retrain instead of escalation |
| 8 | Amber | L1 Loop | Second iteration changes nothing — loop is useless |
| 9 | Critical | Simulate | Stress test F1 identical at all epsilon levels — test broken |
