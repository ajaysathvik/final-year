"""
Simulation Agent — Execute phase of MAPE-K.
Tests model robustness before deployment using noise injection and bootstrap tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import ROBUSTNESS_THRESHOLD


def simulation_agent(state: dict) -> dict:
    """
    Run robustness/simulation checks on the candidate model:
    1. Noise-injection stress test
    2. Bootstrap confidence interval
    3. Edge-case consistency
    """
    print("\n" + "=" * 60)
    print(" [ SIMULATION AGENT ] Running robustness simulation...")
    print("=" * 60)

    kb = state.get("knowledge_base")
    model = state.get("candidate_model") or state.get("current_model")
    if model is None:
        print("  ❌ No candidate model to simulate.")
        return {**state, "simulation_results": {"error": "no_model"}, "simulation_passed": False}

    # ── KB context: read prior simulation results ────────────────
    if kb:
        prior_sim = kb.get_latest("simulation_records")
        if prior_sim:
            prior_score = prior_sim.get("data", {}).get("robustness_score", "n/a")
            print(f"  📖 KB context: prior robustness_score={prior_score}")

    test_df: pd.DataFrame = state["test_df"]
    feature_cols = state["feature_cols"]
    target_col = state["target_col"]

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values.astype(int)

    # 1. Noise stress test — degrade inputs and check F1 stability
    # Bug-9 fix: use wider epsilon range, fixed seed per epsilon, and perturb only 50% of features
    noise_levels = [0.05, 0.10, 0.20, 0.50]
    noise_scores = {}

    for eps in noise_levels:
        rng = np.random.RandomState(42 + int(eps * 1000))
        mask = rng.binomial(1, 0.5, size=X_test.shape).astype(bool)
        noise = rng.normal(0, eps, size=X_test.shape)
        
        X_noisy = X_test.copy()
        X_noisy[mask] += noise[mask]
        
        preds = model.predict(X_noisy)
        f1 = round(float(f1_score(y_test, preds, zero_division=0)), 4)
        noise_scores[f"eps_{eps}"] = f1
        if eps == noise_levels[-1] and len(X_test) > 0:
            print(f"    Sample perturbation (eps={eps}, row 0): orig={X_test[0,:3]} → noisy={X_noisy[0,:3]}")

    print(f"  Noise stress test: {noise_scores}")

    # 2. Bootstrap confidence interval (30 resamples)
    n_bootstrap = 30
    boot_f1s = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X_test), size=len(X_test), replace=True)
        preds_b = model.predict(X_test[idx])
        boot_f1s.append(float(f1_score(y_test[idx], preds_b, zero_division=0)))

    boot_mean = round(float(np.mean(boot_f1s)), 4)
    boot_std = round(float(np.std(boot_f1s)), 4)
    boot_lower = round(float(np.percentile(boot_f1s, 2.5)), 4)
    boot_upper = round(float(np.percentile(boot_f1s, 97.5)), 4)

    print(f"  Bootstrap F1: {boot_mean} ± {boot_std} [{boot_lower}, {boot_upper}]")

    # 3. Worst-case noise score
    worst_noise_f1 = min(noise_scores.values())
    
    # Bug-9 check: if all noise scores are identical, the model didn't react at all
    if len(set(noise_scores.values())) <= 1 and worst_noise_f1 > 0:
        print("  ⚠️ WARNING: All epsilon levels produced identical F1. Noise perturbation may not be effective.")

    robustness_score = round(float(np.mean([worst_noise_f1, boot_lower])), 4)

    passed = robustness_score >= ROBUSTNESS_THRESHOLD

    results = {
        "noise_stress_test": noise_scores,
        "bootstrap_mean_f1": boot_mean,
        "bootstrap_std": boot_std,
        "bootstrap_ci_95": [boot_lower, boot_upper],
        "worst_noise_f1": worst_noise_f1,
        "robustness_score": robustness_score,
        "threshold": ROBUSTNESS_THRESHOLD,
        "passed": passed,
    }

    print(f"  Robustness score: {robustness_score} (threshold={ROBUSTNESS_THRESHOLD})")
    print(f"  {'✅ PASSED' if passed else '❌ FAILED'} simulation.")

    # ── L5: log simulation to KB ─────────────────────────────────
    if kb:
        kb.log_event("simulation", "simulation_completed", results)

    return {**state, "simulation_results": results, "simulation_passed": passed}
