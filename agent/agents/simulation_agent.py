"""
Simulation Agent — Execute phase of MAPE-K.
Tests model robustness before deployment using FGSM-style perturbation and bootstrap tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import ROBUSTNESS_THRESHOLD


def _fgsm_perturb(X: np.ndarray, epsilon: float, seed: int) -> np.ndarray:
    """Match the reference FGSM-style random-sign perturbation used in training."""
    if epsilon <= 0 or X.size == 0:
        return X.copy()

    rng = np.random.RandomState(seed)
    perturbed = X.copy().astype(np.float32)
    noise = epsilon * np.sign(rng.randn(*perturbed.shape))
    perturbed += noise
    return np.clip(perturbed, 0.0, None).astype(np.float32)


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

    # 1. FGSM-style robustness curve — evaluate across increasing epsilon
    epsilons = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    attack_scores = {}

    for eps in epsilons:
        X_attacked = _fgsm_perturb(X_test, epsilon=eps, seed=42 + int(eps * 1000))
        preds = model.predict(X_attacked)
        f1 = round(float(f1_score(y_test, preds, zero_division=0)), 4)
        attack_scores[f"eps_{eps}"] = f1
        if eps == epsilons[-1] and len(X_test) > 0:
            print(f"    Sample perturbation (eps={eps}, row 0): orig={X_test[0,:3]} → attacked={X_attacked[0,:3]}")

    print(f"  FGSM-style robustness curve: {attack_scores}")

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

    # 3. Worst-case attacked score
    attacked_scores = [score for key, score in attack_scores.items() if key != "eps_0.0"]
    worst_noise_f1 = min(attacked_scores) if attacked_scores else min(attack_scores.values())

    if len(set(attack_scores.values())) <= 1 and worst_noise_f1 > 0:
        print("  ⚠️ WARNING: All epsilon levels produced identical F1. FGSM-style perturbation may not be effective.")

    robustness_score = round(float(np.mean([worst_noise_f1, boot_lower])), 4)

    passed = robustness_score >= ROBUSTNESS_THRESHOLD

    results = {
        "noise_stress_test": attack_scores,
        "attack_method": "fgsm_style_random_sign",
        "attack_epsilons": epsilons,
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
