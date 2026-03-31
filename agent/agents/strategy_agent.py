"""
Strategy Agent — Plan phase of MAPE-K (Planning / Adaptation).
Handles: Model selection, Hyperparameters, Adversarial strategy.
Feeds both Training Agent and Evaluation Agent.
L2 feedback loop: Evaluation → Strategy (refinement).
"""
from __future__ import annotations

from config import N_ESTIMATORS, ADVERSARIAL_NOISE_STD, ADVERSARIAL_BOUNDARY_K


def strategy_agent(state: dict) -> dict:
    """
    Build a concrete execution plan including:
    - Which model to train (XGBoost preferred, RandomForest fallback)
    - Hyperparameters to use
    - Adversarial strategy (attack types and parameters)
    Adapts on L2 feedback from Evaluation.
    """
    print("\n" + "=" * 60)
    print(" [ STRATEGY AGENT ] Planning model & adversarial strategy...")
    print("=" * 60)

    kb = state["knowledge_base"]
    l2_count = state.get("l2_count", 0)
    eval_metrics = state.get("eval_metrics", {})

    # ── KB context: read prior eval history (KB→Strategy communication) ──
    prior_eval = kb.get_latest("evaluation_history")
    kb_prior_f1 = None
    if prior_eval:
        kb_prior_f1 = prior_eval.get("data", {}).get("f1", None)
        print(f"  📖 KB context: prior eval F1={kb_prior_f1}")

    # ── Check if this is an L2 refinement ──────────────────────
    if l2_count > 0:
        print(f"  ℹ️  L2 refinement iteration #{l2_count}")
        # Adapt: increase model complexity or change adversarial params
        n_estimators = N_ESTIMATORS + (50 * l2_count)
        noise_std = ADVERSARIAL_NOISE_STD * (1 + 0.5 * l2_count)
    elif kb_prior_f1 is not None and kb_prior_f1 < 0.6:
        # KB shows prior run had low F1 — pre-emptively boost estimators
        n_estimators = N_ESTIMATORS + 50
        noise_std = ADVERSARIAL_NOISE_STD
        print(f"  ℹ️  KB-informed boost: prior F1={kb_prior_f1} → n_estimators+50")
    else:
        n_estimators = N_ESTIMATORS
        noise_std = ADVERSARIAL_NOISE_STD

    # ── Model selection ─────────────────────────────────────────
    try:
        import xgboost  # noqa: F401
        model_type = "XGBoost"
    except ImportError:
        model_type = "RandomForest"

    plan = {
        "model_type": model_type,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": 6 if model_type == "XGBoost" else 10,
            "learning_rate": 0.1 if model_type == "XGBoost" else None,
            "class_weight": None if model_type == "XGBoost" else "balanced",
        },
        "adversarial_strategy": {
            "noise_perturbation": {"enabled": True, "std": round(noise_std, 4)},
            "boundary_attack": {"enabled": True, "k": ADVERSARIAL_BOUNDARY_K},
            "evasion_mutation": {"enabled": True, "std": round(noise_std * 0.6, 4)},
        },
        "l2_refinement": l2_count > 0,
        "l2_count": l2_count,
    }

    print(f"  Model: {model_type} (n_estimators={n_estimators})")
    print(f"  Adversarial: noise_std={plan['adversarial_strategy']['noise_perturbation']['std']}, "
          f"boundary_k={ADVERSARIAL_BOUNDARY_K}")

    kb.log_event("strategy", "strategy_plan", plan)

    return {**state, "strategy_plan": plan}
