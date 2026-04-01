"""
Strategy Agent — Plan phase of MAPE-K (Planning / Adaptation).
Handles model selection, hyperparameters, and routing decisions between
training and evaluation, with adversarial strengthening folded into training.
"""
from __future__ import annotations

from config import (
    N_ESTIMATORS,
    ADVERSARIAL_NOISE_STD,
    ADVERSARIAL_BOUNDARY_K,
    F1_THRESHOLD,
    ROBUSTNESS_THRESHOLD,
)


def strategy_agent(state: dict) -> dict:
    """
    Build a concrete execution plan and decide the next execution step:
    - training when predictive quality is weak or robustness needs strengthening
    - evaluation when the current candidate should be scored
    """
    print("\n" + "=" * 60)
    print(" [ STRATEGY AGENT ] Planning model & adversarial strategy...")
    print("=" * 60)

    kb = state["knowledge_base"]
    l2_count = state.get("l2_count", 0)
    eval_metrics = state.get("eval_metrics") or {}
    simulation_results = state.get("simulation_results") or {}
    candidate_model = state.get("candidate_model")
    current_model = state.get("current_model")
    model_available = candidate_model is not None or current_model is not None
    candidate_needs_evaluation = state.get("candidate_needs_evaluation", candidate_model is not None)
    should_retrain = state.get("should_retrain", False)

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

    eval_f1 = eval_metrics.get("f1")
    robustness_score = simulation_results.get("robustness_score")

    if not model_available:
        strategy_decision = "training"
        decision_reason = "no_model_available"
    elif should_retrain and not candidate_needs_evaluation:
        strategy_decision = "training"
        decision_reason = "policy_requested_retraining"
    elif candidate_needs_evaluation:
        strategy_decision = "evaluation"
        decision_reason = "candidate_requires_evaluation"
    elif eval_f1 < F1_THRESHOLD:
        strategy_decision = "training"
        decision_reason = f"f1_below_threshold:{eval_f1:.4f}<{F1_THRESHOLD}"
    elif robustness_score < ROBUSTNESS_THRESHOLD:
        strategy_decision = "training"
        decision_reason = (
            f"robustness_below_threshold:{robustness_score:.4f}<{ROBUSTNESS_THRESHOLD}"
        )
    else:
        strategy_decision = "evaluation"
        decision_reason = "evaluation_refresh"

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
        "use_adversarial_training": (
            model_available
            and not candidate_needs_evaluation
            and robustness_score is not None
            and robustness_score < ROBUSTNESS_THRESHOLD
        ),
        "l2_refinement": l2_count > 0,
        "l2_count": l2_count,
        "decision": strategy_decision,
        "decision_reason": decision_reason,
        "eval_f1": eval_f1,
        "robustness_score": robustness_score,
    }

    print(f"  Model: {model_type} (n_estimators={n_estimators})")
    print(f"  Adversarial: noise_std={plan['adversarial_strategy']['noise_perturbation']['std']}, "
          f"boundary_k={ADVERSARIAL_BOUNDARY_K}")
    print(f"  Use adversarial strengthening: {plan['use_adversarial_training']}")
    print(f"  Next step: {strategy_decision} ({decision_reason})")

    kb.log_event("strategy", "strategy_plan", plan)

    return {
        **state,
        "strategy_plan": plan,
        "strategy_decision": strategy_decision,
    }
