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


def _get_strategy_training_data(state: dict) -> tuple[object, list[str], str]:
    """Resolve the latest training frame Strategy should inspect."""
    train_df = state.get("balanced_train_df")
    if train_df is None:
        train_df = state.get("remediated_train_df", state.get("train_df"))
    return train_df, state["feature_cols"], state["target_col"]


def _build_model_candidate(
    model_type: str,
    hyperparameters: dict,
    positive_count: int,
    negative_count: int,
):
    """Instantiate a candidate model for quick strategy scoring."""
    if model_type == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=hyperparameters["n_estimators"],
            max_depth=hyperparameters["max_depth"],
            learning_rate=hyperparameters["learning_rate"],
            scale_pos_weight=max(1, int(negative_count / max(1, positive_count))),
            eval_metric="logloss",
            random_state=42,
        )

    if model_type == "LightGBM":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=hyperparameters["n_estimators"],
            max_depth=hyperparameters["max_depth"],
            learning_rate=hyperparameters["learning_rate"],
            class_weight=hyperparameters.get("class_weight", "balanced"),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    if model_type == "CatBoost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=hyperparameters.get("n_estimators", 100),
            depth=hyperparameters.get("max_depth", 6),
            learning_rate=hyperparameters.get("learning_rate", 0.1),
            auto_class_weights="Balanced",
            random_state=42,
            verbose=0,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def _get_candidate_specs(
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> list[dict[str, object]]:
    """Return the available model families with baseline hyperparameters."""
    candidate_specs: list[dict[str, object]] = []

    try:
        import xgboost  # noqa: F401

        candidate_specs.append(
            {
                "model_type": "XGBoost",
                "hyperparameters": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "class_weight": None,
                },
            }
        )
    except ImportError:
        pass

    try:
        import lightgbm  # noqa: F401

        candidate_specs.append(
            {
                "model_type": "LightGBM",
                "hyperparameters": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "class_weight": "balanced",
                },
            }
        )
    except ImportError:
        pass

    try:
        import catboost  # noqa: F401

        candidate_specs.append(
            {
                "model_type": "CatBoost",
                "hyperparameters": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "class_weight": "balanced",
                },
            }
        )
    except ImportError:
        pass

    return candidate_specs


def _score_model_candidates(
    state: dict,
    n_estimators: int,
    l2_count: int,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> tuple[str, dict, list[dict[str, object]]]:
    """Evaluate a small model family set and return the best candidate."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    train_df, feature_cols, target_col = _get_strategy_training_data(state)
    X = train_df[feature_cols].values
    y = train_df[target_col].values.astype(int)
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + l2_count)

    candidate_specs = _get_candidate_specs(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )

    candidate_results: list[dict[str, object]] = []
    for order, spec in enumerate(candidate_specs):
        model_type = spec["model_type"]
        hyperparameters = spec["hyperparameters"]
        try:
            model = _build_model_candidate(
                model_type=model_type,
                hyperparameters=hyperparameters,
                positive_count=positive_count,
                negative_count=negative_count,
            )
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
            recall_scores = cross_val_score(
                model, X, y, cv=cv, scoring="recall", n_jobs=-1
            )
            precision_scores = cross_val_score(
                model, X, y, cv=cv, scoring="precision", n_jobs=-1
            )
            mean_f1 = float(f1_scores.mean())
            mean_recall = float(recall_scores.mean())
            mean_precision = float(precision_scores.mean())
            score = 0.6 * mean_f1 + 0.3 * mean_recall + 0.1 * mean_precision

            candidate_results.append(
                {
                    "model_type": model_type,
                    "hyperparameters": hyperparameters,
                    "cv_f1": round(mean_f1, 4),
                    "cv_recall": round(mean_recall, 4),
                    "cv_precision": round(mean_precision, 4),
                    "selection_score": round(score, 4),
                    "status": "ok",
                    "rank_order": order,
                }
            )
        except Exception as exc:
            candidate_results.append(
                {
                    "model_type": model_type,
                    "hyperparameters": hyperparameters,
                    "status": "failed",
                    "error": str(exc),
                    "selection_score": float("-inf"),
                    "rank_order": order,
                }
            )

    valid_results = [r for r in candidate_results if r["status"] == "ok"]
    if not valid_results:
        fallback_hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "class_weight": None,
        }
        return "XGBoost", fallback_hyperparameters, candidate_results

    best = max(
        valid_results,
        key=lambda result: (result["selection_score"], -result["rank_order"]),
    )
    return best["model_type"], best["hyperparameters"], candidate_results


def _optuna_tune_selected_model(
    state: dict,
    model_type: str,
    baseline_hyperparameters: dict[str, object],
) -> dict[str, object]:
    """Tune only the already-selected model family."""
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    train_df, feature_cols, target_col = _get_strategy_training_data(state)
    X_opt = train_df[feature_cols].values
    y_opt = train_df[target_col].values.astype(int)
    positive_count = int((y_opt == 1).sum())
    negative_count = int((y_opt == 0).sum())

    def objective(trial: optuna.Trial) -> float:
        hyperparameters = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "class_weight": baseline_hyperparameters.get("class_weight"),
        }
        model = _build_model_candidate(
            model_type=model_type,
            hyperparameters=hyperparameters,
            positive_count=positive_count,
            negative_count=negative_count,
        )
        cv_opt = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(
            model, X_opt, y_opt, cv=cv_opt, scoring="f1", n_jobs=-1
        )
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    tuned_hyperparameters = {
        **baseline_hyperparameters,
        **study.best_params,
    }
    tuned_hyperparameters["optuna_best_f1"] = round(float(study.best_value), 4)
    return tuned_hyperparameters


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
    candidate_needs_evaluation = state.get(
        "candidate_needs_evaluation", candidate_model is not None
    )
    needs_strategy_refinement = state.get("needs_strategy_refinement", False)
    should_retrain = state.get("should_retrain", False)
    drift_detected = state.get("drift_detected", False)
    drift_remediation = state.get("drift_remediation", {})

    # ── KB context: read prior eval history (KB→Strategy communication) ──
    prior_eval = kb.get_latest("evaluation_history")
    kb_prior_f1 = None
    if prior_eval:
        kb_prior_f1 = prior_eval.get("data", {}).get("f1", None)
        print(f"  📖 KB context: prior eval F1={kb_prior_f1}")

    eval_f1 = eval_metrics.get("f1")
    robustness_score = simulation_results.get("robustness_score")

    # Decide the next step first. This lets Strategy skip expensive search/tuning
    # when it is only routing an existing plan to Training/Evaluation.
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

    existing_plan = state.get("strategy_plan") or {}
    reuse_existing_plan = (
        bool(existing_plan.get("model_type"))
        and bool(existing_plan.get("hyperparameters"))
        and not needs_strategy_refinement
    )

    optuna_used = False
    model_candidates = list(existing_plan.get("model_candidates") or [])
    selection_reason = existing_plan.get("selection_reason", "reused_existing_plan")
    model_type = existing_plan.get("model_type", "XGBoost")
    selected_hyperparameters = dict(existing_plan.get("hyperparameters") or {})
    noise_std = ADVERSARIAL_NOISE_STD

    if reuse_existing_plan:
        print(
            f"  ℹ️  Reusing existing strategy plan for "
            f"{strategy_decision} routing; skipping search/tuning."
        )
    else:
        # ── Base search space for model comparison ─────────────────
        if kb_prior_f1 is not None and kb_prior_f1 < 0.6:
            # KB shows prior run had low F1 — pre-emptively boost estimators
            n_estimators = N_ESTIMATORS + 50
            max_depth = 6
            learning_rate = 0.1
            print(f"  ℹ️  KB-informed boost: prior F1={kb_prior_f1} → n_estimators+50")
        else:
            n_estimators = N_ESTIMATORS
            max_depth = 6
            learning_rate = 0.1

        # ── Model selection ─────────────────────────────────────────
        model_type, candidate_hyperparameters, model_candidates = (
            _score_model_candidates(
                state=state,
                n_estimators=n_estimators,
                l2_count=l2_count,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )
        )
        selected_hyperparameters = candidate_hyperparameters

        if (
            l2_count > 0
            and needs_strategy_refinement
            and not candidate_needs_evaluation
        ):
            print(
                f"  ℹ️  L2 refinement iteration #{l2_count} — "
                f"benchmarking selected {model_type} with Optuna tuning."
            )
            selected_hyperparameters = _optuna_tune_selected_model(
                state=state,
                model_type=model_type,
                baseline_hyperparameters=candidate_hyperparameters,
            )
            optuna_used = True
            print(f"  ✅ Optuna tuned {model_type}: {selected_hyperparameters}")

        best_candidate = next(
            (
                candidate
                for candidate in model_candidates
                if candidate.get("model_type") == model_type
                and candidate.get("status") == "ok"
            ),
            None,
        )
        selection_reason = (
            f"best_cv_score={best_candidate['selection_score']}"
            if best_candidate is not None
            else "fallback_xgboost"
        )

    use_adversarial_training = (
        model_available
        and not candidate_needs_evaluation
        and robustness_score is not None
        and robustness_score < ROBUSTNESS_THRESHOLD
    ) or (drift_detected and drift_remediation.get("applied", False))

    plan = {
        "model_type": model_type,
        "hyperparameters": selected_hyperparameters,
        "model_candidates": model_candidates,
        "selection_reason": selection_reason,
        "adversarial_strategy": {
            "noise_perturbation": {"enabled": True, "std": round(noise_std, 4)},
            "boundary_attack": {"enabled": False, "k": ADVERSARIAL_BOUNDARY_K},
            "evasion_mutation": {"enabled": False, "std": round(noise_std * 0.6, 4)},
            "method": "fgsm_attack"
            if use_adversarial_training
            else "fgsm_style_random_sign",
            "use_advanced": False,
            "advanced_config": {
                "enable_apfe": False,
                "enable_asc": False,
                "enable_tla": False,
                "enable_fraudgan": False,
                "n_clusters": 5,
                "triplet_weight": 0.1,
            },
        },
        "use_adversarial_training": (
            (
                model_available
                and not candidate_needs_evaluation
                and robustness_score is not None
                and robustness_score < ROBUSTNESS_THRESHOLD
            )
            or (drift_detected and drift_remediation.get("applied", False))
        ),
        "l2_refinement": l2_count > 0,
        "l2_count": l2_count,
        "decision": strategy_decision,
        "decision_reason": decision_reason,
        "eval_f1": eval_f1,
        "robustness_score": robustness_score,
        "drift_adaptation": {
            "drift_detected": drift_detected,
            "remediation_applied": bool(drift_remediation.get("applied", False)),
            "rows_added": int(drift_remediation.get("rows_added", 0)),
        },
        "optuna_used": optuna_used,
    }

    candidate_summary = ", ".join(
        f"{candidate['model_type']}={candidate.get('selection_score')}"
        for candidate in model_candidates
    )
    print(f"  Model: {model_type} ({plan['selection_reason']})")
    print(f"  Candidate scores: {candidate_summary}")
    adv_method = plan["adversarial_strategy"]["method"]
    adv_eps = plan["adversarial_strategy"]["noise_perturbation"]["std"]
    print(f"  Adversarial: method={adv_method}, epsilon={adv_eps}")
    if use_adversarial_training:
        ac = plan["adversarial_strategy"]["advanced_config"]
        print(
            f"    Advanced techniques: APFE={ac['enable_apfe']}, ASC={ac['enable_asc']}, "
            f"TLA={ac['enable_tla']}, FraudGAN={ac['enable_fraudgan']}"
        )
    print(f"  Use adversarial strengthening: {plan['use_adversarial_training']}")
    print(f"  Next step: {strategy_decision} ({decision_reason})")

    kb.log_event("strategy", "strategy_plan", plan)

    return {
        **state,
        "strategy_plan": plan,
        "strategy_decision": strategy_decision,
    }
