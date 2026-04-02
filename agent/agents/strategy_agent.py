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
    OPTUNA_MAX_TRIALS,
    OPTUNA_TIMEOUT_SECONDS,
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

    if model_type == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=hyperparameters["n_estimators"],
            max_depth=hyperparameters["max_depth"],
            class_weight=hyperparameters.get("class_weight", "balanced"),
            random_state=42,
            n_jobs=-1,
        )

    if model_type == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=hyperparameters["C"],
            max_iter=hyperparameters["max_iter"],
            class_weight=hyperparameters.get("class_weight", "balanced"),
            solver=hyperparameters.get("solver", "liblinear"),
            random_state=42,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def _score_model_candidates(
    state: dict,
    n_estimators: int,
    l2_count: int,
) -> tuple[str, dict, list[dict[str, object]]]:
    """Evaluate a small model family set and return the best candidate."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    train_df, feature_cols, target_col = _get_strategy_training_data(state)
    X = train_df[feature_cols].values
    y = train_df[target_col].values.astype(int)
    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + l2_count)

    candidate_specs = []

    try:
        import xgboost  # noqa: F401

        candidate_specs.append({
            "model_type": "XGBoost",
            "hyperparameters": {
                "n_estimators": n_estimators,
                "max_depth": 6,
                "learning_rate": 0.1,
                "class_weight": None,
            },
        })
    except ImportError:
        pass

    candidate_specs.extend([
        {
            "model_type": "RandomForest",
            "hyperparameters": {
                "n_estimators": n_estimators,
                "max_depth": 10,
                "learning_rate": None,
                "class_weight": "balanced",
            },
        },
        {
            "model_type": "LogisticRegression",
            "hyperparameters": {
                "n_estimators": None,
                "max_depth": None,
                "learning_rate": None,
                "class_weight": "balanced",
                "C": 1.0,
                "max_iter": 1000,
                "solver": "liblinear",
            },
        },
    ])

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
            recall_scores = cross_val_score(model, X, y, cv=cv, scoring="recall", n_jobs=-1)
            precision_scores = cross_val_score(model, X, y, cv=cv, scoring="precision", n_jobs=-1)
            mean_f1 = float(f1_scores.mean())
            mean_recall = float(recall_scores.mean())
            mean_precision = float(precision_scores.mean())
            score = 0.6 * mean_f1 + 0.3 * mean_recall + 0.1 * mean_precision

            candidate_results.append({
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "cv_f1": round(mean_f1, 4),
                "cv_recall": round(mean_recall, 4),
                "cv_precision": round(mean_precision, 4),
                "selection_score": round(score, 4),
                "status": "ok",
                "rank_order": order,
            })
        except Exception as exc:
            candidate_results.append({
                "model_type": model_type,
                "hyperparameters": hyperparameters,
                "status": "failed",
                "error": str(exc),
                "selection_score": float("-inf"),
                "rank_order": order,
            })

    valid_results = [r for r in candidate_results if r["status"] == "ok"]
    if not valid_results:
        fallback_hyperparameters = {
            "n_estimators": n_estimators,
            "max_depth": 10,
            "learning_rate": None,
            "class_weight": "balanced",
        }
        return "RandomForest", fallback_hyperparameters, candidate_results

    best = max(valid_results, key=lambda result: (result["selection_score"], -result["rank_order"]))
    return best["model_type"], best["hyperparameters"], candidate_results


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

    # ── Check if this is an L2 refinement ──────────────────────
    optuna_used = False
    tuned_model_type: str | None = None
    selected_hyperparameters: dict[str, object] | None = None
    if l2_count > 0 and needs_strategy_refinement and not candidate_needs_evaluation:
        print(
            f"  ℹ️  L2 refinement iteration #{l2_count} "
            f"— using Optuna HPO ({OPTUNA_MAX_TRIALS} trials, {OPTUNA_TIMEOUT_SECONDS}s timeout)"
        )
        # Bug-7 fix: use Optuna to find optimal hyperparameters
        try:
            import optuna
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # Get training data for HPO validation
            train_df, feature_cols, target_col = _get_strategy_training_data(state)
            X = train_df[feature_cols].values
            y = train_df[target_col].values.astype(int)

            # Determine model type
            try:
                import xgboost  # noqa: F401
                use_xgb = True
            except ImportError:
                use_xgb = False

            def objective(trial):
                n_est = trial.suggest_int("n_estimators", 100, 500, step=50)
                max_d = trial.suggest_int("max_depth", 3, 12)
                if use_xgb:
                    from xgboost import XGBClassifier
                    lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                    clf = XGBClassifier(
                        n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                        eval_metric="logloss", random_state=42,
                        scale_pos_weight=max(1, int((y == 0).sum() / max(1, (y == 1).sum()))),
                    )
                else:
                    clf = RandomForestClassifier(
                        n_estimators=n_est, max_depth=max_d,
                        class_weight="balanced", random_state=42, n_jobs=-1,
                    )
                scores = cross_val_score(clf, X, y, cv=3, scoring="f1", n_jobs=-1)
                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective,
                n_trials=OPTUNA_MAX_TRIALS,
                timeout=OPTUNA_TIMEOUT_SECONDS,
                show_progress_bar=False,
            )

            best = study.best_params
            n_estimators = best["n_estimators"]
            noise_std = ADVERSARIAL_NOISE_STD * (1 + 0.5 * l2_count)
            selected_hyperparameters = {
                "n_estimators": n_estimators,
                "max_depth": best["max_depth"],
                "learning_rate": best.get("learning_rate", 0.1) if use_xgb else None,
                "class_weight": None if use_xgb else "balanced",
            }
            tuned_model_type = "XGBoost" if use_xgb else "RandomForest"
            optuna_used = True
            print(
                f"  🔬 Optuna selected {tuned_model_type} "
                f"(best F1={study.best_value:.4f}, params={best})"
            )

        except ImportError:
            print("  ⚠️  Optuna not installed. Falling back to static L2 adaptation.")
            n_estimators = N_ESTIMATORS + (50 * l2_count)
            noise_std = ADVERSARIAL_NOISE_STD * (1 + 0.5 * l2_count)
        except Exception as exc:
            print(f"  ⚠️  Optuna HPO failed ({exc}). Falling back to static L2 adaptation.")
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
    model_type, candidate_hyperparameters, model_candidates = _score_model_candidates(
        state=state,
        n_estimators=n_estimators,
        l2_count=l2_count,
    )

    if selected_hyperparameters is None:
        selected_hyperparameters = candidate_hyperparameters
    elif tuned_model_type == model_type:
        selected_hyperparameters = {
            **candidate_hyperparameters,
            **selected_hyperparameters,
        }
    else:
        selected_hyperparameters = candidate_hyperparameters

    best_candidate = next(
        (candidate for candidate in model_candidates if candidate.get("model_type") == model_type and candidate.get("status") == "ok"),
        None,
    )

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
        "hyperparameters": selected_hyperparameters,
        "model_candidates": model_candidates,
        "selection_reason": (
            f"best_cv_score={best_candidate['selection_score']}"
            if best_candidate is not None
            else "fallback_random_forest"
        ),
        "adversarial_strategy": {
            "noise_perturbation": {"enabled": True, "std": round(noise_std, 4)},
            "boundary_attack": {"enabled": False, "k": ADVERSARIAL_BOUNDARY_K},
            "evasion_mutation": {"enabled": False, "std": round(noise_std * 0.6, 4)},
            "method": "fgsm_style_random_sign",
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
    print(
        f"  Adversarial: method={plan['adversarial_strategy']['method']}, "
        f"epsilon={plan['adversarial_strategy']['noise_perturbation']['std']}"
    )
    print(f"  Use adversarial strengthening: {plan['use_adversarial_training']}")
    print(f"  Next step: {strategy_decision} ({decision_reason})")

    kb.log_event("strategy", "strategy_plan", plan)

    return {
        **state,
        "strategy_plan": plan,
        "strategy_decision": strategy_decision,
    }
