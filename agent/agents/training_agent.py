"""
Training Agent — Execute phase of MAPE-K.
Trains a model on balanced data using the Strategy Agent's plan.
L3 feedback loop: self-loop for iterative retraining.
On success, returns control to Strategy so Strategy remains the central hub.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from config import MODELS_DIR, RANDOM_STATE, MAX_L3_ITERATIONS, F1_THRESHOLD


def _fgsm_perturb(X: np.ndarray, epsilon: float, rng: np.random.RandomState) -> np.ndarray:
    """
    FGSM-style perturbation for tabular/tree models.
    Since no gradient is available here, use random-sign steps on the
    numeric feature vector, matching the reference implementation.
    """
    if epsilon <= 0 or X.size == 0:
        return X.copy()

    perturbed = X.copy().astype(np.float32)
    noise = epsilon * np.sign(rng.randn(*perturbed.shape))
    perturbed += noise
    return np.clip(perturbed, 0.0, None).astype(np.float32)


def _generate_adversarial_samples(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    adv_strategy: dict,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Generate FGSM-style adversarial copies of the training set."""
    noise_cfg = adv_strategy.get("noise_perturbation", {})
    epsilon = float(noise_cfg.get("std", 0.05))

    if train_df.empty or not feature_cols:
        return pd.DataFrame(), {"generated": 0, "reason": "no_train_features"}

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(int)

    rng = np.random.RandomState(RANDOM_STATE)
    X_adv = _fgsm_perturb(X_train, epsilon=epsilon, rng=rng)

    adv_df = pd.DataFrame(X_adv, columns=feature_cols)
    adv_df[target_col] = y_train

    report = {
        "generated": int(len(adv_df)),
        "base_rows": int(len(train_df)),
        "method": "fgsm_style_random_sign",
        "epsilon": epsilon,
        "augmented_total": int(len(train_df) + len(adv_df)),
    }
    return adv_df, report


def training_agent(state: dict) -> dict:
    """
    Train a classifier using the balanced dataset and strategy plan.
    Supports L3 self-loop for iterative improvement.
    Increments l3_count when self-looping.
    """
    print("\n" + "=" * 60)
    print(" [ TRAINING AGENT ] Training candidate model...")
    print("=" * 60)

    kb = state["knowledge_base"]
    strategy = state.get("strategy_plan", {})
    feature_cols = state["feature_cols"]
    target_col = state["target_col"]
    l3_count = state.get("l3_count", 0)
    f1_history: list[float] = list(state.get("f1_history") or [])
    strategy_decision = state.get("strategy_decision")

    # Prefer drift-remediated data, then balanced data built from it.
    base_train_data = state.get("balanced_train_df")
    if base_train_data is None:
        base_train_data = state.get("remediated_train_df", state["train_df"])
    train_data = base_train_data
    use_adversarial_training = bool(strategy.get("use_adversarial_training", False))
    adv_samples = state.get("adversarial_samples")
    adv_report = state.get("adversarial_report", {})
    used_adversarial_samples = False
    drift_remediation = state.get("drift_remediation", {})
    used_drift_remediation = bool(drift_remediation.get("applied", False))

    if use_adversarial_training:
        adv_samples, adv_report = _generate_adversarial_samples(
            train_df=base_train_data,
            feature_cols=feature_cols,
            target_col=target_col,
            adv_strategy=strategy.get("adversarial_strategy", {}),
        )
        if isinstance(adv_samples, pd.DataFrame) and not adv_samples.empty:
            train_data = pd.concat([base_train_data, adv_samples], ignore_index=True)
            used_adversarial_samples = True

    # ── Bug-5 fix: detect post-augmentation class imbalance ─────
    post_augmentation_imbalanced = False
    y_check = train_data[target_col].values.astype(int)
    n_fraud_post = int(y_check.sum())
    n_non_fraud_post = int(len(y_check) - n_fraud_post)
    if n_non_fraud_post > 0:
        post_aug_ratio = n_fraud_post / n_non_fraud_post
    else:
        post_aug_ratio = float("inf")

    if post_aug_ratio > 3.0 or post_aug_ratio < 0.3:
        post_augmentation_imbalanced = True
        print(f"  ⚠️ Post-augmentation imbalance detected: "
              f"fraud={n_fraud_post}, non-fraud={n_non_fraud_post}, ratio={post_aug_ratio:.2f}")

    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values.astype(int)

    print(f"  Training rows: {len(train_data)} "
          f"(fraud={int(y_train.sum())}, non-fraud={int(len(y_train) - y_train.sum())})")
    if used_drift_remediation:
        print(
            f"  ℹ️  Drift remediation enabled with "
            f"{drift_remediation.get('rows_added', 0)} appended scraped rows"
        )
    if use_adversarial_training:
        if used_adversarial_samples:
            print(
                f"  ℹ️  FGSM-style adversarial strengthening enabled "
                f"(epsilon={adv_report.get('epsilon', 'n/a')}, generated={len(adv_samples)})"
            )
        else:
            print(f"  ℹ️  Adversarial strengthening requested but skipped: {adv_report.get('reason', 'no_samples_generated')}")
    if l3_count > 0:
        print(f"  ℹ️  L3 self-loop iteration #{l3_count}")

    # ── Build model from strategy plan ──────────────────────────
    hyper = strategy.get("hyperparameters", {})
    model_type = strategy.get("model_type", "RandomForest")
    n_estimators = hyper.get("n_estimators", 200)

    # On L3 retrain, boost estimators
    if l3_count > 0:
        n_estimators = n_estimators + (50 * l3_count)
        print(f"  ℹ️  L3 boost: n_estimators increased to {n_estimators}")

    if model_type == "XGBoost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=hyper.get("max_depth", 6),
                learning_rate=hyper.get("learning_rate", 0.1),
                scale_pos_weight=max(1, int((y_train == 0).sum() / max(1, (y_train == 1).sum()))),
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                use_label_encoder=False,
            )
            print(f"  Using XGBoost (n_estimators={n_estimators})...")
        except ImportError:
            model_type = "RandomForest"  # fall through
            print("  ⚠️  XGBoost not available, falling back to RandomForest.")

    elif model_type == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=hyper.get("max_depth", 10),
            class_weight=hyper.get("class_weight", "balanced"),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        print(f"  Using RandomForest (n_estimators={n_estimators})...")
    elif model_type == "LogisticRegression":
        model = LogisticRegression(
            C=hyper.get("C", 1.0),
            max_iter=hyper.get("max_iter", 1000),
            class_weight=hyper.get("class_weight", "balanced"),
            solver=hyper.get("solver", "liblinear"),
            random_state=RANDOM_STATE,
        )
        print(
            f"  Using LogisticRegression "
            f"(C={hyper.get('C', 1.0)}, max_iter={hyper.get('max_iter', 1000)})..."
        )
    else:
        raise ValueError(f"Unsupported model_type from strategy plan: {model_type}")

    model.fit(X_train, y_train)

    # Quick training-set check
    train_preds = model.predict(X_train)
    train_f1 = round(float(f1_score(y_train, train_preds, zero_division=0)), 4)
    print(f"  ✅ Training complete. Train-set F1={train_f1} ({model_type})")

    # Save model
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = str(MODELS_DIR / f"candidate_{model_type.lower()}_{timestamp}.joblib")
    joblib.dump(model, model_path)
    print(f"  💾 Saved model: {model_path}")

    # L3 decision: if train F1 is too low, increment counter for self-loop
    new_l3_count = l3_count
    if train_f1 < F1_THRESHOLD and l3_count < MAX_L3_ITERATIONS:
        new_l3_count = l3_count + 1
        print(f"  🔄 L3: Train-set F1={train_f1} < {F1_THRESHOLD}, will retrain (count={new_l3_count})")

    # Append the real F1 measured this iteration to the history list.
    f1_history.append(train_f1)

    metrics = {
        "model_type": model_type,
        "train_f1": train_f1,
        "f1_history": f1_history,
        "total_rows": len(train_data),
        "n_fraud": int(y_train.sum()),
        "n_non_fraud": int(len(y_train) - y_train.sum()),
        "n_estimators": n_estimators,
        "l3_count": new_l3_count,
        "used_drift_remediation": used_drift_remediation,
        "drift_rows_added": int(drift_remediation.get("rows_added", 0)) if used_drift_remediation else 0,
        "used_adversarial_samples": used_adversarial_samples,
        "adversarial_sample_count": int(len(adv_samples)) if used_adversarial_samples else 0,
        "adversarial_method": adv_report.get("method") if used_adversarial_samples else None,
        "adversarial_epsilon": adv_report.get("epsilon") if used_adversarial_samples else None,
        "timestamp": timestamp,
    }

    if use_adversarial_training and kb:
        kb.log_event("adversarial", "adversarial_samples_generated", adv_report)

    kb.log_event("training", "model_trained", metrics)
    kb.save_metrics(f"training_metrics_{timestamp}", metrics)

    return {
        **state,
        "candidate_model": model,
        "candidate_model_path": model_path,
        "training_metrics": metrics,
        "candidate_needs_evaluation": True,
        "simulation_results": {},
        "simulation_passed": None,
        "needs_strategy_refinement": False,
        "needs_rebalance": False,
        "l3_count": new_l3_count,
        "f1_history": f1_history,
        "adversarial_samples": adv_samples if use_adversarial_training else pd.DataFrame(),
        "adversarial_report": adv_report if use_adversarial_training else {},
        "adversarial_trained": used_adversarial_samples,
        "augmented_train_df": train_data,  # Bug-5: store full augmented data for balance re-check
        "post_augmentation_imbalanced": post_augmentation_imbalanced,  # Bug-5: flag for L1
    }
