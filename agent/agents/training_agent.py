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
from sklearn.metrics import f1_score

from config import MODELS_DIR, RANDOM_STATE, MAX_L3_ITERATIONS, F1_THRESHOLD


def _noise_perturbation(X: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to all features."""
    noise = np.random.normal(0, std, size=X.shape)
    return X + noise


def _boundary_attack(X_fraud: np.ndarray, X_non_fraud: np.ndarray, k: int) -> np.ndarray:
    """Generate samples near the decision boundary via interpolation."""
    from sklearn.neighbors import NearestNeighbors

    if len(X_non_fraud) == 0:
        return np.empty((0, X_fraud.shape[1]))

    if len(X_non_fraud) < k:
        k = max(1, len(X_non_fraud))

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_non_fraud)
    _, indices = nn.kneighbors(X_fraud)

    boundary_samples = []
    for i in range(len(X_fraud)):
        for j in range(min(k, len(indices[i]))):
            alpha = np.random.uniform(0.3, 0.7)
            sample = alpha * X_fraud[i] + (1 - alpha) * X_non_fraud[indices[i][j]]
            boundary_samples.append(sample)

    if not boundary_samples:
        return np.empty((0, X_fraud.shape[1]))
    return np.array(boundary_samples)


def _evasion_mutation(X: np.ndarray, std: float) -> np.ndarray:
    """Mutate a subset of features to mimic evasive behavior."""
    mutated = X.copy()
    n_features = X.shape[1]
    n_mutate = max(1, n_features // 3)

    for i in range(len(mutated)):
        cols = np.random.choice(n_features, size=n_mutate, replace=False)
        mutated[i, cols] += np.random.normal(0, std, size=n_mutate)

    return mutated


def _generate_adversarial_samples(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    adv_strategy: dict,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Generate adversarial fraud samples for training-time augmentation."""
    noise_cfg = adv_strategy.get("noise_perturbation", {})
    boundary_cfg = adv_strategy.get("boundary_attack", {})
    evasion_cfg = adv_strategy.get("evasion_mutation", {})
    noise_std = noise_cfg.get("std", 0.05)
    boundary_k = boundary_cfg.get("k", 5)
    evasion_std = evasion_cfg.get("std", round(noise_std * 0.6, 4))

    fraud_mask = train_df[target_col] == 1
    X_fraud = train_df.loc[fraud_mask, feature_cols].values.astype(float)
    X_non_fraud = train_df.loc[~fraud_mask, feature_cols].values.astype(float)

    if len(X_fraud) < 2:
        return pd.DataFrame(), {"generated": 0, "reason": "too_few_fraud_samples"}

    noise_samples = (
        _noise_perturbation(X_fraud, noise_std)
        if noise_cfg.get("enabled", True)
        else np.empty((0, X_fraud.shape[1]))
    )
    boundary_samples = (
        _boundary_attack(X_fraud, X_non_fraud, boundary_k)
        if boundary_cfg.get("enabled", True)
        else np.empty((0, X_fraud.shape[1]))
    )
    evasion_samples = (
        _evasion_mutation(X_fraud, evasion_std)
        if evasion_cfg.get("enabled", True)
        else np.empty((0, X_fraud.shape[1]))
    )

    all_adv = np.vstack([noise_samples, boundary_samples, evasion_samples])
    adv_df = pd.DataFrame(all_adv, columns=feature_cols)
    adv_df[target_col] = 1

    report = {
        "generated": int(len(adv_df)),
        "base_rows": int(len(train_df)),
        "noise_samples": int(len(noise_samples)),
        "boundary_samples": int(len(boundary_samples)),
        "evasion_samples": int(len(evasion_samples)),
        "noise_std": float(noise_std),
        "boundary_k": int(boundary_k),
        "evasion_std": float(evasion_std),
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
    strategy_decision = state.get("strategy_decision")

    # Use balanced data if available, else original training data
    base_train_data = state.get("balanced_train_df", state["train_df"])
    train_data = base_train_data
    use_adversarial_training = bool(strategy.get("use_adversarial_training", False))
    adv_samples = state.get("adversarial_samples")
    adv_report = state.get("adversarial_report", {})
    used_adversarial_samples = False

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

    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values.astype(int)

    print(f"  Training rows: {len(train_data)} "
          f"(fraud={int(y_train.sum())}, non-fraud={int(len(y_train) - y_train.sum())})")
    if use_adversarial_training:
        if used_adversarial_samples:
            print(f"  ℹ️  Adversarial strengthening enabled with {len(adv_samples)} generated samples")
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

    if model_type == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=hyper.get("max_depth", 10),
            class_weight=hyper.get("class_weight", "balanced"),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        print(f"  Using RandomForest (n_estimators={n_estimators})...")

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

    metrics = {
        "model_type": model_type,
        "train_f1": train_f1,
        "total_rows": len(train_data),
        "n_fraud": int(y_train.sum()),
        "n_non_fraud": int(len(y_train) - y_train.sum()),
        "n_estimators": n_estimators,
        "l3_count": new_l3_count,
        "used_adversarial_samples": used_adversarial_samples,
        "adversarial_sample_count": int(len(adv_samples)) if used_adversarial_samples else 0,
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
        "adversarial_samples": adv_samples if use_adversarial_training else pd.DataFrame(),
        "adversarial_report": adv_report if use_adversarial_training else {},
        "adversarial_trained": used_adversarial_samples,
    }
