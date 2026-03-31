"""
Training Agent — Execute phase of MAPE-K.
Trains a model on balanced data using the Strategy Agent's plan.
L3 feedback loop: self-loop for iterative retraining.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config import MODELS_DIR, RANDOM_STATE, MAX_L3_ITERATIONS, F1_THRESHOLD


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

    # Use balanced data if available, else original training data
    train_data = state.get("balanced_train_df", state["train_df"])

    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values.astype(int)

    print(f"  Training rows: {len(train_data)} "
          f"(fraud={int(y_train.sum())}, non-fraud={int(len(y_train) - y_train.sum())})")
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
    print(f"  ✅ Training complete. Train F1={train_f1} ({model_type})")

    # Save model
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = str(MODELS_DIR / f"candidate_{model_type.lower()}_{timestamp}.joblib")
    joblib.dump(model, model_path)
    print(f"  💾 Saved model: {model_path}")

    # L3 decision: if train F1 is too low, increment counter for self-loop
    new_l3_count = l3_count
    if train_f1 < F1_THRESHOLD and l3_count < MAX_L3_ITERATIONS:
        new_l3_count = l3_count + 1
        print(f"  🔄 L3: Train F1={train_f1} < {F1_THRESHOLD}, will retrain (count={new_l3_count})")

    metrics = {
        "model_type": model_type,
        "train_f1": train_f1,
        "total_rows": len(train_data),
        "n_fraud": int(y_train.sum()),
        "n_non_fraud": int(len(y_train) - y_train.sum()),
        "n_estimators": n_estimators,
        "l3_count": new_l3_count,
        "timestamp": timestamp,
    }

    kb.log_event("training", "model_trained", metrics)
    kb.save_metrics(f"training_metrics_{timestamp}", metrics)

    return {
        **state,
        "candidate_model": model,
        "candidate_model_path": model_path,
        "training_metrics": metrics,
        "l3_count": new_l3_count,
    }
