"""
Balance Agent — Execute phase of MAPE-K.
Handles class imbalance via CTGAN or SMOTE-like oversampling.
Receives data from Drift Agent, feeds into Supervisor.
Connected to KB for balance history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CTGAN_EPOCHS, CTGAN_SAMPLE_RATIO


def balance_agent(state: dict) -> dict:
    """
    Analyze class imbalance and generate synthetic minority samples
    using CTGAN (preferred) or noise-oversampling (fallback).
    """
    print("\n" + "=" * 60)
    print(" [ BALANCE AGENT ] Handling class imbalance (CTGAN / SMOTE)...")
    print("=" * 60)

    kb = state["knowledge_base"]
    train_df: pd.DataFrame = state["train_df"]
    feature_cols: list[str] = state["feature_cols"]
    target_col: str = state["target_col"]
    l1_count = state.get("l1_count", 0)

    # ── KB context: read prior balance history (KB→Balance communication) ──
    prior_balance = kb.get_latest("balance_records")
    if prior_balance:
        prior_action = prior_balance.get("data", {}).get("action", "unknown")
        prior_method = prior_balance.get("data", {}).get("method", "n/a")
        print(f"  📖 KB context: prior balance action={prior_action}, method={prior_method}")
        if l1_count > 0:
            print(f"  ℹ️  L1 re-balance iteration #{l1_count} — adapting from KB history")

    fraud_df = train_df[train_df[target_col] == 1]
    non_fraud_df = train_df[train_df[target_col] != 1]
    n_fraud = len(fraud_df)
    n_non_fraud = len(non_fraud_df)

    print(f"  Fraud: {n_fraud}, Non-fraud: {n_non_fraud}")

    # Determine if balancing is needed
    if n_fraud < 5:
        print("  ⚠️  Too few fraud samples for balancing. Passing data through.")
        kb.log_event("balance", "balance_skipped", {"reason": "too_few_fraud", "n_fraud": n_fraud})
        return {
            **state,
            "balanced_train_df": train_df,
            "ctgan_samples": pd.DataFrame(),
            "balance_report": {"action": "skipped", "reason": "too_few_fraud"},
        }

    ratio = n_fraud / max(1, n_non_fraud)
    needs_balance = ratio < 0.3 or ratio > 3.0
    n_needed = int(abs(n_non_fraud - n_fraud) * CTGAN_SAMPLE_RATIO) if needs_balance else 0

    if not needs_balance or n_needed <= 0:
        print(f"  ✅ No balancing needed (ratio={ratio:.2f}).")
        kb.log_event("balance", "balance_skipped", {"reason": "balanced", "ratio": ratio})
        return {
            **state,
            "balanced_train_df": train_df,
            "ctgan_samples": pd.DataFrame(),
            "balance_report": {"action": "skipped", "reason": "already_balanced", "ratio": round(ratio, 4)},
        }

    print(f"  Generating {n_needed} synthetic fraud rows...")

    try:
        from ctgan import CTGAN

        fraud_features = fraud_df[feature_cols + [target_col]]
        ctgan_model = CTGAN(epochs=CTGAN_EPOCHS, verbose=False)
        ctgan_model.fit(fraud_features, discrete_columns=[target_col])
        synthetic = ctgan_model.sample(n_needed)
        synthetic[target_col] = 1
        method = "CTGAN"
        print(f"  ✅ Generated {len(synthetic)} synthetic fraud samples via CTGAN.")
    except ImportError:
        print("  ⚠️  CTGAN not installed. Falling back to noise-oversampling.")
        indices = np.random.choice(len(fraud_df), size=n_needed, replace=True)
        synthetic = fraud_df.iloc[indices].copy().reset_index(drop=True)
        for col in feature_cols:
            if synthetic[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                noise = np.random.normal(0, 0.01, size=len(synthetic))
                synthetic[col] = synthetic[col].astype(float) + noise
        synthetic[target_col] = 1
        method = "noise_oversampling"
        print(f"  ✅ Generated {len(synthetic)} samples via noise-oversampling fallback.")
    except Exception as exc:
        print(f"  ❌ Balancing failed: {exc}.")
        kb.log_event("balance", "balance_failed", {"error": str(exc)})
        return {
            **state,
            "balanced_train_df": train_df,
            "ctgan_samples": pd.DataFrame(),
            "balance_report": {"action": "failed", "error": str(exc)},
        }

    balanced = pd.concat([train_df, synthetic[feature_cols + [target_col]]], ignore_index=True)
    report = {
        "action": "balanced",
        "method": method,
        "original_fraud": n_fraud,
        "original_non_fraud": n_non_fraud,
        "synthetic_generated": len(synthetic),
        "total_after": len(balanced),
    }

    kb.log_event("balance", "balance_completed", report)
    print(f"  ✅ Balanced dataset: {len(balanced)} rows total.")

    return {
        **state,
        "balanced_train_df": balanced,
        "ctgan_samples": synthetic,
        "balance_report": report,
    }
