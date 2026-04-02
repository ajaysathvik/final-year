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


def _format_class_ratio(n_fraud: int, n_non_fraud: int) -> str:
    """Return class ratio in fraud:non-fraud form with the smaller side normalized to 1."""
    if n_fraud <= 0 and n_non_fraud <= 0:
        return "0:0"
    if n_fraud <= 0:
        return f"0:{n_non_fraud}"
    if n_non_fraud <= 0:
        return f"{n_fraud}:0"

    if n_fraud <= n_non_fraud:
        return f"1:{(n_non_fraud / n_fraud):.2f}"
    return f"{(n_fraud / n_non_fraud):.2f}:1"


def _select_balance_target(n_fraud: int, n_non_fraud: int) -> tuple[int, str]:
    """Return the minority class label and name.

    For a 1:x imbalance, add one new sample for each current sample in the
    smaller class. That changes the smaller side from 1 to 2, instead of
    forcing strict 1:1 parity.
    """
    if n_fraud <= n_non_fraud:
        return 1, "fraud"
    return 0, "non_fraud"


def balance_agent(state: dict) -> dict:
    """
    Analyze class imbalance and generate synthetic minority samples
    using CTGAN (preferred) or noise-oversampling (fallback).
    """
    print("\n" + "=" * 60)
    print(" [ BALANCE AGENT ] Handling class imbalance (CTGAN / SMOTE)...")
    print("=" * 60)

    kb = state["knowledge_base"]
    l1_count = state.get("l1_count", 0)
    feature_cols: list[str] = state["feature_cols"]
    target_col: str = state["target_col"]

    # Bug-5 fix: on L1 re-entry, prefer the post-augmentation data (augmented_train_df)
    # so we can detect and fix the imbalance created by adversarial augmentation.
    if l1_count > 0 and state.get("augmented_train_df") is not None:
        train_df: pd.DataFrame = state["augmented_train_df"]
        print(f"  ℹ️  L1 re-entry: using augmented training data ({len(train_df)} rows) for rebalancing")
    else:
        train_df: pd.DataFrame = state.get("remediated_train_df", state["train_df"])

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
    class_ratio = _format_class_ratio(n_fraud, n_non_fraud)

    print(f"  Fraud: {n_fraud}, Non-fraud: {n_non_fraud}")
    print(f"  Class ratio (fraud:non-fraud): {class_ratio}")

    # Determine if balancing is needed
    if n_fraud < 5:
        print("  ⚠️  Too few fraud samples for balancing. Passing data through.")
        kb.log_event("balance", "balance_skipped", {
            "reason": "too_few_fraud",
            "n_fraud": n_fraud,
            "class_ratio": class_ratio,
        })
        return {
            **state,
            "balanced_train_df": train_df,
            "ctgan_samples": pd.DataFrame(),
            "balance_report": {
                "action": "skipped",
                "reason": "too_few_fraud",
                "class_ratio": class_ratio,
            },
        }

    ratio = n_fraud / max(1, n_non_fraud)
    needs_balance = ratio < 0.3 or ratio > 3.0

    minority_label, minority_name = _select_balance_target(n_fraud, n_non_fraud)
    minority_df = fraud_df if minority_label == 1 else non_fraud_df
    n_needed = int(len(minority_df) * CTGAN_SAMPLE_RATIO) if needs_balance else 0

    if not needs_balance or n_needed <= 0:
        print(f"  ✅ No balancing needed (ratio={ratio:.2f}, class_ratio={class_ratio}).")
        kb.log_event("balance", "balance_skipped", {
            "reason": "balanced",
            "ratio": ratio,
            "class_ratio": class_ratio,
        })
        return {
            **state,
            "balanced_train_df": train_df,
            "ctgan_samples": pd.DataFrame(),
            "balance_report": {
                "action": "skipped",
                "reason": "already_balanced",
                "ratio": round(ratio, 4),
                "class_ratio": class_ratio,
            },
        }

    print(f"  Generating {n_needed} synthetic {minority_name} rows...")

    try:
        from ctgan import CTGAN

        minority_features = minority_df[feature_cols + [target_col]]
        ctgan_model = CTGAN(epochs=CTGAN_EPOCHS, verbose=False)
        ctgan_model.fit(minority_features, discrete_columns=[target_col])
        synthetic = ctgan_model.sample(n_needed)
        synthetic[target_col] = minority_label
        method = "CTGAN"
        print(f"  ✅ Generated {len(synthetic)} synthetic {minority_name} samples via CTGAN.")
    except ImportError:
        print("  ⚠️  CTGAN not installed. Falling back to noise-oversampling.")
        indices = np.random.choice(len(minority_df), size=n_needed, replace=True)
        synthetic = minority_df.iloc[indices].copy().reset_index(drop=True)
        for col in feature_cols:
            if synthetic[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                noise = np.random.normal(0, 0.01, size=len(synthetic))
                synthetic[col] = synthetic[col].astype(float) + noise
        synthetic[target_col] = minority_label
        method = "noise_oversampling"
        print(f"  ✅ Generated {len(synthetic)} synthetic {minority_name} samples via noise-oversampling fallback.")
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
    balanced_fraud = int((balanced[target_col] == 1).sum())
    balanced_non_fraud = int((balanced[target_col] != 1).sum())
    report = {
        "action": "balanced",
        "method": method,
        "minority_class": minority_name,
        "original_fraud": n_fraud,
        "original_non_fraud": n_non_fraud,
        "original_class_ratio": class_ratio,
        "synthetic_generated": len(synthetic),
        "total_after": len(balanced),
        "balanced_fraud": balanced_fraud,
        "balanced_non_fraud": balanced_non_fraud,
        "balanced_class_ratio": _format_class_ratio(balanced_fraud, balanced_non_fraud),
    }

    kb.log_event("balance", "balance_completed", report)
    print(f"  ✅ Balanced dataset: {len(balanced)} rows total.")

    return {
        **state,
        "balanced_train_df": balanced,
        "ctgan_samples": synthetic,
        "balance_report": report,
    }
