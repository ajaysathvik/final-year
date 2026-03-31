"""
Augmentation Agent — Execute phase of MAPE-K.
Generates synthetic fraud samples using CTGAN to handle class imbalance.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from config import CTGAN_EPOCHS, CTGAN_SAMPLE_RATIO


def augmentation_agent(state: dict) -> dict:
    """
    Train a CTGAN on the minority (fraud) class and generate synthetic samples
    to balance the training data.
    """
    print("\n" + "=" * 60)
    print(" [ AUGMENTATION AGENT ] Generating synthetic fraud samples (CTGAN)...")
    print("=" * 60)

    train_df: pd.DataFrame = state["train_df"]
    feature_cols: list[str] = state["feature_cols"]
    target_col: str = state["target_col"]

    fraud_df = train_df[train_df[target_col] == 1][feature_cols + [target_col]]
    non_fraud_df = train_df[train_df[target_col] != 1]

    n_fraud = len(fraud_df)
    n_non_fraud = len(non_fraud_df)
    n_needed = int(abs(n_non_fraud - n_fraud) * CTGAN_SAMPLE_RATIO)

    if n_needed <= 0 or n_fraud < 5:
        print(f"  ⚠️  No CTGAN generation needed (fraud={n_fraud}, non_fraud={n_non_fraud}).")
        return {**state, "ctgan_samples": pd.DataFrame()}

    print(f"  Fraud={n_fraud}, Non-fraud={n_non_fraud}, Generating {n_needed} synthetic fraud rows...")

    try:
        from ctgan import CTGAN

        ctgan_model = CTGAN(epochs=CTGAN_EPOCHS, verbose=False)
        ctgan_model.fit(fraud_df, discrete_columns=[target_col])
        synthetic = ctgan_model.sample(n_needed)
        synthetic[target_col] = 1  # ensure label
        print(f"  ✅ Generated {len(synthetic)} synthetic fraud samples via CTGAN.")
    except ImportError:
        print("  ⚠️  CTGAN not installed. Falling back to SMOTE-like oversampling.")
        # Simple noise-based oversampling as fallback
        indices = np.random.choice(len(fraud_df), size=n_needed, replace=True)
        synthetic = fraud_df.iloc[indices].copy().reset_index(drop=True)
        # Add small Gaussian noise to numeric columns
        for col in feature_cols:
            if synthetic[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                noise = np.random.normal(0, 0.01, size=len(synthetic))
                synthetic[col] = synthetic[col].astype(float) + noise
        synthetic[target_col] = 1
        print(f"  ✅ Generated {len(synthetic)} synthetic samples via noise-oversampling fallback.")
    except Exception as exc:
        print(f"  ❌ CTGAN failed: {exc}. Returning empty.")
        return {**state, "ctgan_samples": pd.DataFrame()}

    return {**state, "ctgan_samples": synthetic}
