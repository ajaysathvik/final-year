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
    Train a CTGAN on the minority class and generate synthetic samples
    to balance the training data.
    """
    print("\n" + "=" * 60)
    print(" [ AUGMENTATION AGENT ] Generating synthetic samples for balancing...")
    print("=" * 60)

    train_df: pd.DataFrame = state["train_df"]
    feature_cols: list[str] = state["feature_cols"]
    target_col: str = state["target_col"]

    fraud_df = train_df[train_df[target_col] == 1][feature_cols + [target_col]]
    non_fraud_df = train_df[train_df[target_col] == 0][feature_cols + [target_col]]

    n_fraud = len(fraud_df)
    n_non_fraud = len(non_fraud_df)

    if n_fraud < n_non_fraud:
        minority_df = fraud_df
        minority_label = 1
        minority_name = "Fraud"
        n_needed = int((n_non_fraud - n_fraud) * CTGAN_SAMPLE_RATIO)
    else:
        minority_df = non_fraud_df
        minority_label = 0
        minority_name = "Non-Fraud"
        n_needed = int((n_fraud - n_non_fraud) * CTGAN_SAMPLE_RATIO)

    if n_needed <= 0 or len(minority_df) < 5:
        print(f"  ⚠️  No augmentation needed (fraud={n_fraud}, non_fraud={n_non_fraud}).")
        return {**state, "ctgan_samples": pd.DataFrame()}

    print(f"  Minority={minority_name} ({len(minority_df)}), Majority ({max(n_fraud, n_non_fraud)}).")
    print(f"  Generating {n_needed} synthetic {minority_name} rows...")

    try:
        from ctgan import CTGAN

        ctgan_model = CTGAN(epochs=CTGAN_EPOCHS, verbose=False)
        ctgan_model.fit(minority_df, discrete_columns=[target_col])
        synthetic = ctgan_model.sample(n_needed)
        synthetic[target_col] = minority_label  # ensure correct label
        print(f"  ✅ Generated {len(synthetic)} synthetic {minority_name} samples via CTGAN.")
    except ImportError:
        print("  ⚠️  CTGAN not installed. Falling back to noise-based oversampling.")
        # Simple noise-based oversampling as fallback
        indices = np.random.choice(len(minority_df), size=n_needed, replace=True)
        synthetic = minority_df.iloc[indices].copy().reset_index(drop=True)
        # Add small Gaussian noise to numeric columns
        for col in feature_cols:
            if synthetic[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                noise = np.random.normal(0, 0.01, size=len(synthetic))
                synthetic[col] = synthetic[col].astype(float) + noise
        synthetic[target_col] = minority_label
        print(f"  ✅ Generated {len(synthetic)} synthetic {minority_name} samples via noise-oversampling.")
    except Exception as exc:
        print(f"  ❌ Augmentation failed: {exc}. Returning empty.")
        return {**state, "ctgan_samples": pd.DataFrame()}

    return {**state, "ctgan_samples": synthetic}
