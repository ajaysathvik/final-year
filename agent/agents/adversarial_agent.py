"""
Adversarial Agent — Execute phase of MAPE-K.
Generates adversarial fraud samples using three attack strategies:
  1. Noise perturbation
  2. Boundary attack (decision-boundary neighbors)
  3. Evasion-style mutation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ADVERSARIAL_NOISE_STD,
    ADVERSARIAL_BOUNDARY_K,
    ADVERSARIAL_EVASION_STD,
)


def _noise_perturbation(X: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to all features."""
    noise = np.random.normal(0, std, size=X.shape)
    return X + noise


def _boundary_attack(X_fraud: np.ndarray, X_non_fraud: np.ndarray, k: int) -> np.ndarray:
    """
    Generate samples near the decision boundary by interpolating between
    fraud and the nearest non-fraud samples.
    """
    from sklearn.neighbors import NearestNeighbors

    if len(X_non_fraud) < k:
        k = max(1, len(X_non_fraud))

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_non_fraud)
    distances, indices = nn.kneighbors(X_fraud)

    boundary_samples = []
    for i in range(len(X_fraud)):
        for j in range(min(k, len(indices[i]))):
            alpha = np.random.uniform(0.3, 0.7)  # interpolation weight
            sample = alpha * X_fraud[i] + (1 - alpha) * X_non_fraud[indices[i][j]]
            boundary_samples.append(sample)

    if not boundary_samples:
        return np.empty((0, X_fraud.shape[1]))
    return np.array(boundary_samples)


def _evasion_mutation(X: np.ndarray, std: float) -> np.ndarray:
    """
    Simulate evasion attacks by selectively mutating a subset of features
    (mimicking an attacker tweaking features to evade detection).
    """
    mutated = X.copy()
    n_features = X.shape[1]
    n_mutate = max(1, n_features // 3)  # mutate ~33% of features

    for i in range(len(mutated)):
        cols = np.random.choice(n_features, size=n_mutate, replace=False)
        mutated[i, cols] += np.random.normal(0, std, size=n_mutate)

    return mutated


def adversarial_agent(state: dict) -> dict:
    """
    Generate adversarial fraud samples using three techniques and combine them.
    """
    print("\n" + "=" * 60)
    print(" [ ADVERSARIAL AGENT ] Generating adversarial samples...")
    print("=" * 60)

    train_df: pd.DataFrame = state["train_df"]
    feature_cols: list[str] = state["feature_cols"]
    target_col: str = state["target_col"]

    fraud_mask = train_df[target_col] == 1
    X_fraud = train_df.loc[fraud_mask, feature_cols].values.astype(float)
    X_non_fraud = train_df.loc[~fraud_mask, feature_cols].values.astype(float)

    if len(X_fraud) < 2:
        print("  ⚠️  Too few fraud samples for adversarial generation.")
        return {**state, "adversarial_samples": pd.DataFrame()}

    # 1. Noise perturbation
    noise_samples = _noise_perturbation(X_fraud, ADVERSARIAL_NOISE_STD)
    print(f"  ✅ Noise perturbation: {len(noise_samples)} samples")

    # 2. Boundary attack
    boundary_samples = _boundary_attack(X_fraud, X_non_fraud, ADVERSARIAL_BOUNDARY_K)
    print(f"  ✅ Boundary attack: {len(boundary_samples)} samples")

    # 3. Evasion mutation
    evasion_samples = _evasion_mutation(X_fraud, ADVERSARIAL_EVASION_STD)
    print(f"  ✅ Evasion mutation: {len(evasion_samples)} samples")

    # Combine all adversarial samples
    all_adv = np.vstack([noise_samples, boundary_samples, evasion_samples])
    adv_df = pd.DataFrame(all_adv, columns=feature_cols)
    adv_df[target_col] = 1  # all adversarial samples are labeled fraud

    print(f"  ✅ Total adversarial samples: {len(adv_df)}")

    return {**state, "adversarial_samples": adv_df}
