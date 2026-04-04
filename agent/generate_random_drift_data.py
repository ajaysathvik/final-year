"""
generate_random_drift_data.py
─────────────────────────────
Procedurally generates a large, diverse drift test dataset and saves it to
  agent/output/scraped_data/drift_test_dataset.csv

Activated via:
    GENERATE_RANDOM_DRIFT_DATA=true python agent/main.py

Each run uses a fresh random seed so drift intensity, class balance, and
feature distributions shift continuously across runs — letting you observe
how the MAPE-K model adapts to evolving data.

Usage (standalone):
    python agent/generate_random_drift_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).resolve().parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

import numpy as np
import pandas as pd

from config import (
    NUMERIC_FEATURE_COLS,
    TARGET_COL,
    RANDOM_DRIFT_N_SAMPLES,
    RANDOM_DRIFT_SEED,
)

DRIFT_CSV = _AGENT_ROOT / "output" / "scraped_data" / "drift_test_dataset.csv"

# ── Columns we'll generate ────────────────────────────────────────────────────
# Matches NUMERIC_FEATURE_COLS + TARGET_COL exactly.
# We also use an alias map so the generator stays readable.
COLS = [TARGET_COL] + NUMERIC_FEATURE_COLS

# ── Drift Scenarios ───────────────────────────────────────────────────────────
#
# Each scenario defines baseline parameter ranges for the 9 numeric features.
# The generator will sample from these ranges AND apply per-run random offsets
# (drift intensity) so every run produces statistically distinct data.
#
# Feature order inside each scenario:
#   urgency | fear | authority | reward
#   num_comments | scam_confirmations | not_scam_claims | advice_requests
#   has_amount
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    # ── FRAUD SCENARIOS ─────────────────────────────────────────────────────
    "high_urgency_fraud": {
        "label": 1,
        "weight": 0.25,          # relative sampling frequency (normalised later)
        "features": {
            "annotation.psychological_tactics.urgency":    (0.70, 1.00),
            "annotation.psychological_tactics.fear":       (0.60, 1.00),
            "annotation.psychological_tactics.authority":  (0.50, 1.00),
            "annotation.psychological_tactics.reward":     (0.00, 0.30),
            "annotation.community_signals.num_comments":   (30,   150),
            "annotation.community_signals.scam_confirmations": (5, 25),
            "annotation.community_signals.not_scam_claims":    (0,  3),
            "annotation.community_signals.advice_requests":    (5, 20),
            "annotation.key_features.has_amount":          (0.70, 1.00),
        },
    },
    "silent_fraud": {
        "label": 1,
        "weight": 0.20,
        "features": {
            # No psychological tactics — hard to detect
            "annotation.psychological_tactics.urgency":    (0.00, 0.20),
            "annotation.psychological_tactics.fear":       (0.00, 0.20),
            "annotation.psychological_tactics.authority":  (0.00, 0.20),
            "annotation.psychological_tactics.reward":     (0.10, 0.50),
            "annotation.community_signals.num_comments":   (80,  250),
            "annotation.community_signals.scam_confirmations": (8, 30),
            "annotation.community_signals.not_scam_claims":    (0,  5),
            "annotation.community_signals.advice_requests":    (10, 40),
            "annotation.key_features.has_amount":          (0.50, 1.00),
        },
    },
    "coordinated_fraud": {
        "label": 1,
        "weight": 0.15,
        # Organised fraud — all tactics on, very large threads
        "features": {
            "annotation.psychological_tactics.urgency":    (0.80, 1.00),
            "annotation.psychological_tactics.fear":       (0.80, 1.00),
            "annotation.psychological_tactics.authority":  (0.80, 1.00),
            "annotation.psychological_tactics.reward":     (0.70, 1.00),
            "annotation.community_signals.num_comments":   (200, 600),
            "annotation.community_signals.scam_confirmations": (20, 80),
            "annotation.community_signals.not_scam_claims":    (0, 10),
            "annotation.community_signals.advice_requests":    (30, 80),
            "annotation.key_features.has_amount":          (0.85, 1.00),
        },
    },
    # ── NON-FRAUD SCENARIOS ──────────────────────────────────────────────────
    "reward_bait_legit": {
        "label": 0,
        "weight": 0.20,
        # Legitimate giveaway/promo posts — high reward, lots of non-scam claims
        "features": {
            "annotation.psychological_tactics.urgency":    (0.00, 0.30),
            "annotation.psychological_tactics.fear":       (0.00, 0.20),
            "annotation.psychological_tactics.authority":  (0.00, 0.40),
            "annotation.psychological_tactics.reward":     (0.70, 1.00),
            "annotation.community_signals.num_comments":   (1,   20),
            "annotation.community_signals.scam_confirmations": (0,  2),
            "annotation.community_signals.not_scam_claims":    (3, 15),
            "annotation.community_signals.advice_requests":    (0,  3),
            "annotation.key_features.has_amount":          (0.00, 0.50),
        },
    },
    "noisy_legit": {
        "label": 0,
        "weight": 0.20,
        # Mixed signals — hard negatives that look borderline
        "features": {
            "annotation.psychological_tactics.urgency":    (0.20, 0.70),
            "annotation.psychological_tactics.fear":       (0.20, 0.60),
            "annotation.psychological_tactics.authority":  (0.30, 0.80),
            "annotation.psychological_tactics.reward":     (0.10, 0.60),
            "annotation.community_signals.num_comments":   (60,  200),
            "annotation.community_signals.scam_confirmations": (0,  5),
            "annotation.community_signals.not_scam_claims":    (15, 60),
            "annotation.community_signals.advice_requests":    (0, 10),
            "annotation.key_features.has_amount":          (0.00, 0.80),
        },
    },
}


def _binary_feature(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    """Sample a binary 0/1 feature whose P(1) is drawn from [low, high]."""
    p = rng.uniform(low, high)
    return rng.binomial(1, p, size=size).astype(float)


def _continuous_feature(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    """Sample a continuous int-like feature with Gaussian noise."""
    mid = rng.uniform(low, high)
    spread = (high - low) * 0.25
    vals = rng.normal(loc=mid, scale=max(spread, 1.0), size=size)
    return np.clip(vals, low * 0.5, high * 1.5).astype(float)


def _generate_scenario(
    name: str,
    scenario: dict,
    n: int,
    rng: np.random.Generator,
    drift_offset: float,
) -> pd.DataFrame:
    """
    Generate `n` rows for a given scenario, applying a per-run drift offset
    to continuous features so distribution shifts between runs.
    """
    rows = {}
    rows[TARGET_COL] = np.full(n, scenario["label"], dtype=int)

    for feat, (lo, hi) in scenario["features"].items():
        is_binary_tactic = feat.startswith("annotation.psychological_tactics.")
        is_has_amount    = feat == "annotation.key_features.has_amount"

        if is_binary_tactic or is_has_amount:
            # Apply drift_offset as a nudge to the probability range
            lo_d = float(np.clip(lo + drift_offset * rng.uniform(-0.3, 0.3), 0.0, 1.0))
            hi_d = float(np.clip(hi + drift_offset * rng.uniform(-0.2, 0.2), lo_d, 1.0))
            rows[feat] = _binary_feature(rng, lo_d, hi_d, n)
        else:
            # Scale continuous range by a drift multiplier each run
            drift_mult = 1.0 + drift_offset * rng.uniform(-0.4, 0.6)
            lo_d = lo * max(drift_mult * 0.8, 0.1)
            hi_d = hi * max(drift_mult, 0.1)
            rows[feat] = _continuous_feature(rng, lo_d, hi_d, n)

    return pd.DataFrame(rows)


def generate_drift_dataframe(
    n_samples: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate a diverse drift test DataFrame.

    Parameters
    ----------
    n_samples : int, optional
        Total rows to generate. Defaults to RANDOM_DRIFT_N_SAMPLES from config.
    seed : int, optional
        RNG seed. Defaults to RANDOM_DRIFT_SEED (None → new seed each call).

    Returns
    -------
    pd.DataFrame  —  with columns: annotation.is_fraud + all NUMERIC_FEATURE_COLS
    """
    if n_samples is None:
        n_samples = RANDOM_DRIFT_N_SAMPLES
    if seed is None:
        seed = RANDOM_DRIFT_SEED  # None → truly random each run

    rng = np.random.default_rng(seed)

    # Draw a per-run drift intensity [0, 1]  — low = mild, high = severe drift
    drift_offset = float(rng.uniform(0.1, 1.0))
    print(f"  🌀 Drift intensity this run: {drift_offset:.3f}")

    # Draw scenario weights, then randomly perturb them (so mix changes each run)
    names   = list(SCENARIOS.keys())
    weights = np.array([SCENARIOS[n]["weight"] for n in names], dtype=float)
    weights += rng.uniform(0, 0.15, size=len(weights))   # small random nudge
    weights /= weights.sum()

    # Allocate sample budget across scenarios
    counts = (weights * n_samples).astype(int)
    counts[-1] += n_samples - counts.sum()   # fix rounding remainder

    # Build a class-balance target — fraud ratio varies each run
    target_fraud_ratio = float(rng.uniform(0.25, 0.60))
    target_fraud_n     = int(n_samples * target_fraud_ratio)
    target_nonfr_n     = n_samples - target_fraud_n
    print(f"  🎯 Target fraud ratio: {target_fraud_ratio:.2f} "
          f"({target_fraud_n} fraud / {target_nonfr_n} non-fraud)")

    # Generate per-scenario DataFrames
    dfs = []
    for i, name in enumerate(names):
        n = int(counts[i])
        if n == 0:
            continue
        df_s = _generate_scenario(name, SCENARIOS[name], n, rng, drift_offset)
        dfs.append(df_s)
        print(f"    Scenario [{name}]: {n} rows (label={SCENARIOS[name]['label']})")

    combined = pd.concat(dfs, ignore_index=True)

    # ── Re-balance to hit the target fraud ratio ───────────────────────────
    fraud_rows  = combined[combined[TARGET_COL] == 1]
    nonfr_rows  = combined[combined[TARGET_COL] == 0]

    # Sample with replacement if we need more than available
    fraud_sample  = fraud_rows.sample(n=target_fraud_n, replace=len(fraud_rows) < target_fraud_n,
                                      random_state=int(rng.integers(0, 99999)))
    nonfr_sample  = nonfr_rows.sample(n=target_nonfr_n, replace=len(nonfr_rows) < target_nonfr_n,
                                      random_state=int(rng.integers(0, 99999)))

    out = pd.concat([fraud_sample, nonfr_sample], ignore_index=True)
    out = out.sample(frac=1, random_state=int(rng.integers(0, 99999))).reset_index(drop=True)

    # ── Enforce column order matching the CSV header ───────────────────────
    ordered_cols = [TARGET_COL] + NUMERIC_FEATURE_COLS
    out = out[[c for c in ordered_cols if c in out.columns]]

    # Keep label strictly 0/1
    out[TARGET_COL] = out[TARGET_COL].astype(int)

    return out


def generate_and_save(
    n_samples: int | None = None,
    seed: int | None = None,
    path: Path | None = None,
) -> Path:
    """
    Generate a drift dataset and save it to disk, overwriting the existing CSV.

    Returns the path saved to.
    """
    if path is None:
        path = DRIFT_CSV

    path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "─" * 60)
    print("  🔄  GENERATE_RANDOM_DRIFT_DATA: generating new drift CSV")
    print("─" * 60)

    df = generate_drift_dataframe(n_samples=n_samples, seed=seed)

    # Write header as the original CSV uses full dotted column names
    df.to_csv(path, index=False)

    fraud_n  = int((df[TARGET_COL] == 1).sum())
    nonfr_n  = int((df[TARGET_COL] == 0).sum())
    print(f"  ✅ Saved {len(df)} rows → {path}")
    print(f"     Fraud={fraud_n} | Non-fraud={nonfr_n} | "
          f"Ratio={fraud_n / len(df):.2f}")
    print("─" * 60)

    return path


if __name__ == "__main__":
    generate_and_save()
