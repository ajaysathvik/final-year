"""
Drift Agent — Monitor phase of MAPE-K.
Detects feature drift using PSI and KS test.
Reads drift history from Knowledge Base.

Fingerprint-aware: computes a content hash of the drift data and checks
the KB for prior remediation records with the same fingerprint.  If the
exact same drift data was already remediated in a previous run the agent
short-circuits and reports **no new drift**, because the model has already
learned to classify this distribution.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy import stats

from config import PSI_THRESHOLD, KS_THRESHOLD, USE_SCRAPED_DRIFT_DATA


# ── helpers ────────────────────────────────────────────────────────

def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    eps = 1e-4
    ref_min, ref_max = reference.min(), reference.max()
    if ref_min == ref_max:
        return 0.0
    breakpoints = np.linspace(ref_min, ref_max, bins + 1)
    ref_counts = np.histogram(reference, bins=breakpoints)[0] + eps
    cur_counts = np.histogram(current, bins=breakpoints)[0] + eps
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _fingerprint_df(df: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 hex digest of a DataFrame's content.

    The fingerprint is computed from sorted column values serialised to
    bytes so that row ordering doesn't matter (the data distribution is
    what we care about, not presentation order).
    """
    # Sort rows for order-independence, then serialise to CSV bytes
    sorted_df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
    csv_bytes = sorted_df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _get_remediated_fingerprints(kb) -> set[str]:
    """Collect all drift-data fingerprints that were previously remediated.

    Scans the full drift_history in the Knowledge Base for entries whose
    remediation was applied and that carry a ``drift_data_fingerprint``.
    """
    fps: set[str] = set()
    for entry in kb.get_history("drift_history", limit=100):
        data = entry.get("data", {})
        remediation = data.get("drift_remediation", {})
        fp = data.get("drift_data_fingerprint")
        if fp and remediation.get("applied"):
            fps.add(fp)
    return fps


# ── main agent function ───────────────────────────────────────────

def drift_agent(state: dict) -> dict:
    """
    Detect feature drift between the training distribution and new
    (scraped / generated) data using PSI and KS tests.

    **Fingerprint check** — before running any statistical tests the
    agent computes a SHA-256 fingerprint of the current drift data and
    looks up the Knowledge Base for a prior run where the *same*
    fingerprint was already remediated.  If found, the agent reports
    "no new drift" and skips remediation, because the model was already
    retrained on this exact distribution.
    """
    print("\n" + "=" * 60)
    print(" [ DRIFT AGENT ] Detecting feature drift (PSI / KS)...")
    print("=" * 60)

    kb = state["knowledge_base"]
    train_df: pd.DataFrame = state["train_df"]
    feature_cols: list[str] = state["feature_cols"]

    # Read prior drift history from KB
    prior = kb.get_latest("drift_history")
    if prior:
        print(f"  ℹ️  Prior drift record found: {prior.get('timestamp', 'unknown')}")

    scraped_df = state.get("scraped_df")
    allow_scraped_drift_data = USE_SCRAPED_DRIFT_DATA
    effective_scraped_df = scraped_df if allow_scraped_drift_data else None

    if scraped_df is not None and len(scraped_df) > 0 and not allow_scraped_drift_data:
        print("  ℹ️  Scraped drift data found but disabled by USE_SCRAPED_DRIFT_DATA=false.")

    # ── Fingerprint check: skip drift if same data was already remediated ──
    drift_data_fingerprint: str | None = None
    if effective_scraped_df is not None and len(effective_scraped_df) > 0:
        drift_data_fingerprint = _fingerprint_df(effective_scraped_df)
        previously_remediated = _get_remediated_fingerprints(kb)

        if drift_data_fingerprint in previously_remediated:
            print(f"  🔁 Drift data fingerprint {drift_data_fingerprint[:12]}… matches a "
                  f"previously remediated run — skipping drift detection.")
            print(f"  ✅ No new drift (model already adapted to this distribution).")

            # Still append the scraped rows so the model trains on the
            # combined distribution (consistency with prior run).
            required_cols = feature_cols + [state["target_col"]]
            usable_scraped = effective_scraped_df[required_cols].copy()
            remediated_train_df = pd.concat([train_df, usable_scraped], ignore_index=True)

            drift_remediation = {
                "applied": True,
                "method": "reuse_prior_remediation",
                "base_rows": int(len(train_df)),
                "scraped_rows_available": int(len(scraped_df)) if scraped_df is not None else 0,
                "rows_added": int(len(usable_scraped)),
                "total_rows_after": int(len(remediated_train_df)),
                "use_scraped_drift_data": allow_scraped_drift_data,
            }

            # Log to KB (drift_detected = False because it was already handled)
            kb.log_event("drift", "drift_detection", {
                "features_analyzed": 0,
                "drifted_features": [],
                "drift_detected": False,
                "drift_data_fingerprint": drift_data_fingerprint,
                "skipped_reason": "already_remediated",
                "drift_remediation": drift_remediation,
            })

            return {
                **state,
                "drift_report": {},
                "drift_detected": False,
                "drift_remediation": drift_remediation,
                "remediated_train_df": remediated_train_df,
            }

    # ── Normal drift detection path ────────────────────────────────
    if effective_scraped_df is not None and len(effective_scraped_df) > 0:
        print(f"  📥 Using scraped data ({len(effective_scraped_df)} rows) for current distribution vs original ({len(train_df)}).")
        ref = train_df
        cur = effective_scraped_df
    else:
        print("  ℹ️  No scraped data. Using 70/30 split of training data for drift check.")
        split_idx = int(len(train_df) * 0.7)
        ref = train_df.iloc[:split_idx]
        cur = train_df.iloc[split_idx:]

    drift_report: dict = {}
    any_drift = False

    for col in feature_cols:
        if col not in ref.columns or col not in cur.columns:
            continue
        ref_vals = ref[col].dropna().values.astype(float)
        cur_vals = cur[col].dropna().values.astype(float)
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        psi_val = _psi(ref_vals, cur_vals)
        ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
        drifted = psi_val > PSI_THRESHOLD or ks_pval < KS_THRESHOLD

        drift_report[col] = {
            "psi": round(psi_val, 6),
            "ks_stat": round(ks_stat, 6),
            "ks_pval": round(ks_pval, 6),
            "drifted": drifted,
        }
        if drifted:
            any_drift = True

    drifted_features = [c for c, v in drift_report.items() if v.get("drifted")]
    remediated_train_df = train_df
    drift_remediation = {
        "applied": False,
        "method": "none" if allow_scraped_drift_data else "disabled_by_env",
        "base_rows": int(len(train_df)),
        "scraped_rows_available": int(len(scraped_df)) if scraped_df is not None else 0,
        "rows_added": 0,
        "total_rows_after": int(len(train_df)),
        "use_scraped_drift_data": allow_scraped_drift_data,
    }

    if any_drift and effective_scraped_df is not None and len(effective_scraped_df) > 0:
        required_cols = feature_cols + [state["target_col"]]
        usable_scraped = effective_scraped_df[required_cols].copy()
        remediated_train_df = pd.concat([train_df, usable_scraped], ignore_index=True)
        drift_remediation = {
            "applied": True,
            "method": "append_scraped_rows",
            "base_rows": int(len(train_df)),
            "scraped_rows_available": int(len(scraped_df)),
            "rows_added": int(len(usable_scraped)),
            "total_rows_after": int(len(remediated_train_df)),
            "use_scraped_drift_data": allow_scraped_drift_data,
        }
        print(
            f"  🔧 Drift remediation: appended {len(usable_scraped)} scraped rows "
            f"to training data ({len(remediated_train_df)} rows total)."
        )
    print(f"  ✅ Analyzed {len(drift_report)} features.")
    print(f"  {'⚠️  Drift detected' if any_drift else '✅ No drift detected'} "
          f"in {len(drifted_features)} feature(s)")

    # Log to Knowledge Base (include fingerprint so future runs can match it)
    kb.log_event("drift", "drift_detection", {
        "features_analyzed": len(drift_report),
        "drifted_features": drifted_features,
        "drift_detected": any_drift,
        "drift_data_fingerprint": drift_data_fingerprint,
        "drift_remediation": drift_remediation,
    })

    return {
        **state,
        "drift_report": drift_report,
        "drift_detected": any_drift,
        "drift_remediation": drift_remediation,
        "remediated_train_df": remediated_train_df,
    }
