#!/usr/bin/env python3
"""
evaluate_on_drift_data.py
─────────────────────────
Evaluates BOTH models against the held-out drift test dataset:
  • Baseline  — Random Forest trained on original data (no augmentation)
  • MAPE-K    — Promoted XGBoost model (Adv-CTGAN augmented)

Usage:
    python evaluate_on_drift_data.py

Output:
    agent/output/plots/drift_confusion_matrix_comparison.png
"""

import sys
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent
AGENT_ROOT = REPO_ROOT / "agent"
sys.path.insert(0, str(AGENT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    DATASET_PATH,
    NUMERIC_FEATURE_COLS,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    PLOTS_DIR,
)

DRIFT_CSV = AGENT_ROOT / "output" / "scraped_data" / "drift_test_dataset.csv"
OUT_PATH  = PLOTS_DIR / "drift_confusion_matrix_comparison.png"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_drift_data():
    df = pd.read_csv(DRIFT_CSV, low_memory=False)
    available = [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
    missing = set(NUMERIC_FEATURE_COLS) - set(available)
    if missing:
        print(f"  ⚠️  Missing feature columns in drift CSV: {missing}")
    df = df.dropna(subset=available + [TARGET_COL])
    df = df[df[TARGET_COL].isin([0, 1])].reset_index(drop=True)
    X = df[available].values
    y = df[TARGET_COL].values.astype(int)
    print(f"  Drift dataset: {len(X)} samples | "
          f"{y.sum()} fraud / {(y == 0).sum()} non-fraud | "
          f"{len(available)} features")
    return X, y, available


def build_baseline(feature_cols):
    """Train a fresh baseline RF on the ORIGINAL dataset (no synthetic data)."""
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    available = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=available + [TARGET_COL])
    df = df[df[TARGET_COL].isin([0, 1])].reset_index(drop=True)
    X_orig = df[available].values
    y_orig = df[TARGET_COL].values.astype(int)

    X_train, _, y_train, _ = train_test_split(
        X_orig, y_orig, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_orig
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print(f"  Baseline RF trained on {len(X_train)} original samples.")
    return clf


def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
        "fpr":       round(fpr, 4),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

FRAUD_CMAP = LinearSegmentedColormap.from_list("fraud_blue",   ["#f0f4ff", "#1a3c8f"])
MAPEK_CMAP = LinearSegmentedColormap.from_list("mapek_indigo", ["#fdf0ff", "#5b21b6"])


def plot_single_cm(ax, cm, title, cmap, metrics):
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["Non-Fraud (0)", "Fraud (1)"]
    ticks  = np.arange(2)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)

    thresh = cm.max() / 2.0
    cell_names = {(0,0):"TN", (0,1):"FP", (1,0):"FN", (1,1):"TP"}
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{cell_names[(i,j)]}\n{cm[i, j]:,}",
                ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    strip = (
        f"F1={metrics['f1']:.3f}   Precision={metrics['precision']:.3f}   "
        f"Recall={metrics['recall']:.3f}   ROC-AUC={metrics['roc_auc']:.3f}   FPR={metrics['fpr']:.3f}"
    )
    ax.text(
        0.5, -0.18, strip,
        ha="center", va="top", transform=ax.transAxes,
        fontsize=10.5, color="#444",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8f8f8", ec="#ccc", lw=0.8),
    )


def generate_plot(bl_cm, bl_m, mk_cm, mk_m):
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#fafafa")

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.06, right=0.97,
        bottom=0.18, top=0.88,
        wspace=0.38,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    plot_single_cm(ax1, bl_cm, "Baseline (RF — No Augmentation)",  FRAUD_CMAP, bl_m)
    plot_single_cm(ax2, mk_cm, "MAPE-K + Adv-CTGAN (XGBoost)",     MAPEK_CMAP, mk_m)

    # Delta banner
    d_f1  = mk_m["f1"]      - bl_m["f1"]
    d_rec = mk_m["recall"]  - bl_m["recall"]
    d_fpr = mk_m["fpr"]     - bl_m["fpr"]

    def sign(v): return "+" if v >= 0 else ""

    delta_text = (
        f"Δ F1 {sign(d_f1)}{d_f1:+.3f}    "
        f"Δ Recall {sign(d_rec)}{d_rec:+.3f}    "
        f"Δ FPR {sign(d_fpr)}{d_fpr:+.3f}"
    )
    fig.text(
        0.5, 0.03, delta_text,
        ha="center", va="bottom",
        fontsize=12, fontweight="bold",
        color="#1a3c8f",
        bbox=dict(boxstyle="round,pad=0.5", fc="#e8eeff", ec="#1a3c8f", lw=1.2),
    )

    fig.suptitle(
        "Fraud Detection on Drift Data — Confusion Matrix Comparison",
        fontsize=17, fontweight="bold", y=0.97,
    )

    # Sub-title showing this is the DRIFT evaluation
    fig.text(
        0.5, 0.90,
        "Evaluated on: drift_test_dataset.csv  (unseen distribution shift data)",
        ha="center", fontsize=10.5, color="#666",
        style="italic",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✅ Saved → {OUT_PATH}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 62)
    print("  Drift-Data Evaluation  |  Baseline RF  vs  MAPE-K XGBoost")
    print("=" * 62)

    print("\n[1/4] Loading drift test dataset …")
    X_drift, y_drift, feature_cols = load_drift_data()

    print("\n[2/4] Training baseline RF on original dataset …")
    baseline = build_baseline(feature_cols)
    bl_pred  = baseline.predict(X_drift)
    bl_proba = baseline.predict_proba(X_drift)[:, 1]
    bl_cm    = confusion_matrix(y_drift, bl_pred, labels=[0, 1])
    bl_m     = compute_metrics(y_drift, bl_pred, bl_proba)
    print(f"  Baseline metrics  → {bl_m}")

    print("\n[3/4] Loading MAPE-K promoted model …")
    model_path = AGENT_ROOT / "output" / "models" / "promoted_model.joblib"
    if not model_path.exists():
        sys.exit(f"  ❌  Promoted model not found at {model_path}.\n"
                 "      Run the pipeline first:  python agent/main.py")
    mapek = joblib.load(model_path)
    mk_pred  = mapek.predict(X_drift)
    mk_proba = mapek.predict_proba(X_drift)[:, 1]
    mk_cm    = confusion_matrix(y_drift, mk_pred, labels=[0, 1])
    mk_m     = compute_metrics(y_drift, mk_pred, mk_proba)
    print(f"  MAPE-K metrics    → {mk_m}")

    print("\n[4/4] Generating comparison plot …")
    generate_plot(bl_cm, bl_m, mk_cm, mk_m)

    print("\n" + "=" * 62)
    print(f"  {'Metric':<14}  {'Baseline':>10}  {'MAPE-K':>10}  {'Delta':>10}")
    print(f"  {'-' * 52}")
    for k in ("f1", "precision", "recall", "roc_auc", "fpr"):
        d = mk_m[k] - bl_m[k]
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
        print(f"  {k:<14}  {bl_m[k]:>10.4f}  {mk_m[k]:>10.4f}  {arrow} {abs(d):.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
