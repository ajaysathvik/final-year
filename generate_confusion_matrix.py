#!/usr/bin/env python3
"""
generate_confusion_matrix.py
────────────────────────────
Generates a publication-quality confusion matrix comparing:
  • Baseline (no augmentation / plain SMOTE)   — approximated from run_1 data
  • MAPE-K Adv-CTGAN model                      — promoted_model.joblib

Run from the repo root:
    python generate_confusion_matrix.py

Output is saved to:
    agent/output/plots/confusion_matrix_comparison.png
"""

import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
AGENT_ROOT = REPO_ROOT / "agent"
sys.path.insert(0, str(AGENT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

from config import (
    DATASET_PATH,
    NUMERIC_FEATURE_COLS,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    PLOTS_DIR,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    available = [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=available + [TARGET_COL])
    # Remove ambiguous labels (-1); keep only binary classes 0 (non-fraud) and 1 (fraud)
    df = df[df[TARGET_COL].isin([0, 1])].reset_index(drop=True)
    X = df[available].values
    y = df[TARGET_COL].values.astype(int)
    return X, y, available


def get_test_split(X, y):
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test


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


# ── build baseline model (no adversarial augmentation, plain RF) ─────────────

def build_baseline(X, y):
    """Train a simple XGBoost on the raw imbalanced data — no CTGAN, no Adv samples."""
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    clf = XGBClassifier(
        n_estimators=200,
        scale_pos_weight=max(1, int((y_train == 0).sum() / max(1, (y_train == 1).sum()))),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss"
    )
    clf.fit(X_train, y_train)
    return clf


# ── plotting ──────────────────────────────────────────────────────────────────

FRAUD_CMAP  = LinearSegmentedColormap.from_list(
    "fraud_blue", ["#f0f4ff", "#1a3c8f"]
)
MAPEK_CMAP  = LinearSegmentedColormap.from_list(
    "mapek_indigo", ["#fdf0ff", "#5b21b6"]
)


def plot_single_cm(ax, cm, title, cmap, metrics):
    """Render one confusion matrix with metrics annotation."""
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["Non-Fraud (0)", "Fraud (1)"]
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label", fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            cell_label = {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}[(i,j)]
            ax.text(
                j, i,
                f"{cell_label}\n{cm[i, j]:,}",
                ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    # metrics strip below
    strip = (
        f"F1={metrics['f1']:.3f}   Precision={metrics['precision']:.3f}   "
        f"Recall={metrics['recall']:.3f}   ROC-AUC={metrics['roc_auc']:.3f}   FPR={metrics['fpr']:.3f}"
    )
    ax.text(
        0.5, -0.18, strip,
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#444",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8f8f8", ec="#ccc", lw=0.8),
    )


def generate_comparison_plot(baseline_cm, baseline_m, mapek_cm, mapek_m, out_path):
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

    plot_single_cm(ax1, baseline_cm, "Baseline (XGBoost — No Augmentation)", FRAUD_CMAP, baseline_m)
    plot_single_cm(ax2, mapek_cm,    "MAPE-K + Adv-CTGAN (XGBoost)",    MAPEK_CMAP,  mapek_m)

    # Delta annotations
    delta_f1  = mapek_m["f1"]        - baseline_m["f1"]
    delta_rec = mapek_m["recall"]    - baseline_m["recall"]
    delta_fpr = mapek_m["fpr"]       - baseline_m["fpr"]

    def sign(v): return "+" if v >= 0 else ""

    delta_text = (
        f"Δ F1 {sign(delta_f1)}{delta_f1:+.3f}    "
        f"Δ Recall {sign(delta_rec)}{delta_rec:+.3f}    "
        f"Δ FPR {sign(delta_fpr)}{delta_fpr:+.3f}"
    )
    fig.text(
        0.5, 0.03, delta_text,
        ha="center", va="bottom",
        fontsize=12, fontweight="bold",
        color="#1a3c8f",
        bbox=dict(boxstyle="round,pad=0.5", fc="#e8eeff", ec="#1a3c8f", lw=1.2),
    )

    fig.suptitle(
        "Fraud Detection — Confusion Matrix Comparison",
        fontsize=18, fontweight="bold", y=0.97
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Confusion Matrix Generator for MAPE-K Paper")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/4] Loading dataset …")
    X, y, feature_cols = load_data()
    print(f"      {len(X)} samples, {len(feature_cols)} features, "
          f"{y.sum()} fraud / {(y==0).sum()} non-fraud")

    X_test, y_test = get_test_split(X, y)
    print(f"      Test split: {len(X_test)} samples")

    # 2. Baseline
    print("\n[2/4] Training baseline model (XGBoost, no augmentation) …")
    baseline_model = build_baseline(X, y)
    bl_pred  = baseline_model.predict(X_test)
    bl_proba = baseline_model.predict_proba(X_test)[:, 1]
    bl_cm    = confusion_matrix(y_test, bl_pred, labels=[0, 1])
    bl_m     = compute_metrics(y_test, bl_pred, bl_proba)
    print(f"      Baseline metrics: {bl_m}")

    # 3. MAPE-K promoted model
    print("\n[3/4] Loading MAPE-K promoted model …")
    model_path = AGENT_ROOT / "output" / "models" / "promoted_model.joblib"
    if not model_path.exists():
        print(f"  ⚠️  Promoted model not found at {model_path}")
        print("      Please run the pipeline first:  python agent/main.py")
        sys.exit(1)

    mapek_model = joblib.load(model_path)
    mk_pred     = mapek_model.predict(X_test)
    mk_proba    = mapek_model.predict_proba(X_test)[:, 1]
    mk_cm       = confusion_matrix(y_test, mk_pred, labels=[0, 1])
    mk_m        = compute_metrics(y_test, mk_pred, mk_proba)
    print(f"      MAPE-K metrics:   {mk_m}")

    # 4. Plot
    print("\n[4/4] Generating confusion matrix comparison plot …")
    out_path = PLOTS_DIR / "confusion_matrix_comparison.png"
    generate_comparison_plot(bl_cm, bl_m, mk_cm, mk_m, out_path)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<14}  {'Baseline':>10}  {'MAPE-K':>10}  {'Delta':>10}")
    print(f"  {'-'*50}")
    for k in ("f1", "precision", "recall", "roc_auc", "fpr"):
        d = mk_m[k] - bl_m[k]
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
        print(f"  {k:<14}  {bl_m[k]:>10.4f}  {mk_m[k]:>10.4f}  {arrow} {abs(d):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
