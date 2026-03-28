"""
Performance Matrix Generator for XGBoost on CTGAN-Augmented Data
-----------------------------------------------------------------
Generates confusion matrices, classification reports, and ROC curves
for each augmentation size (500, 700, 1000).
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_fscore_support,
    accuracy_score, roc_auc_score
)

# ── Paths & Constants ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from train_xgb_baseline import build_pipeline, LEAKY_COLUMNS, TARGET, RANDOM_STATE
except ImportError:
    TARGET = "annotation.is_fraud"
    RANDOM_STATE = 42
    LEAKY_COLUMNS = [TARGET, "annotation.fraud_type", "annotation.key_features.amount_mentioned"]
    LEAKY_COLUMNS.extend([
        "annotation.fraud_labels.transaction_upi_fraud", "annotation.fraud_labels.transaction_card_fraud",
        "annotation.fraud_labels.transaction_bank_transfer", "annotation.fraud_labels.commerce_nondelivery",
        "annotation.fraud_labels.commerce_fake_seller", "annotation.fraud_labels.credential_phishing",
        "annotation.fraud_labels.social_authority_scam", "annotation.fraud_labels.social_urgency_scam",
        "annotation.fraud_labels.meta_victim_story", "annotation.fraud_labels.meta_fraud_question",
    ])

    def build_pipeline(X):
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBClassifier

        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]
        prep = ColumnTransformer(transformers=[
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value=0)),
            ]), num_cols),
        ])
        model = XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, objective="binary:logistic",
            eval_metric="logloss", random_state=RANDOM_STATE,
        )
        return Pipeline([("preprocessor", prep), ("model", model)])

CTGAN_DIR = BASE_DIR / "ctgan"
ARTIFACT_DIR = CTGAN_DIR / "artifacts"
AUGMENTATION_SIZES = [500, 700, 1000]


def plot_confusion_matrix(ax, cm, title):
    """Plot a single confusion matrix on the given axes."""
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Fraud (0)", "Fraud (1)"], fontsize=10)
    ax.set_yticklabels(["Not Fraud (0)", "Fraud (1)"], fontsize=10)
    return im


def main():
    # ── 1. Load saved splits ───────────────────────────────────────
    train_path = ARTIFACT_DIR / "train_split.csv"
    test_path = ARTIFACT_DIR / "test_split.csv"

    if not train_path.exists() or not test_path.exists():
        print("ERROR: train/test splits not found. Run train_ctgan_augmentation.py first.")
        sys.exit(1)

    train_full = pd.read_csv(train_path)
    test_full = pd.read_csv(test_path)

    X_test = test_full.drop(columns=LEAKY_COLUMNS)
    y_test = test_full[TARGET].astype(int)

    train_not_fraud = train_full[train_full[TARGET] == 0].copy()
    X_train_0 = train_not_fraud.drop(columns=LEAKY_COLUMNS)

    cat_cols = X_train_0.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [col for col in X_train_0.columns if col not in cat_cols]
    X_train_0[cat_cols] = X_train_0[cat_cols].fillna("unknown")
    X_train_0[num_cols] = X_train_0[num_cols].fillna(0.0)

    train_fraud = train_full[train_full[TARGET] == 1].copy()

    # Storage for plots
    all_cm = {}
    all_reports = {}
    all_fpr = {}
    all_tpr = {}
    all_roc_auc = {}

    # ── 2. Train & collect per-size metrics ────────────────────────
    for size in AUGMENTATION_SIZES:
        syn_path = ARTIFACT_DIR / f"synthetic_0_{size}.csv"
        if not syn_path.exists():
            print(f"WARNING: {syn_path} not found, skipping size={size}")
            continue

        synthetic_0 = pd.read_csv(syn_path)
        print(f"\n=== Augmentation +{size} ===")

        combined_0 = pd.concat([X_train_0, synthetic_0], ignore_index=True)
        y_0 = pd.Series([0] * len(combined_0))

        target_count = min(len(combined_0), len(train_fraud))
        sampled_1 = train_fraud.sample(n=target_count, random_state=RANDOM_STATE)
        X_train_1 = sampled_1.drop(columns=LEAKY_COLUMNS)
        y_1 = pd.Series([1] * target_count)

        X_train = pd.concat([combined_0, X_train_1], ignore_index=True)
        y_train = pd.concat([y_0, y_1], ignore_index=True)
        shuffle_idx = X_train.sample(frac=1.0, random_state=RANDOM_STATE).index
        X_train = X_train.loc[shuffle_idx].reset_index(drop=True)
        y_train = y_train.loc[shuffle_idx].reset_index(drop=True)

        pipeline = build_pipeline(X_train)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]

        predictions = X_test.copy()
        predictions["y_true"] = y_test.values
        predictions["y_pred"] = y_pred
        predictions["y_score"] = y_score
        pred_file = ARTIFACT_DIR / f"test_predictions_{size}.csv"
        predictions.to_csv(pred_file, index=False)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        all_cm[size] = cm

        # Classification report
        report = classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"],
                                       output_dict=True)
        all_reports[size] = report
        print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        all_fpr[size] = fpr
        all_tpr[size] = tpr
        all_roc_auc[size] = auc(fpr, tpr)

    # ── 3. FIGURE 1: Confusion Matrices ────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle("Confusion Matrices — XGBoost on CTGAN-Augmented Data",
                  fontsize=16, fontweight="bold", y=1.02)
    for idx, size in enumerate(AUGMENTATION_SIZES):
        if size in all_cm:
            plot_confusion_matrix(axes[idx], all_cm[size], f"+{size} Synthetic Samples")
    fig1.tight_layout()
    cm_path = ARTIFACT_DIR / "confusion_matrices.png"
    fig1.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved: {cm_path}")

    # ── 4. FIGURE 2: ROC Curves ────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    for idx, size in enumerate(AUGMENTATION_SIZES):
        if size in all_fpr:
            ax2.plot(all_fpr[size], all_tpr[size], color=colors[idx], lw=2.5,
                     label=f"+{size} synthetic (AUC = {all_roc_auc[size]:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.set_title("ROC Curves — XGBoost on CTGAN-Augmented Data",
                  fontsize=14, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=11)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    roc_path = ARTIFACT_DIR / "roc_curves.png"
    fig2.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {roc_path}")

    # ── 5. FIGURE 3: Metrics Comparison Bar Chart ──────────────────
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_names))
    width = 0.25

    for idx, size in enumerate(AUGMENTATION_SIZES):
        if size not in all_reports:
            continue
        r = all_reports[size]
        vals = [
            r["accuracy"],
            r["Fraud"]["precision"],
            r["Fraud"]["recall"],
            r["Fraud"]["f1-score"],
            all_roc_auc[size],
        ]
        bars = ax3.bar(x + idx * width, vals, width, label=f"+{size} synthetic",
                       color=colors[idx], alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax3.set_ylabel("Score", fontsize=12)
    ax3.set_title("Performance Metrics Comparison — XGBoost on CTGAN-Augmented Data",
                  fontsize=14, fontweight="bold")
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(metrics_names, fontsize=11)
    ax3.set_ylim(0.90, 1.02)
    ax3.legend(fontsize=11)
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    bar_path = ARTIFACT_DIR / "metrics_comparison.png"
    fig3.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {bar_path}")

    # ── 6. Save detailed classification reports to JSON ────────────
    report_path = ARTIFACT_DIR / "classification_reports.json"
    # Convert numpy types for JSON serialization
    serializable = {}
    for size, r in all_reports.items():
        serializable[str(size)] = {
            k: {kk: float(vv) for kk, vv in v.items()} if isinstance(v, dict) else float(v)
            for k, v in r.items()
        }
    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=4)
    print(f"Saved: {report_path}")

    print("\n[SUCCESS] All performance matrices generated!")


if __name__ == "__main__":
    main()
