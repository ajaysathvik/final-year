import json
import math
import os
from pathlib import Path
import joblib
import pandas as pd
from ctgan import CTGAN
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

BASE_DIR = Path(__file__).resolve().parents[3]
if BASE_DIR.name != "9k-dataset":
    BASE_DIR = Path(__file__).resolve().parents[2] # Fallback based on running dir

ARTIFACT_DIR = BASE_DIR / "artifacts" / "xgb_ctgan_3k"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(ARTIFACT_DIR / "mplconfig")

INPUT_FILE = BASE_DIR / "processed" / "data_binary_only_first_3000.csv"
TRAIN_FILE = ARTIFACT_DIR / "train_3k.csv"
TEST_FILE = ARTIFACT_DIR / "test_3k.csv"
SUMMARY_FILE = ARTIFACT_DIR / "summary_metrics_3k.json"
REPORT_FILE = BASE_DIR / "reports" / "XGB_CTGAN_3K_REPORT.md"
MATRIX_REPORT = BASE_DIR / "reports" / "PERFORMANCE_MATRIX_3K.md"
PLOT_FILE = ARTIFACT_DIR / "comparison_metrics_3k.png"
BASELINE_MODEL_FILE = ARTIFACT_DIR / "baseline" / "xgb_pipeline.joblib"
AUGMENTED_MODEL_FILE = ARTIFACT_DIR / "augmented" / "xgb_pipeline.joblib"

TARGET = "annotation.is_fraud"
RANDOM_STATE = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.08
TARGET_FRAUD_PER_NON_FRAUD = max(int(os.getenv("BALANCE_TARGET_FRAUD_PER_NON_FRAUD", "18")), 1)
BATCH_FRAUD_ROWS = max(int(os.getenv("BALANCE_BATCH_FRAUD_ROWS", "0")), 0)
BATCH_NON_FRAUD_ROWS = max(int(os.getenv("BALANCE_BATCH_NON_FRAUD_ROWS", "0")), 0)
REQUIRED_SYNTHETIC_NON_FRAUD = max(int(os.getenv("BALANCE_REQUIRED_SYNTHETIC_NON_FRAUD", "0")), 0)
CTGAN_CONFIG = {
    "epochs": 350,
    "batch_size": 16,
    "generator_dim": (64, 64),
    "discriminator_dim": (64, 64),
    "generator_lr": 2e-4,
    "discriminator_lr": 2e-4,
    "pac": 1,
    "enable_gpu": False,
    "verbose": True,
}

LEAKY_COLUMNS = [
    TARGET, "annotation.fraud_type", "annotation.key_features.amount_mentioned",
    "annotation.fraud_labels.transaction_upi_fraud", "annotation.fraud_labels.transaction_card_fraud",
    "annotation.fraud_labels.transaction_bank_transfer", "annotation.fraud_labels.commerce_nondelivery",
    "annotation.fraud_labels.commerce_fake_seller", "annotation.fraud_labels.credential_phishing",
    "annotation.fraud_labels.social_authority_scam", "annotation.fraud_labels.social_urgency_scam",
    "annotation.fraud_labels.meta_victim_story", "annotation.fraud_labels.meta_fraud_question"
]

def compute_scale_pos_weight(train_df: pd.DataFrame) -> float:
    y_train = train_df[TARGET].astype(int)
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    return negative_count / positive_count

def build_pipeline(X: pd.DataFrame, scale_pos_weight: float) -> Pipeline:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]),
                numeric_cols,
            ),
        ]
    )

    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df.drop(columns=LEAKY_COLUMNS, errors="ignore")
    y_train = train_df[TARGET].astype(int)
    X_test = test_df.drop(columns=LEAKY_COLUMNS, errors="ignore")
    y_test = test_df[TARGET].astype(int)

    scale_pos_weight = compute_scale_pos_weight(train_df)
    pipeline = build_pipeline(X_train, scale_pos_weight)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)

    metrics = {
        "accuracy": float((y_pred == y_test).mean()),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(auc(recall_curve, precision_curve)),
        "average_precision": float(average_precision_score(y_test, y_score)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "non_fraud_f1": float(report["0"]["f1-score"]),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report,
        "train_class_counts": {
            str(k): int(v) for k, v in y_train.value_counts().sort_index().to_dict().items()
        },
        "test_class_counts": {
            str(k): int(v) for k, v in y_test.value_counts().sort_index().to_dict().items()
        },
        "params": {
            "learning_rate": LEARNING_RATE,
            "scale_pos_weight": scale_pos_weight,
        },
    }

    predictions = X_test.copy()
    predictions["y_true"] = y_test.values
    predictions["y_pred"] = y_pred
    predictions["y_score"] = y_score
    curves = {
        "precision": precision_curve.tolist(),
        "recall": recall_curve.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }
    return metrics, predictions, curves, pipeline

def train_ctgan(train_non_fraud_features: pd.DataFrame):
    prepared = train_non_fraud_features.copy()
    discrete_columns = prepared.select_dtypes(include=["object"]).columns.tolist()
    if "annotation.key_features.has_amount" in prepared.columns:
        discrete_columns.append("annotation.key_features.has_amount")
    discrete_columns = list(dict.fromkeys(discrete_columns))

    numeric_columns = [col for col in prepared.columns if col not in discrete_columns]
    if discrete_columns:
        prepared[discrete_columns] = SimpleImputer(
            strategy="constant", fill_value="unknown"
        ).fit_transform(prepared[discrete_columns])
    if numeric_columns:
        prepared[numeric_columns] = SimpleImputer(strategy="constant", fill_value=0).fit_transform(
            prepared[numeric_columns]
        )

    model = CTGAN(**CTGAN_CONFIG)
    model.fit(prepared, discrete_columns=discrete_columns)
    return model


def save_pr_curve(curves: dict, output_file: Path, title: str):
    if plt is None:
        return
    plt.figure(figsize=(6.5, 5))
    plt.plot(curves["recall"], curves["precision"], color="#2a9d8f", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()

def save_roc_curve(curves: dict, output_file: Path, title: str):
    if plt is None:
        return
    plt.figure(figsize=(6.5, 5))
    plt.plot(curves["fpr"], curves["tpr"], color="#264653", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()

def write_report(subset_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, results: dict):
    baseline = results["baseline"]
    augmented = results["augmented"]
    synth_added = augmented["train_class_counts"]["0"] - baseline["train_class_counts"]["0"]
    lines = [
        "# XGBoost + CTGAN 3K Report",
        "",
        "Date: 2026-03-26",
        "",
        "## Setup",
        "",
        f"- Source data: `{INPUT_FILE.name}`.",
        f"- 3k dataset class counts: `{int((subset_df[TARGET] == 1).sum())} fraud`, `{int((subset_df[TARGET] == 0).sum())} non_fraud`.",
        f"- Train split: `{int((train_df[TARGET] == 1).sum())} fraud`, `{int((train_df[TARGET] == 0).sum())} non_fraud`.",
        f"- Test split: `{int((test_df[TARGET] == 1).sum())} fraud`, `{int((test_df[TARGET] == 0).sum())} non_fraud`.",
        "- CTGAN trains only on the real minority rows from the train split.",
        f"- Scraped batch counts: `{BATCH_FRAUD_ROWS} fraud`, `{BATCH_NON_FRAUD_ROWS} non_fraud`.",
        f"- Target balance: `1:{TARGET_FRAUD_PER_NON_FRAUD}` in `non_fraud:fraud` terms.",
        f"- Synthetic non-fraud rows requested from scraped batch delta: `{REQUIRED_SYNTHETIC_NON_FRAUD}`.",
        "",
        "## Results",
        "",
        "| Run | Scale Pos Weight | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Macro F1 | Non Fraud F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| baseline | {baseline['params']['scale_pos_weight']:.4f} | {baseline['accuracy']:.4f} | {baseline['precision']:.4f} | {baseline['recall']:.4f} | {baseline['f1']:.4f} | {baseline['roc_auc']:.4f} | {baseline['pr_auc']:.4f} | {baseline['macro_f1']:.4f} | {baseline['non_fraud_f1']:.4f} |",
        f"| augmented | {augmented['params']['scale_pos_weight']:.4f} | {augmented['accuracy']:.4f} | {augmented['precision']:.4f} | {augmented['recall']:.4f} | {augmented['f1']:.4f} | {augmented['roc_auc']:.4f} | {augmented['pr_auc']:.4f} | {augmented['macro_f1']:.4f} | {augmented['non_fraud_f1']:.4f} |",
        "",
        "## Interpretation",
        "",
        f"- Baseline scale_pos_weight: `{baseline['params']['scale_pos_weight']:.4f}`.",
        f"- Augmented scale_pos_weight: `{augmented['params']['scale_pos_weight']:.4f}`.",
        f"- Synthetic minority rows added: `{synth_added}`.",
        f"- Macro F1 change: `{augmented['macro_f1'] - baseline['macro_f1']:+.4f}`.",
        f"- Non fraud F1 change: `{augmented['non_fraud_f1'] - baseline['non_fraud_f1']:+.4f}`.",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

def write_matrix_report(results: dict):
    def row(label: str, run: dict):
        c0 = run["classification_report"]["0"]
        c1 = run["classification_report"]["1"]
        return (
            f"| {label} | {run['params']['scale_pos_weight']:.4f} | {run['accuracy']:.4f} | {run['precision']:.4f} | {run['recall']:.4f} | "
            f"{run['f1']:.4f} | {run['roc_auc']:.4f} | {run['pr_auc']:.4f} | {run['macro_f1']:.4f} | "
            f"{c0['precision']:.4f} | {c0['recall']:.4f} | {c0['f1-score']:.4f} | "
            f"{c1['precision']:.4f} | {c1['recall']:.4f} | {c1['f1-score']:.4f} |"
        )

    lines = [
        "# Performance Matrix 3K",
        "",
        "Date: 2026-03-26",
        "",
        "| Run | Scale Pos Weight | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Macro F1 | Non Fraud Precision | Non Fraud Recall | Non Fraud F1 | Fraud Precision | Fraud Recall | Fraud F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        row("baseline", results["baseline"]),
        row("augmented", results["augmented"]),
    ]
    MATRIX_REPORT.write_text("\n".join(lines), encoding="utf-8")

def plot_comparison(results: dict):
    if plt is None:
        return
    labels = ["baseline", "augmented"]
    accuracy_values = [results["baseline"]["accuracy"], results["augmented"]["accuracy"]]
    macro_values = [results["baseline"]["macro_f1"], results["augmented"]["macro_f1"]]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    plt.bar([i - width / 2 for i in x], accuracy_values, width=width, label="Accuracy", color="#264653")
    plt.bar([i + width / 2 for i in x], macro_values, width=width, label="Macro F1", color="#2a9d8f")
    plt.xticks(list(x), labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("3K Baseline vs CTGAN-Augmented Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=160)
    plt.close()

def main():
    subset_df = pd.read_csv(INPUT_FILE)

    train_df, test_df = train_test_split(
        subset_df,
        test_size=TEST_SIZE,
        stratify=subset_df[TARGET],
        random_state=RANDOM_STATE,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    results = {}

    baseline_metrics, baseline_predictions, baseline_curves, baseline_pipeline = evaluate_model(train_df, test_df)
    results["baseline"] = baseline_metrics
    baseline_dir = ARTIFACT_DIR / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    joblib.dump(baseline_pipeline, BASELINE_MODEL_FILE)
    baseline_predictions.to_csv(baseline_dir / "test_predictions.csv", index=False)
    with (baseline_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(baseline_metrics, handle, indent=2)
    save_pr_curve(baseline_curves, baseline_dir / "pr_auc_curve.png", "PR Curve 3K Baseline")
    save_roc_curve(baseline_curves, baseline_dir / "roc_auc_curve.png", "ROC Curve 3K Baseline")

    train_non_fraud = train_df[train_df[TARGET] == 0].copy().reset_index(drop=True)
    train_non_fraud_features = train_non_fraud.drop(columns=LEAKY_COLUMNS, errors="ignore")
    ctgan = train_ctgan(train_non_fraud_features)

    synthetic_rows_needed = REQUIRED_SYNTHETIC_NON_FRAUD
    if synthetic_rows_needed > 0:
        synthetic_features = ctgan.sample(synthetic_rows_needed).reset_index(drop=True)
        synthetic_df = synthetic_features.copy()
        synthetic_df[TARGET] = 0
        synthetic_df["annotation.fraud_type"] = "none"
        synthetic_df["annotation.key_features.amount_mentioned"] = "synthetic"
        for column in LEAKY_COLUMNS:
            if column not in synthetic_df.columns:
                synthetic_df[column] = 0
        synthetic_df = synthetic_df.reindex(columns=train_df.columns)
    else:
        synthetic_df = train_df.head(0).copy()

    augmented_train = pd.concat([train_df, synthetic_df], ignore_index=True).reset_index(drop=True)
    augmented_metrics, augmented_predictions, augmented_curves, augmented_pipeline = evaluate_model(augmented_train, test_df)
    results["augmented"] = augmented_metrics

    augmented_dir = ARTIFACT_DIR / "augmented"
    augmented_dir.mkdir(exist_ok=True)
    joblib.dump(augmented_pipeline, AUGMENTED_MODEL_FILE)
    synthetic_df.to_csv(augmented_dir / "synthetic_not_fraud.csv", index=False)
    augmented_train.to_csv(augmented_dir / "train_augmented.csv", index=False)
    augmented_predictions.to_csv(augmented_dir / "test_predictions.csv", index=False)
    with (augmented_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(augmented_metrics, handle, indent=2)
    save_pr_curve(augmented_curves, augmented_dir / "pr_auc_curve.png", "PR Curve 3K Augmented")
    save_roc_curve(augmented_curves, augmented_dir / "roc_auc_curve.png", "ROC Curve 3K Augmented")

    summary = {
        "train_file": str(TRAIN_FILE),
        "test_file": str(TEST_FILE),
        "target_fraud_per_non_fraud": TARGET_FRAUD_PER_NON_FRAUD,
        "batch_fraud_rows": BATCH_FRAUD_ROWS,
        "batch_non_fraud_rows": BATCH_NON_FRAUD_ROWS,
        "required_synthetic_non_fraud": REQUIRED_SYNTHETIC_NON_FRAUD,
        "synthetic_non_fraud_rows_added": int(len(synthetic_df)),
        "model_paths": {
            "baseline_pipeline": str(BASELINE_MODEL_FILE),
            "augmented_pipeline": str(AUGMENTED_MODEL_FILE),
        },
        "ctgan_config": CTGAN_CONFIG,
        "results": results,
    }
    with SUMMARY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    write_report(subset_df, train_df, test_df, results)
    write_matrix_report(results)
    plot_comparison(results)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
