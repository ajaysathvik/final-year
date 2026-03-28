import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "artifacts" / "xgb_balanced_train_imbalanced_test"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(ARTIFACT_DIR / "mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


INPUT_FILE = BASE_DIR / "processed" / "data_preprocessed.csv"
METRICS_FILE = ARTIFACT_DIR / "metrics.json"
PREDICTIONS_FILE = ARTIFACT_DIR / "test_predictions.csv"
ROC_FILE = ARTIFACT_DIR / "roc_curve.png"
TRAIN_FILE = ARTIFACT_DIR / "train_balanced.csv"
TEST_FILE = ARTIFACT_DIR / "test_imbalanced.csv"
REPORT_FILE = BASE_DIR / "reports" / "XGBOOST_IMBALANCED_TEST_REPORT.md"

TARGET = "annotation.is_fraud"
RANDOM_STATE = 42
TEST_SIZE = 0.2

LEAKY_COLUMNS = [
    TARGET,
    "annotation.fraud_type",
    "annotation.key_features.amount_mentioned",
]

LEAKY_COLUMNS.extend(
    [
        "annotation.fraud_labels.transaction_upi_fraud",
        "annotation.fraud_labels.transaction_card_fraud",
        "annotation.fraud_labels.transaction_bank_transfer",
        "annotation.fraud_labels.commerce_nondelivery",
        "annotation.fraud_labels.commerce_fake_seller",
        "annotation.fraud_labels.credential_phishing",
        "annotation.fraud_labels.social_authority_scam",
        "annotation.fraud_labels.social_urgency_scam",
        "annotation.fraud_labels.meta_victim_story",
        "annotation.fraud_labels.meta_fraud_question",
    ]
)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
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
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def plot_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Balanced-Train / Imbalanced-Test XGBoost ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_FILE, dpi=150)
    plt.close()


def write_report(metrics: dict):
    report_text = f"""# XGBoost Imbalanced Test Report

Date: 2026-03-23

## Overview

This report summarizes an XGBoost baseline trained on a balanced training subset and evaluated on a real imbalanced held-out test set from [processed/data_preprocessed.csv]({INPUT_FILE.as_posix()}).

## Dataset Design

- Rows with `annotation.is_fraud = -1` were dropped.
- The full usable dataset kept the real class distribution for the test set.
- A stratified holdout test set was created first.
- The training portion was then balanced by downsampling `fraud` to match the available `not_fraud` rows.

Train/test sizes:
- Balanced train rows: {metrics['train_rows']}
- Imbalanced test rows: {metrics['test_rows']}
- Train class counts: {metrics['train_class_counts']}
- Test class counts: {metrics['test_class_counts']}

## Results

- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1: {metrics['f1']:.4f}
- ROC-AUC: {metrics['roc_auc']:.4f}

Confusion matrix:
- TN: {metrics['confusion_matrix'][0][0]}
- FP: {metrics['confusion_matrix'][0][1]}
- FN: {metrics['confusion_matrix'][1][0]}
- TP: {metrics['confusion_matrix'][1][1]}

## Artifacts

- Metrics JSON: [{METRICS_FILE.name}]({METRICS_FILE.as_posix()})
- Test predictions: [{PREDICTIONS_FILE.name}]({PREDICTIONS_FILE.as_posix()})
- ROC curve: [{ROC_FILE.name}]({ROC_FILE.as_posix()})
- Balanced train split: [{TRAIN_FILE.name}]({TRAIN_FILE.as_posix()})
- Imbalanced test split: [{TEST_FILE.name}]({TEST_FILE.as_posix()})
"""
    REPORT_FILE.write_text(report_text, encoding="utf-8")


def main():
    df = pd.read_csv(INPUT_FILE)
    usable = df[df[TARGET].isin([0, 1])].copy()

    train_full, test_full = train_test_split(
        usable,
        test_size=TEST_SIZE,
        stratify=usable[TARGET],
        random_state=RANDOM_STATE,
    )

    train_not_fraud = train_full[train_full[TARGET] == 0].copy()
    train_fraud = train_full[train_full[TARGET] == 1].sample(
        n=len(train_not_fraud), random_state=RANDOM_STATE
    )
    train_balanced = (
        pd.concat([train_not_fraud, train_fraud], ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    X_train = train_balanced.drop(columns=LEAKY_COLUMNS)
    y_train = train_balanced[TARGET].astype(int)
    X_test = test_full.drop(columns=LEAKY_COLUMNS)
    y_test = test_full[TARGET].astype(int)

    train_balanced.to_csv(TRAIN_FILE, index=False)
    test_full.to_csv(TEST_FILE, index=False)

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1
    )
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_score)
    accuracy = float((y_pred == y_test).mean())

    predictions = X_test.copy()
    predictions["y_true"] = y_test.values
    predictions["y_pred"] = y_pred
    predictions["y_score"] = y_score
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    metrics = {
        "train_rows": int(len(train_balanced)),
        "test_rows": int(len(test_full)),
        "train_class_counts": {
            str(k): int(v) for k, v in y_train.value_counts().sort_index().to_dict().items()
        },
        "test_class_counts": {
            str(k): int(v) for k, v in y_test.value_counts().sort_index().to_dict().items()
        },
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_columns": X_train.columns.tolist(),
    }

    with METRICS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    plot_roc(y_test, y_score)
    write_report(metrics)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
