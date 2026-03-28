"""
XGBoost Evaluation on CTGAN-Augmented Data
--------------------------------------------
Reads the saved train/test splits and synthetic samples produced by
train_ctgan_augmentation.py, builds balanced training sets, trains
XGBoost, and reports classification metrics.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

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

    # Fill NaNs the same way as the CTGAN script
    cat_cols = X_train_0.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [col for col in X_train_0.columns if col not in cat_cols]
    X_train_0[cat_cols] = X_train_0[cat_cols].fillna("unknown")
    X_train_0[num_cols] = X_train_0[num_cols].fillna(0.0)

    train_fraud = train_full[train_full[TARGET] == 1].copy()

    results = {}

    # ── 2. For each augmentation size, build balanced set & train ──
    for size in AUGMENTATION_SIZES:
        syn_path = ARTIFACT_DIR / f"synthetic_0_{size}.csv"
        if not syn_path.exists():
            print(f"WARNING: {syn_path} not found, skipping size={size}")
            continue

        synthetic_0 = pd.read_csv(syn_path)
        print(f"\n--- Augmentation +{size} synthetic not-fraud rows ---")

        # Combine real not-fraud + synthetic not-fraud
        combined_0 = pd.concat([X_train_0, synthetic_0], ignore_index=True)
        y_0 = pd.Series([0] * len(combined_0))

        # Sample matching number of fraud rows (capped by available)
        target_count = min(len(combined_0), len(train_fraud))
        sampled_1 = train_fraud.sample(n=target_count, random_state=RANDOM_STATE)
        X_train_1 = sampled_1.drop(columns=LEAKY_COLUMNS)
        y_1 = pd.Series([1] * target_count)

        # Combine & shuffle
        X_train = pd.concat([combined_0, X_train_1], ignore_index=True)
        y_train = pd.concat([y_0, y_1], ignore_index=True)
        shuffle_idx = X_train.sample(frac=1.0, random_state=RANDOM_STATE).index
        X_train = X_train.loc[shuffle_idx].reset_index(drop=True)
        y_train = y_train.loc[shuffle_idx].reset_index(drop=True)

        print(f"Training XGBoost: {target_count} not-fraud + {target_count} fraud rows")
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

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", pos_label=1
        )
        roc_auc = roc_auc_score(y_test, y_score)

        metrics = {
            "synthetic_samples": size,
            "train_rows": len(X_train),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
        }
        results[size] = metrics
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")

    # ── 3. Save results ────────────────────────────────────────────
    json_path = ARTIFACT_DIR / "ctgan_evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    md_path = ARTIFACT_DIR / "CTGAN_REPORT.md"
    report = "# CTGAN Augmentation — XGBoost Evaluation Results\n\n"
    report += "| Synthetic Added | Total Train `not_fraud` | Total Train `fraud` | Accuracy | Precision | Recall | F1 Score | ROC AUC |\n"
    report += "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for size, m in results.items():
        count = m["train_rows"] // 2
        report += (
            f"| +{size} | {count} | {count} "
            f"| {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {m['f1']:.4f} | {m['roc_auc']:.4f} |\n"
        )

    with open(md_path, "w") as f:
        f.write(report)

    print(f"\n[SUCCESS] Evaluation complete! Results saved to:")
    print(f"  JSON → {json_path}")
    print(f"  Report → {md_path}")


if __name__ == "__main__":
    main()
