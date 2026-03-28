"""
Shared utilities for all classifier models.
Handles data loading, preprocessing, feature alignment, and metric computation.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASETS = {
    "Original":  os.path.join(ROOT, "original_dataset", "final1.csv"),
    "CTGAN":     os.path.join(ROOT, "ctgan", "ctgan_balanced_data.csv"),
    "Adv-CTGAN": os.path.join(ROOT, "adversarial_training", "adv_balanced_data.csv"),
}

OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    if "title" in df.columns:
        df = df.drop(columns=["title"])
    if "body" in df.columns:
        df = df.drop(columns=["body"])
    if "is_fraud" in df.columns:
        df = df[df["is_fraud"] != -1].copy()
    y = df["is_fraud"].values
    X = df.drop(columns=["is_fraud"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def align_features(X_train, X_test):
    missing = set(X_train.columns) - set(X_test.columns)
    for c in missing:
        X_test[c] = 0
    return X_train, X_test[X_train.columns]


def get_test_set():
    """Returns a held-out 20% test set from original data."""
    X_orig, y_orig = load_and_preprocess(DATASETS["Original"])
    _, X_test, _, y_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
    )
    return X_test, y_test


def prepare_data(ds_path, X_test_raw):
    """Load a dataset, align features with test set, impute and scale."""
    X_train_raw, y_train = load_and_preprocess(ds_path)
    X_train_raw, X_test_aligned = align_features(X_train_raw, X_test_raw.copy())

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp  = imputer.transform(X_test_aligned)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_test  = scaler.transform(X_test_imp).astype(np.float32)

    return X_train, y_train, X_test


def compute_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float("nan")
    try:
        ll = log_loss(y_true, y_prob)
    except:
        ll = float("nan")
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC-ROC":   round(auc, 4),
        "Log Loss":  round(ll, 4),
    }
