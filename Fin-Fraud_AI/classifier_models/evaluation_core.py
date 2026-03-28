import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "Original": os.path.join(ROOT, "original_dataset", "final1.csv"),
    "CTGAN": os.path.join(ROOT, "CTGAN", "ctgan_balanced_data.csv"),
    "Adv-CTGAN": os.path.join(ROOT, "adversarial_training", "adv_balanced_data.csv"),
}


def get_selected_datasets():
    selected = os.environ.get("FINFRAUD_DATASET_FILTER", "").strip()
    if not selected:
        return DATASETS
    allowed = {item.strip() for item in selected.split(",") if item.strip()}
    return {name: path for name, path in DATASETS.items() if name in allowed}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0,
}

CLASSIFIERS = {
    "XGBoost": XGBClassifier(**XGB_PARAMS),
}


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    for col in ["title", "body"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    if "is_fraud" in df.columns:
        df = df[df["is_fraud"] != -1].copy()
    y = df["is_fraud"].values
    X = df.drop(columns=["is_fraud"])
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def align_features(X_train, X_test):
    missing = set(X_train.columns) - set(X_test.columns)
    for col in missing:
        X_test[col] = 0
    return X_train, X_test[X_train.columns]


def get_master_test_split(test_size=0.2, random_state=42):
    X_orig, y_orig = load_and_preprocess(DATASETS["Original"])
    return train_test_split(
        X_orig, y_orig, test_size=test_size, random_state=random_state, stratify=y_orig
    )


def prepare_scaled_data(ds_path, X_test_raw):
    X_train_raw, y_train = load_and_preprocess(ds_path)
    X_train_raw, X_test_aligned = align_features(X_train_raw, X_test_raw.copy())

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_aligned)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_test = scaler.transform(X_test_imp).astype(np.float32)

    return X_train, y_train, X_test, X_train_raw.columns.tolist()


def clone_classifier(clf):
    return clf.__class__(**clf.get_params())


def safe_predict_proba(clf, X):
    try:
        return clf.predict_proba(X)[:, 1]
    except Exception:
        return clf.predict(X).astype(float)


def compute_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    try:
        ll = log_loss(y_true, y_prob)
    except Exception:
        ll = float("nan")
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC-ROC": round(auc, 4),
        "Log Loss": round(ll, 4),
    }


def min_max_normalize(series, invert=False):
    values = series.astype(float)
    vmin = values.min()
    vmax = values.max()
    if np.isclose(vmax, vmin):
        normalized = pd.Series(np.ones(len(values)), index=series.index)
    else:
        normalized = (values - vmin) / (vmax - vmin)
    if invert:
        normalized = 1 - normalized
    return normalized
