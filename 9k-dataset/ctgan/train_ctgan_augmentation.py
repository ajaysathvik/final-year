"""
CTGAN Training & Synthetic Data Generation
-------------------------------------------
Trains a CTGAN on not-fraud rows and generates synthetic samples
at multiple augmentation sizes. Outputs are saved to ctgan/artifacts/.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from ctgan import CTGAN
except ImportError:
    print("Please install ctgan: pip install ctgan")
    sys.exit(1)

# ── Paths & Constants ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

try:
    from train_xgb_baseline import LEAKY_COLUMNS, TARGET, RANDOM_STATE, TEST_SIZE
except ImportError:
    TARGET = "annotation.is_fraud"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    LEAKY_COLUMNS = [TARGET, "annotation.fraud_type", "annotation.key_features.amount_mentioned"]
    LEAKY_COLUMNS.extend([
        "annotation.fraud_labels.transaction_upi_fraud", "annotation.fraud_labels.transaction_card_fraud",
        "annotation.fraud_labels.transaction_bank_transfer", "annotation.fraud_labels.commerce_nondelivery",
        "annotation.fraud_labels.commerce_fake_seller", "annotation.fraud_labels.credential_phishing",
        "annotation.fraud_labels.social_authority_scam", "annotation.fraud_labels.social_urgency_scam",
        "annotation.fraud_labels.meta_victim_story", "annotation.fraud_labels.meta_fraud_question",
    ])

CTGAN_DIR = BASE_DIR / "ctgan"
ARTIFACT_DIR = CTGAN_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE = BASE_DIR / "processed" / "data_preprocessed.csv"

AUGMENTATION_SIZES = [500, 700, 1000]


def main():
    # ── 1. Load & split ────────────────────────────────────────────
    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    usable = df[df[TARGET].isin([0, 1])].copy()

    train_full, test_full = train_test_split(
        usable, test_size=TEST_SIZE, stratify=usable[TARGET], random_state=RANDOM_STATE
    )

    # Save the train/test split so the evaluation script uses the SAME split
    train_full.to_csv(ARTIFACT_DIR / "train_split.csv", index=False)
    test_full.to_csv(ARTIFACT_DIR / "test_split.csv", index=False)
    print(f"Saved train ({len(train_full)}) and test ({len(test_full)}) splits to artifacts/")

    # ── 2. Prepare non-fraud training features ─────────────────────
    train_not_fraud = train_full[train_full[TARGET] == 0].copy()
    X_train_0 = train_not_fraud.drop(columns=LEAKY_COLUMNS)

    cat_cols = X_train_0.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [col for col in X_train_0.columns if col not in cat_cols]

    X_train_0[cat_cols] = X_train_0[cat_cols].fillna("unknown")
    X_train_0[num_cols] = X_train_0[num_cols].fillna(0.0)

    discrete_columns = cat_cols + [
        col for col in num_cols
        if X_train_0[col].nunique() < 10 and not pd.api.types.is_float_dtype(X_train_0[col])
    ]
    discrete_columns = list(set(discrete_columns))

    # ── 3. Train CTGAN ─────────────────────────────────────────────
    print(f"Fitting CTGAN on {len(X_train_0)} real not-fraud rows...")
    print(f"Discrete columns: {discrete_columns}")

    ctgan = CTGAN(
        epochs=350,
        generator_dim=(64, 64),
        discriminator_dim=(64, 64),
    )

    ctgan.fit(X_train_0, discrete_columns)

    model_path = ARTIFACT_DIR / "ctgan_model.pkl"
    ctgan.save(str(model_path))
    print(f"CTGAN model saved to {model_path}")

    # ── 4. Generate synthetic samples at each size ─────────────────
    for size in AUGMENTATION_SIZES:
        print(f"Generating {size} synthetic not-fraud samples...")
        synthetic = ctgan.sample(size)
        out_path = ARTIFACT_DIR / f"synthetic_0_{size}.csv"
        synthetic.to_csv(out_path, index=False)
        print(f"  → saved to {out_path}")

    print("\n[SUCCESS] CTGAN training & generation completed!")
    print("Next step: run  evaluate_xgb_ctgan.py  to train & evaluate the XGBoost classifier.")


if __name__ == "__main__":
    main()
