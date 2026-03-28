import os

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_IN = os.path.join(ROOT, "original_dataset", "final1.csv")
DATA_OUT = os.path.join(ROOT, "CTGAN", "ctgan_balanced_data.csv")
NOVEL_SCAM_SEEDS = os.path.join(ROOT, "original_dataset", "new_scam_seeds.csv")
SYNTHETIC_NOVEL_OUT = os.path.join(ROOT, "CTGAN", "synthetic_new_scams_ctgan.csv")

TEXT_COLS = ["title", "body"]
MIN_TARGET_ROWS = 20


def _prepare_training_slice(df):
    working = df.drop(columns=[col for col in TEXT_COLS if col in df.columns], errors="ignore").copy()
    working["amount_numeric"] = pd.to_numeric(working.get("amount_numeric"), errors="coerce")
    if "amount_numeric" in working.columns:
        if working["amount_numeric"].notna().any():
            working["amount_numeric"] = working["amount_numeric"].fillna(working["amount_numeric"].median())
        else:
            working["amount_numeric"] = 0.0
    return working


def _related_fraud_rows(full_df, seeds_df):
    fraud_df = full_df[full_df["is_fraud"] == 1].copy()
    if seeds_df.empty:
        return fraud_df.iloc[0:0].copy()

    types = set(seeds_df.get("annotation.fraud_type", pd.Series(dtype=str)).dropna().astype(str))
    channels = set(seeds_df.get("annotation.key_features.fraud_channel", pd.Series(dtype=str)).dropna().astype(str))
    payments = set(seeds_df.get("annotation.key_features.payment_method", pd.Series(dtype=str)).dropna().astype(str))

    mask = pd.Series(False, index=fraud_df.index)
    if types:
        mask |= fraud_df.get("annotation.fraud_type", pd.Series(index=fraud_df.index, dtype=object)).astype(str).isin(types)
    if channels:
        mask |= fraud_df.get("annotation.key_features.fraud_channel", pd.Series(index=fraud_df.index, dtype=object)).astype(str).isin(channels)
    if payments:
        mask |= fraud_df.get("annotation.key_features.payment_method", pd.Series(index=fraud_df.index, dtype=object)).astype(str).isin(payments)
    return fraud_df[mask].copy()


def _choose_generation_target(full_df):
    if os.path.exists(NOVEL_SCAM_SEEDS):
        seeds_df = pd.read_csv(NOVEL_SCAM_SEEDS)
        seeds_df = seeds_df[seeds_df.get("is_fraud", 0) == 1].copy()
    else:
        seeds_df = pd.DataFrame()

    if not seeds_df.empty:
        target_df = pd.concat([seeds_df, _related_fraud_rows(full_df, seeds_df)], ignore_index=True).drop_duplicates()
        mode = "novel_scam_fraud"
        synthetic_label = 1
        sample_size = min(500, max(100, len(seeds_df) * 4))
        if len(target_df) >= MIN_TARGET_ROWS:
            return target_df, mode, synthetic_label, sample_size
        print(
            f"Novel scam seed set too small for stable CTGAN ({len(target_df)} rows). "
            "Falling back to legacy non-fraud augmentation."
        )

    fallback_df = full_df[full_df["is_fraud"] == 0].copy()
    return fallback_df, "legacy_non_fraud", 0, 500


def main():
    print("Loading Original Dataset...")
    df = pd.read_csv(DATA_IN)
    df.replace("unknown", pd.NA, inplace=True)

    target_df, mode, synthetic_label, sample_size = _choose_generation_target(df)
    if target_df.empty:
        raise RuntimeError("No rows available for CTGAN training.")

    train_df = _prepare_training_slice(target_df)

    discrete_columns = []
    for col in train_df.columns:
        if col == "amount_numeric":
            continue
        if train_df[col].dtype == "object" or train_df[col].dtype == "bool" or train_df[col].nunique(dropna=False) < 10:
            discrete_columns.append(col)

    print(f"Training Standard CTGAN on {len(train_df)} rows ({mode})...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    if "amount_numeric" in train_df.columns:
        metadata.update_column(column_name="amount_numeric", sdtype="numerical")

    synthesizer = CTGANSynthesizer(metadata, epochs=300, batch_size=50)
    synthesizer.fit(train_df)

    print(f"Generating {sample_size} synthetic rows for mode={mode}...")
    synthetic_df = synthesizer.sample(num_rows=sample_size)
    synthetic_df["is_fraud"] = synthetic_label
    for col in TEXT_COLS:
        synthetic_df[col] = pd.NA

    if mode == "novel_scam_fraud":
        synthetic_df.to_csv(SYNTHETIC_NOVEL_OUT, index=False)
        print(f"Saved synthetic novel scam rows to {SYNTHETIC_NOVEL_OUT}")
    elif os.path.exists(SYNTHETIC_NOVEL_OUT):
        os.remove(SYNTHETIC_NOVEL_OUT)

    balanced_df = pd.concat([df, synthetic_df], ignore_index=True)
    balanced_df.to_csv(DATA_OUT, index=False)
    print(f"Saved CTGAN balanced data to {DATA_OUT}")


if __name__ == "__main__":
    main()
