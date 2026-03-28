import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "processed" / "data_binary_only.csv"
OUTPUT_DIR = BASE_DIR / "processed" / "controlled_real_ratio_split_3000"

TRAIN_FILE = OUTPUT_DIR / "train_ratio_preserved_3000.csv"
TEST_FILE = OUTPUT_DIR / "test_ratio_preserved_3000.csv"
REMAINING_FILE = OUTPUT_DIR / "remaining_unused.csv"
SUMMARY_FILE = OUTPUT_DIR / "split_summary.json"

RANDOM_STATE = 42
TOTAL_ROWS = 3000
TRAIN_RATIO = 0.8


def allocate_counts(total_rows: int, fraud_count: int, total_count: int):
    fraud_rows = round(total_rows * fraud_count / total_count)
    non_fraud_rows = total_rows - fraud_rows
    return fraud_rows, non_fraud_rows


def main():
    df = pd.read_csv(INPUT_FILE)
    fraud = df[df["annotation.is_fraud"] == 1].copy()
    non_fraud = df[df["annotation.is_fraud"] == 0].copy()

    train_total = int(TOTAL_ROWS * TRAIN_RATIO)
    test_total = TOTAL_ROWS - train_total

    train_fraud_n, train_non_fraud_n = allocate_counts(train_total, len(fraud), len(df))
    test_fraud_n, test_non_fraud_n = allocate_counts(test_total, len(fraud), len(df))

    train_fraud = fraud.sample(n=train_fraud_n, random_state=RANDOM_STATE)
    remaining_fraud = fraud.drop(train_fraud.index)

    train_non_fraud = non_fraud.sample(n=train_non_fraud_n, random_state=RANDOM_STATE)
    remaining_non_fraud = non_fraud.drop(train_non_fraud.index)

    test_fraud = remaining_fraud.sample(n=test_fraud_n, random_state=RANDOM_STATE)
    remaining_fraud = remaining_fraud.drop(test_fraud.index)

    test_non_fraud = remaining_non_fraud.sample(n=test_non_fraud_n, random_state=RANDOM_STATE)
    remaining_non_fraud = remaining_non_fraud.drop(test_non_fraud.index)

    train_df = (
        pd.concat([train_fraud, train_non_fraud], ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat([test_fraud, test_non_fraud], ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    remaining_df = (
        pd.concat([remaining_fraud, remaining_non_fraud], ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    remaining_df.to_csv(REMAINING_FILE, index=False)

    summary = {
        "input_file": str(INPUT_FILE),
        "train_file": str(TRAIN_FILE),
        "test_file": str(TEST_FILE),
        "remaining_file": str(REMAINING_FILE),
        "random_state": RANDOM_STATE,
        "total_rows": TOTAL_ROWS,
        "train_total": train_total,
        "test_total": test_total,
        "base_counts": {"fraud": int(len(fraud)), "non_fraud": int(len(non_fraud))},
        "base_ratio": {
            "fraud_to_non_fraud": float(len(fraud) / len(non_fraud)),
            "fraud_share": float(len(fraud) / len(df)),
            "non_fraud_share": float(len(non_fraud) / len(df)),
        },
        "actual_train_counts": {
            str(k): int(v)
            for k, v in train_df["annotation.is_fraud"].value_counts().sort_index().to_dict().items()
        },
        "actual_test_counts": {
            str(k): int(v)
            for k, v in test_df["annotation.is_fraud"].value_counts().sort_index().to_dict().items()
        },
        "actual_remaining_counts": {
            str(k): int(v)
            for k, v in remaining_df["annotation.is_fraud"].value_counts().sort_index().to_dict().items()
        },
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "remaining_rows": int(len(remaining_df)),
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
