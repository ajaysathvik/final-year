"""
Remove rows with fraud label -1 from an annotation CSV.

Usage:
    python "Data labeling/remove_negative_one_labels.py" \
        --input "Data labeling/outputs/annotations_20260321_173533.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_COLS = ["is_fraud", "annotation.is_fraud"]


def resolve_target_col(df: pd.DataFrame) -> str:
    """Return the first available fraud label column."""
    for col in TARGET_COLS:
        if col in df.columns:
            return col
    raise KeyError(f"Expected one of {TARGET_COLS}, but none were found.")


def build_output_path(input_path: Path, output_path: str | None) -> Path:
    """Create a default output path when one is not provided."""
    if output_path:
        return Path(output_path)
    return input_path.with_name(f"{input_path.stem}_binary_only{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove rows where the fraud label is -1."
    )
    parser.add_argument("--input", required=True, help="Path to the source CSV")
    parser.add_argument(
        "--output",
        help="Path to write the cleaned CSV. Defaults to <input>_binary_only.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = build_output_path(input_path, args.output)

    df = pd.read_csv(input_path, low_memory=False)
    target_col = resolve_target_col(df)

    original_rows = len(df)
    cleaned = df[df[target_col] != -1].copy()
    removed_rows = original_rows - len(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print(f"Input: {input_path}")
    print(f"Target column: {target_col}")
    print(f"Removed rows: {removed_rows}")
    print(f"Remaining rows: {len(cleaned)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
