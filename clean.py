"""
clean.py
Remove trailing always-empty CSV columns, then split cleaned dataset in 3:2 ratio.

Usage:
    python clean.py input.csv [output_prefix]
"""
import csv
import sys
from pathlib import Path


def read_rows(input_path):
    with open(input_path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def get_keep_columns(rows):
    header = rows[0]
    data = rows[1:]
    last_used = 0

    for i, _ in enumerate(header):
        col_has_data = any((row[i].strip() if i < len(row) else "") for row in data)
        if col_has_data:
            last_used = i

    return last_used + 1


def trim_rows(rows, keep):
    cleaned = []
    for row in rows:
        padded = row + [""] * max(0, keep - len(row))
        cleaned.append(padded[:keep])
    return cleaned


def write_rows(output_path, rows):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, quoting=csv.QUOTE_MINIMAL).writerows(rows)


def clean_and_split(input_path, output_prefix=None):
    rows = read_rows(input_path)
    if not rows:
        print("Empty file.")
        sys.exit(1)

    input_path = Path(input_path)
    prefix = Path(output_prefix) if output_prefix else input_path.with_suffix("")

    keep = get_keep_columns(rows)
    cleaned_rows = trim_rows(rows, keep)

    header = cleaned_rows[0]
    data = cleaned_rows[1:]
    total_rows = len(data)
    first_count = (total_rows * 3) // 5
    second_count = total_rows - first_count

    cleaned_path = Path(f"{prefix}_cleaned.csv")
    split1_path = Path(f"{prefix}_part1.csv")
    split2_path = Path(f"{prefix}_part2.csv")

    write_rows(cleaned_path, cleaned_rows)
    write_rows(split1_path, [header] + data[:first_count])
    write_rows(split2_path, [header] + data[first_count:])

    print(f"Input file        : {input_path}")
    print(f"Columns kept      : {keep}")
    print(f"Columns removed   : {len(rows[0]) - keep}")
    print(f"Cleaned rows      : {total_rows}")
    print(f"Cleaned file      : {cleaned_path}")
    print(f"Split ratio       : 3:2")
    print(f"Part 1 rows       : {first_count}")
    print(f"Part 1 file       : {split1_path}")
    print(f"Part 2 rows       : {second_count}")
    print(f"Part 2 file       : {split2_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python clean.py input.csv [output_prefix]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) == 3 else None
    clean_and_split(input_csv, output_prefix)
