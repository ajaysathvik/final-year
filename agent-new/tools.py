from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ADVERSARIAL_ROBUSTNESS_CURVE_PATH,
    ADVERSARIAL_SUMMARY_PATH,
    ADVERSARIAL_TRAINING_SCRIPT_PATH,
    ADV_CTGAN_SCRIPT_PATH,
    BOOTSTRAP_TEST_PATH,
    BOOTSTRAP_TRAIN_PATH,
    CLASSIFIER_RESULTS_PATH,
    CTGAN_SCRIPT_PATH,
    EVAL_SCRIPT_PATH,
    TRAINING_SCRIPT_PATH,
    FIN_FRAUD_ROOT,
    INGESTED_POST_IDS_PATH,
    LABEL_SCRIPT_PATH,
    LABEL_WORKDIR,
    MIN_NEW_FRAUD_ROWS_TO_UPDATE,
    PREPARED_DATASET_PATH,
    PREPROCESS_SCRIPT_PATH,
    PREPROCESSED_SCRAPED_CSV_PATH,
    PREPROCESSED_SCRAPED_SUMMARY_PATH,
    ROBUSTNESS_CURVE_SCRIPT_PATH,
    ROBUSTNESS_RESULTS_PATH,
    ROBUSTNESS_SCRIPT_PATH,
    SCRAPED_LABELED_CSV_PATH,
    SCRAPED_LABELED_JSON_PATH,
    SCRAPER_SCRIPT_PATH,
    SCRAPER_WORKDIR,
    TEST_PATH,
    TRAIN_PATH,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEXT_COLUMNS = ("title", "body")
TARGET_COLUMN_CANDIDATES = ("annotation.is_fraud", "is_fraud")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}")
SCAM_LEXICON = {
    "upi",
    "otp",
    "kyc",
    "chargeback",
    "refund",
    "giftcard",
    "crypto",
    "telegram",
    "escrow",
    "impersonation",
    "phishing",
    "spoof",
    "wallet",
    "investment",
    "romance",
}


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(len(_load_csv(path)))


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_column(df: pd.DataFrame) -> str:
    for column in TARGET_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise KeyError(f"Missing target column. Checked: {TARGET_COLUMN_CANDIDATES}")


def _combined_text(df: pd.DataFrame) -> pd.Series:
    frames = [df[column].fillna("").astype(str) for column in TEXT_COLUMNS if column in df.columns]
    if not frames:
        return pd.Series([""] * len(df))
    text = frames[0]
    for column in frames[1:]:
        text = text.str.cat(column, sep=" ")
    return text.str.lower()


def _safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _normalized_counts(values: pd.Series) -> dict[str, float]:
    counts = values.value_counts(normalize=True)
    return {str(key): float(value) for key, value in counts.items()}


def _is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _append_new_rows_to_splits(
    labeled_path: Path,
    train_path: Path,
    test_path: Path,
    ingested_ids_path: Path,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    bootstrap_result = _ensure_bootstrap_splits(train_path, test_path)
    known_ids = _load_ingested_post_ids(ingested_ids_path)
    result: dict[str, Any] = {
        "new_rows_added": 0,
        "new_fraud_rows_added": 0,
        "train_rows_added": 0,
        "test_rows_added": 0,
        "bootstrap_used": bootstrap_result["bootstrap_used"],
        "bootstrap_sources": bootstrap_result["bootstrap_sources"],
        "min_new_fraud_rows_to_update": MIN_NEW_FRAUD_ROWS_TO_UPDATE,
    }

    if not labeled_path.exists():
        return result

    labeled_df = _load_csv(labeled_path)
    if labeled_df.empty:
        return result

    target_column = _target_column(labeled_df)
    labeled_df[target_column] = pd.to_numeric(labeled_df[target_column], errors="coerce")
    labeled_df = labeled_df[labeled_df[target_column].isin([0, 1])].copy()

    usable_col = "annotation.label_quality.usable_for_training"
    if usable_col in labeled_df.columns:
        labeled_df = labeled_df[labeled_df[usable_col].map(_is_truthy)].copy()

    post_id_col = "post_metadata.post_id"
    if post_id_col not in labeled_df.columns:
        return {
            "new_rows_added": 0,
            "train_rows_added": 0,
            "test_rows_added": 0,
            "reason": "missing post id column",
        }

    labeled_df[post_id_col] = labeled_df[post_id_col].fillna("").astype(str).str.strip()
    labeled_df = labeled_df[labeled_df[post_id_col] != ""].copy()

    new_rows = labeled_df[~labeled_df[post_id_col].isin(known_ids)].copy()
    if new_rows.empty:
        result["meets_update_threshold"] = False
        return result

    random_values = np.random.RandomState(42).rand(len(new_rows))
    test_mask = random_values < test_fraction
    if len(new_rows) == 1:
        test_mask[0] = False
    train_rows = new_rows.loc[~test_mask].copy()
    test_rows = new_rows.loc[test_mask].copy()

    if train_path.exists():
        train_df = _load_csv(train_path)
        train_columns = list(train_df.columns)
        train_df = pd.concat([train_df, train_rows.reindex(columns=train_columns)], ignore_index=True, sort=False)
        train_df.to_csv(train_path, index=False)
    if test_path.exists():
        test_df = _load_csv(test_path)
        test_columns = list(test_df.columns)
        test_df = pd.concat([test_df, test_rows.reindex(columns=test_columns)], ignore_index=True, sort=False)
        test_df.to_csv(test_path, index=False)

    appended_ids = set(new_rows[post_id_col].astype(str))
    _save_ingested_post_ids(ingested_ids_path, known_ids | appended_ids)
    fraud_rows = new_rows[new_rows[target_column].astype(int) == 1]
    result.update(
        {
            "new_rows_added": int(len(new_rows)),
            "new_fraud_rows_added": int(len(fraud_rows)),
            "train_rows_added": int(len(train_rows)),
            "test_rows_added": int(len(test_rows)),
            "class_balance_new_rows": _normalized_counts(new_rows[target_column].astype(int)),
            "meets_update_threshold": int(len(fraud_rows)) >= MIN_NEW_FRAUD_ROWS_TO_UPDATE,
        }
    )
    return result


def _load_ingested_post_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, list):
        return set()
    return {str(item).strip() for item in payload if str(item).strip()}


def _save_ingested_post_ids(path: Path, post_ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_ids = sorted(post_ids)
    path.write_text(json.dumps(ordered_ids, indent=2) + "\n", encoding="utf-8")


def _normalize_bootstrap_test_predictions(path: Path) -> pd.DataFrame:
    df = _load_csv(path)
    if "y_true" not in df.columns:
        raise KeyError(f"Missing y_true column in bootstrap test predictions: {path}")
    df = df.copy()
    df["annotation.is_fraud"] = pd.to_numeric(df["y_true"], errors="coerce")
    ordered_columns = ["annotation.is_fraud"] + [column for column in df.columns if column != "annotation.is_fraud"]
    return df.reindex(columns=ordered_columns)


def _existing_labeled_post_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = _load_csv(path)
    except Exception:
        return set()
    post_id_col = "post_metadata.post_id"
    if post_id_col not in df.columns:
        return set()
    return set(df[post_id_col].fillna("").astype(str).str.strip()) - {""}


def _prepare_incremental_posts_file(posts_path: Path, labeled_path: Path, output_path: Path) -> dict[str, Any]:
    posts_df = _load_csv(posts_path)
    posts_df.columns = posts_df.columns.str.lower().str.strip()
    if "post_id" not in posts_df.columns:
        raise KeyError(f"Missing post_id column in {posts_path}")
    posts_df["post_id"] = posts_df["post_id"].fillna("").astype(str).str.strip()
    known_ids = _existing_labeled_post_ids(labeled_path)
    new_posts_df = posts_df[~posts_df["post_id"].isin(known_ids)].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_posts_df.to_csv(output_path, index=False)
    return {
        "total_posts_available": int(len(posts_df)),
        "already_labeled_posts": int(len(posts_df) - len(new_posts_df)),
        "new_posts_to_label": int(len(new_posts_df)),
    }


def _merge_labeled_csv(existing_path: Path, incremental_path: Path, output_path: Path) -> int:
    frames: list[pd.DataFrame] = []
    if existing_path.exists():
        frames.append(_load_csv(existing_path))
    if incremental_path.exists() and incremental_path.stat().st_size > 0:
        frames.append(_load_csv(incremental_path))
    if not frames:
        pd.DataFrame().to_csv(output_path, index=False)
        return 0

    merged = pd.concat(frames, ignore_index=True, sort=False)
    post_id_col = "post_metadata.post_id"
    if post_id_col in merged.columns:
        merged[post_id_col] = merged[post_id_col].fillna("").astype(str).str.strip()
        merged = merged.drop_duplicates(subset=[post_id_col], keep="last")
    merged.to_csv(output_path, index=False)
    return int(len(merged))


def _merge_labeled_json(existing_path: Path, incremental_path: Path, output_path: Path) -> int:
    items: list[dict[str, Any]] = []
    for path in (existing_path, incremental_path):
        if not path.exists() or path.stat().st_size == 0:
            continue
        payload = _read_json(path)
        if isinstance(payload, list):
            items.extend(item for item in payload if isinstance(item, dict))

    if not items:
        output_path.write_text("[]\n", encoding="utf-8")
        return 0

    deduped: dict[str, dict[str, Any]] = {}
    ordered_fallback: list[dict[str, Any]] = []
    for item in items:
        post_meta = item.get("post_metadata", {})
        post_id = str(post_meta.get("post_id", "")).strip()
        if post_id:
            deduped[post_id] = item
        else:
            ordered_fallback.append(item)

    merged = ordered_fallback + list(deduped.values())
    output_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return int(len(merged))


def _ensure_bootstrap_splits(train_path: Path, test_path: Path) -> dict[str, Any]:
    train_exists = train_path.exists() and train_path.stat().st_size > 0
    test_exists = test_path.exists() and test_path.stat().st_size > 0
    if train_exists and test_exists:
        return {"bootstrap_used": False, "bootstrap_sources": {}}

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = _load_csv(BOOTSTRAP_TRAIN_PATH)
    test_df = _normalize_bootstrap_test_predictions(BOOTSTRAP_TEST_PATH)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return {
        "bootstrap_used": True,
        "bootstrap_sources": {
            "train": str(BOOTSTRAP_TRAIN_PATH),
            "test": str(BOOTSTRAP_TEST_PATH),
        },
    }


def jensen_shannon_divergence(p: pd.Series, q: pd.Series, bins: int = 10) -> float:
    p = pd.to_numeric(p, errors="coerce").dropna()
    q = pd.to_numeric(q, errors="coerce").dropna()
    if p.empty or q.empty:
        return 1.0

    merged = pd.concat([p, q], ignore_index=True)
    cut_points = np.unique(np.quantile(merged, np.linspace(0, 1, bins + 1)))
    if len(cut_points) < 3:
        return 0.0

    p_hist, _ = np.histogram(p, bins=cut_points, density=True)
    q_hist, _ = np.histogram(q, bins=cut_points, density=True)
    p_hist = p_hist + 1e-8
    q_hist = q_hist + 1e-8
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    m_hist = 0.5 * (p_hist + q_hist)
    return float(0.5 * np.sum(p_hist * np.log(p_hist / m_hist)) + 0.5 * np.sum(q_hist * np.log(q_hist / m_hist)))


def run_python_script(script_path: Path, cwd: Path, extra_env: dict[str, str] | None = None) -> CommandResult:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )
    return CommandResult(
        command=[sys.executable, str(script_path)],
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


class DatasetProfilerTool:
    name = "dataset_profiler"

    def run(self) -> dict[str, Any]:
        train_df = _load_csv(TRAIN_PATH if TRAIN_PATH.exists() else DATASET_PATH)
        test_df = _load_csv(TEST_PATH if TEST_PATH.exists() else DATASET_PATH)
        target = _target_column(train_df)

        train_text = _combined_text(train_df)
        test_text = _combined_text(test_df)
        train_tokens = Counter(token for text in train_text for token in TOKEN_RE.findall(text))
        test_tokens = Counter(token for text in test_text for token in TOKEN_RE.findall(text))

        scam_token_share_train = sum(train_tokens[token] for token in SCAM_LEXICON) / max(sum(train_tokens.values()), 1)
        scam_token_share_test = sum(test_tokens[token] for token in SCAM_LEXICON) / max(sum(test_tokens.values()), 1)
        slang_drift = abs(scam_token_share_test - scam_token_share_train)

        confidence_col = "annotation.fraud_confidence"
        usable_col = "annotation.label_quality.usable_for_training"
        label_noise = 0.0
        if confidence_col in train_df.columns:
            confidence = _safe_series(train_df, confidence_col).fillna(0.0)
            label_noise = float((confidence < 0.6).mean())
        elif usable_col in train_df.columns:
            usable = train_df[usable_col].astype(str).str.lower()
            label_noise = float((usable.isin({"false", "0", "no"})).mean())

        summary = {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "target_column": target,
            "class_balance_train": _normalized_counts(pd.to_numeric(train_df[target], errors="coerce").fillna(0).astype(int)),
            "class_balance_test": _normalized_counts(pd.to_numeric(test_df[target], errors="coerce").fillna(0).astype(int)),
            "label_noise_score": round(label_noise, 4),
            "slang_drift_score": round(slang_drift, 4),
            "top_train_tokens": train_tokens.most_common(15),
            "top_test_tokens": test_tokens.most_common(15),
            "prepared_dataset_exists": PREPARED_DATASET_PATH.exists(),
        }
        return summary


class ScraperTool:
    name = "reddit_scraper"

    def run(self) -> dict[str, Any]:
        before_posts = 0
        posts_path = SCRAPER_WORKDIR / "posts.csv"
        comments_path = SCRAPER_WORKDIR / "comments.csv"
        if posts_path.exists():
            before_posts = _csv_row_count(posts_path)
        result = run_python_script(SCRAPER_SCRIPT_PATH, SCRAPER_WORKDIR)
        after_posts = _csv_row_count(posts_path) if posts_path.exists() else before_posts
        after_comments = _csv_row_count(comments_path)
        return {
            "returncode": result.returncode,
            "posts_before": before_posts,
            "posts_after": after_posts,
            "new_posts_detected": max(after_posts - before_posts, 0),
            "comments_after": after_comments,
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
        }


class LabellerTool:
    name = "reddit_labeller"

    def run(self) -> dict[str, Any]:
        posts_path = LABEL_WORKDIR / "posts.csv"
        comments_path = LABEL_WORKDIR / "comments.csv"
        temp_dir = LABEL_WORKDIR / "outputs" / "incremental"
        temp_posts_path = temp_dir / "posts_to_label.csv"
        temp_csv_path = temp_dir / "new_annotations.csv"
        temp_json_path = temp_dir / "new_annotations.json"

        incremental = _prepare_incremental_posts_file(posts_path, SCRAPED_LABELED_CSV_PATH, temp_posts_path)
        if incremental["new_posts_to_label"] == 0:
            rows = _csv_row_count(SCRAPED_LABELED_CSV_PATH)
            return {
                "returncode": 0,
                "rows": rows,
                "new_posts_to_label": 0,
                "new_labels_created": 0,
                "output_csv": str(SCRAPED_LABELED_CSV_PATH),
                "output_json": str(SCRAPED_LABELED_JSON_PATH),
                "stdout_tail": "No unseen scraped posts. Skipping labeling.",
                "stderr_tail": "",
                **incremental,
            }

        result = subprocess.run(
            [
                sys.executable,
                str(LABEL_SCRIPT_PATH),
                "--mode",
                "full",
                "--posts-file",
                str(temp_posts_path),
                "--comments-file",
                str(comments_path),
                "--output-csv",
                str(temp_csv_path),
                "--output-json",
                str(temp_json_path),
            ],
            cwd=str(LABEL_WORKDIR),
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        new_rows = _csv_row_count(temp_csv_path)
        rows = _csv_row_count(SCRAPED_LABELED_CSV_PATH)
        if result.returncode == 0:
            rows = _merge_labeled_csv(SCRAPED_LABELED_CSV_PATH, temp_csv_path, SCRAPED_LABELED_CSV_PATH)
            _merge_labeled_json(SCRAPED_LABELED_JSON_PATH, temp_json_path, SCRAPED_LABELED_JSON_PATH)
        return {
            "returncode": result.returncode,
            "rows": rows,
            "new_labels_created": new_rows,
            "output_csv": str(SCRAPED_LABELED_CSV_PATH),
            "output_json": str(SCRAPED_LABELED_JSON_PATH),
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
            **incremental,
        }


class PostProcessorTool:
    name = "post_processor"

    def run(self) -> dict[str, Any]:
        result = subprocess.run(
            [
                sys.executable,
                str(PREPROCESS_SCRIPT_PATH),
                "--input-file",
                str(SCRAPED_LABELED_CSV_PATH),
                "--output-file",
                str(PREPROCESSED_SCRAPED_CSV_PATH),
                "--summary-file",
                str(PREPROCESSED_SCRAPED_SUMMARY_PATH),
            ],
            cwd=str(PREPROCESS_SCRIPT_PATH.parent),
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        summary: dict[str, Any] = {}
        if PREPROCESSED_SCRAPED_SUMMARY_PATH.exists():
            summary = json.loads(PREPROCESSED_SCRAPED_SUMMARY_PATH.read_text(encoding="utf-8"))
        rows = _csv_row_count(PREPROCESSED_SCRAPED_CSV_PATH)
        return {
            "returncode": result.returncode,
            "rows": rows,
            "output_csv": str(PREPROCESSED_SCRAPED_CSV_PATH),
            "summary_path": str(PREPROCESSED_SCRAPED_SUMMARY_PATH),
            "summary": summary,
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
        }


class DatasetAppendTool:
    name = "dataset_append"

    def run(self) -> dict[str, Any]:
        sync_result = _append_new_rows_to_splits(
            labeled_path=SCRAPED_LABELED_CSV_PATH,
            train_path=TRAIN_PATH,
            test_path=TEST_PATH,
            ingested_ids_path=INGESTED_POST_IDS_PATH,
        )
        return {
            "sync_result": sync_result,
            "train_rows": _csv_row_count(TRAIN_PATH),
            "test_rows": _csv_row_count(TEST_PATH),
        }


class LabelReviewTool:
    name = "label_review"

    def run(self, sample_size: int = 50) -> dict[str, Any]:
        df = _load_csv(TRAIN_PATH if TRAIN_PATH.exists() else DATASET_PATH)
        target = _target_column(df)
        sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
        confidence = _safe_series(sample, "annotation.fraud_confidence").fillna(0.5)
        ambiguous = sample[confidence.between(0.4, 0.6, inclusive="both")]
        target_distribution = _normalized_counts(pd.to_numeric(sample[target], errors="coerce").fillna(0).astype(int))
        return {
            "sample_size": int(len(sample)),
            "ambiguous_rows": int(len(ambiguous)),
            "ambiguous_ratio": float(len(ambiguous) / max(len(sample), 1)),
            "target_distribution": target_distribution,
            "recommendation": "relabel" if len(ambiguous) / max(len(sample), 1) > 0.15 else "keep",
        }


class BalanceSearchTool:
    name = "balance_search"

    def run(self, ratios: tuple[int, ...]) -> dict[str, Any]:
        source = _load_csv(PREPARED_DATASET_PATH if PREPARED_DATASET_PATH.exists() else DATASET_PATH)
        target = _target_column(source)
        class_counts = pd.to_numeric(source[target], errors="coerce").fillna(0).astype(int).value_counts().to_dict()
        fraud = int(class_counts.get(1, 0))
        non_fraud = int(class_counts.get(0, 0))
        if fraud == 0 or non_fraud == 0:
            return {"best_ratio": None, "candidates": [], "reason": "dataset is single-class"}

        candidates: list[dict[str, Any]] = []
        for ratio in ratios:
            expected_false_positive_penalty = abs((non_fraud / fraud) - ratio) / max(ratio, 1)
            utility = 1.0 - expected_false_positive_penalty
            candidates.append(
                {
                    "ratio": ratio,
                    "estimated_false_positive_penalty": round(expected_false_positive_penalty, 4),
                    "utility": round(utility, 4),
                }
            )
        best = max(candidates, key=lambda item: item["utility"])
        return {"best_ratio": best["ratio"], "candidates": candidates}


class SyntheticQualityTool:
    name = "synthetic_quality"

    def run(self, synthetic_path: Path) -> dict[str, Any]:
        real_df = _load_csv(PREPARED_DATASET_PATH if PREPARED_DATASET_PATH.exists() else DATASET_PATH)
        synthetic_df = _load_csv(synthetic_path)
        scores: dict[str, float] = {}
        js_values: list[float] = []
        for column in real_df.columns:
            if column not in synthetic_df.columns:
                continue
            if pd.api.types.is_numeric_dtype(real_df[column]) or column == "amount_numeric":
                value = jensen_shannon_divergence(real_df[column], synthetic_df[column])
                scores[column] = round(value, 4)
                js_values.append(value)
        mean_jsd = float(np.mean(js_values)) if js_values else 1.0
        return {
            "synthetic_path": str(synthetic_path),
            "mean_jsd": round(mean_jsd, 4),
            "column_jsd": scores,
            "accepted": mean_jsd <= 0.20,
        }


class CTGANTool:
    name = "ctgan_runner"

    def run(self) -> dict[str, Any]:
        result = run_python_script(CTGAN_SCRIPT_PATH, CTGAN_SCRIPT_PATH.parent)
        output_path = CTGAN_SCRIPT_PATH.parent / "ctgan_balanced_data.csv"
        return {
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
            "output_path": str(output_path),
            "exists": output_path.exists(),
        }


class AdversarialTrainerTool:
    name = "adversarial_trainer"

    def run(self, focus: str = "generic") -> dict[str, Any]:
        attack_surface = AttackSurfaceTool().run()
        env = {
            "ADV_CTGAN_FORCE_MODE": "novel_scam_fraud" if "fraud" in focus else "",
            "STRATEGY_RECOMMENDED_FOCUS": attack_surface["recommended_focus"],
        }
        training_result = run_python_script(
            ADVERSARIAL_TRAINING_SCRIPT_PATH,
            ADVERSARIAL_TRAINING_SCRIPT_PATH.parent,
            extra_env=env,
        )
        robustness_result = run_python_script(
            ROBUSTNESS_CURVE_SCRIPT_PATH,
            ROBUSTNESS_CURVE_SCRIPT_PATH.parent,
            extra_env=env,
        )

        summary_df = _load_csv(ADVERSARIAL_SUMMARY_PATH) if ADVERSARIAL_SUMMARY_PATH.exists() else pd.DataFrame()
        baseline_attack_f1 = 0.0
        adversarial_attack_f1 = 0.0
        clean_f1 = 0.0
        if not summary_df.empty and {"Model", "Test Set", "F1"}.issubset(summary_df.columns):
            baseline_row = summary_df[
                (summary_df["Model"] == "Baseline XGB") & (summary_df["Test Set"] == "FGSM Attack")
            ]
            adversarial_row = summary_df[
                (summary_df["Model"] == "Adversarial XGB") & (summary_df["Test Set"] == "FGSM Attack")
            ]
            clean_row = summary_df[
                (summary_df["Model"] == "Adversarial XGB") & (summary_df["Test Set"] == "Clean")
            ]
            if not baseline_row.empty:
                baseline_attack_f1 = _safe_float(baseline_row["F1"].iloc[0]) or 0.0
            if not adversarial_row.empty:
                adversarial_attack_f1 = _safe_float(adversarial_row["F1"].iloc[0]) or 0.0
            if not clean_row.empty:
                clean_f1 = _safe_float(clean_row["F1"].iloc[0]) or 0.0

        robustness_gain = adversarial_attack_f1 - baseline_attack_f1
        return {
            "focus": focus,
            "recommended_focus": attack_surface["recommended_focus"],
            "attack_surface": attack_surface,
            "training_returncode": training_result.returncode,
            "robustness_returncode": robustness_result.returncode,
            "training_stdout_tail": training_result.stdout[-3000:],
            "training_stderr_tail": training_result.stderr[-3000:],
            "robustness_stdout_tail": robustness_result.stdout[-3000:],
            "robustness_stderr_tail": robustness_result.stderr[-3000:],
            "summary_path": str(ADVERSARIAL_SUMMARY_PATH),
            "robustness_curve_path": str(ADVERSARIAL_ROBUSTNESS_CURVE_PATH),
            "summary_exists": ADVERSARIAL_SUMMARY_PATH.exists(),
            "robustness_curve_exists": ADVERSARIAL_ROBUSTNESS_CURVE_PATH.exists(),
            "baseline_attack_f1": round(baseline_attack_f1, 4),
            "adversarial_attack_f1": round(adversarial_attack_f1, 4),
            "adversarial_clean_f1": round(clean_f1, 4),
            "robustness_gain": round(robustness_gain, 4),
            "accepted": training_result.returncode == 0 and robustness_result.returncode == 0 and robustness_gain >= 0.0,
        }


class ClassifierTrainingTool:
    name = "classifier_training"

    def run(self) -> dict[str, Any]:
        env = {"FINFRAUD_DATASET_FILTER": "Original,CTGAN"}
        result = run_python_script(TRAINING_SCRIPT_PATH, TRAINING_SCRIPT_PATH.parent, extra_env=env)
        classifier_df = _load_csv(CLASSIFIER_RESULTS_PATH) if CLASSIFIER_RESULTS_PATH.exists() else pd.DataFrame()
        
        best_f1 = 0.0
        if not classifier_df.empty and "F1-Score" in classifier_df.columns:
            best_f1 = float(classifier_df["F1-Score"].max())
            
        return {
            "returncode": result.returncode,
            "best_f1": round(best_f1, 4),
            "results_path": str(CLASSIFIER_RESULTS_PATH),
            "stdout_tail": result.stdout[-3000:],
            "stderr_tail": result.stderr[-3000:],
        }


class EvaluationTool:
    name = "evaluation_runner"

    def run(self) -> dict[str, Any]:
        eval_result = run_python_script(EVAL_SCRIPT_PATH, FIN_FRAUD_ROOT)
        robustness_result = run_python_script(ROBUSTNESS_SCRIPT_PATH, FIN_FRAUD_ROOT)
        classifier_df = _load_csv(CLASSIFIER_RESULTS_PATH) if CLASSIFIER_RESULTS_PATH.exists() else pd.DataFrame()
        robustness_df = _load_csv(ROBUSTNESS_RESULTS_PATH) if ROBUSTNESS_RESULTS_PATH.exists() else pd.DataFrame()

        best_f1 = float(classifier_df["F1-Score"].max()) if "F1-Score" in classifier_df.columns else 0.0
        robustness = 0.0
        if "Robustness Score" in robustness_df.columns:
            robustness = float(1.0 - robustness_df["Robustness Score"].min())

        return {
            "eval_returncode": eval_result.returncode,
            "robustness_returncode": robustness_result.returncode,
            "best_f1": round(best_f1, 4),
            "robustness_score": round(robustness, 4),
            "classifier_results_path": str(CLASSIFIER_RESULTS_PATH),
            "robustness_results_path": str(ROBUSTNESS_RESULTS_PATH),
            "eval_stdout_tail": eval_result.stdout[-2000:],
            "robustness_stdout_tail": robustness_result.stdout[-2000:],
        }


class AttackSurfaceTool:
    name = "attack_surface"

    def run(self) -> dict[str, Any]:
        df = _load_csv(TEST_PATH if TEST_PATH.exists() else DATASET_PATH)
        text = _combined_text(df)
        lengths = text.str.split().map(len)
        long_form_ratio = float((lengths >= 120).mean())
        channel_counts: dict[str, int] = {}
        channel_col = "annotation.key_features.fraud_channel"
        if channel_col in df.columns:
            channel_counts = {str(key): int(value) for key, value in df[channel_col].fillna("unknown").value_counts().head(10).items()}
        return {
            "long_form_ratio": round(long_form_ratio, 4),
            "recommended_focus": "long-form text fraud" if long_form_ratio > 0.20 else "transactional short-form fraud",
            "channel_counts": channel_counts,
        }


def build_toolkit() -> dict[str, Any]:
    return {
        "reddit_scraper": ScraperTool(),
        "reddit_labeller": LabellerTool(),
        "post_processor": PostProcessorTool(),
        "dataset_append": DatasetAppendTool(),
        "dataset_profiler": DatasetProfilerTool(),
        "label_review": LabelReviewTool(),
        "balance_search": BalanceSearchTool(),
        "synthetic_quality": SyntheticQualityTool(),
        "ctgan_runner": CTGANTool(),
        "classifier_training": ClassifierTrainingTool(),
        "adversarial_trainer": AdversarialTrainerTool(),
        "evaluation_runner": EvaluationTool(),
        "attack_surface": AttackSurfaceTool(),
    }
