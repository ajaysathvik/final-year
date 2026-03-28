from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from agent.config import (
    MAIN_CSV_PATH,
    NOVEL_SCAM_REPORT_PATH,
    NOVEL_SCAM_SEEDS_PATH,
    ORIGINAL_DATASET_DIR,
    ORIGINAL_DATASET_PATH,
    SCRAPED_LABELED_CSV_PATH,
    SCRAPED_POSTS_PATH,
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
)
from sklearn.model_selection import train_test_split


TRAINING_COLUMNS = [
    "title",
    "body",
    "is_fraud",
    "amount_numeric",
    "post_metadata.subreddit",
    "post_metadata.num_comments",
    "annotation.fraud_confidence",
    "annotation.fraud_type",
    "annotation.fraud_labels.transaction_upi_fraud",
    "annotation.fraud_labels.transaction_card_fraud",
    "annotation.fraud_labels.transaction_bank_transfer",
    "annotation.fraud_labels.transaction_nondelivery",
    "annotation.fraud_labels.transaction_fake_seller",
    "annotation.fraud_labels.commerce_nondelivery",
    "annotation.fraud_labels.commerce_fake_seller",
    "annotation.fraud_labels.credential_phishing",
    "annotation.fraud_labels.social_authority_scam",
    "annotation.fraud_labels.social_urgency_scam",
    "annotation.fraud_labels.meta_victim_story",
    "annotation.fraud_labels.meta_fraud_question",
    "annotation.key_features.payment_method",
    "annotation.key_features.fraud_channel",
    "annotation.key_features.victim_action",
    "annotation.key_features.request_type",
    "annotation.key_features.impersonated_entity",
    "annotation.key_features.currency",
    "annotation.key_features.urgency_level",
    "annotation.psychological_tactics.urgency",
    "annotation.psychological_tactics.fear",
    "annotation.psychological_tactics.authority",
    "annotation.psychological_tactics.reward",
    "annotation.community_signals.num_comments",
    "annotation.community_signals.scam_confirmations",
    "annotation.community_signals.not_scam_claims",
    "annotation.community_signals.advice_requests",
    "annotation.label_quality.confidence_bucket",
    "annotation.label_quality.usable_for_training",
    "annotation.gan_quality.suitable_for_gan",
    "annotation.gan_quality.quality_score",
    "post_metadata.body_length",
    "post_metadata.body_language",
    "post_metadata.scam_confirmations",
    "post_metadata.not_scam_claims",
    "post_metadata.advice_requests",
]

FRAUD_SIGNATURE_COLUMNS = [
    "annotation.fraud_type",
    "annotation.key_features.payment_method",
    "annotation.key_features.fraud_channel",
    "annotation.key_features.request_type",
    "annotation.key_features.impersonated_entity",
]


@dataclass
class PreparedDataset:
    output_path: Path
    row_count: int
    class_counts: dict[str, int]
    columns: list[str]
    new_scam_row_count: int
    new_scam_seed_path: Path
    new_scam_report_path: Path
    base_row_count: int
    labeled_scraped_row_count: int
    labeled_scraped_class_counts: dict[str, int]
    labeled_scraped_usable_count: int
    new_scam_signatures: list[dict]

    def to_memory_dict(self) -> dict:
        payload = asdict(self)
        payload["output_path"] = str(self.output_path)
        payload["new_scam_seed_path"] = str(self.new_scam_seed_path)
        payload["new_scam_report_path"] = str(self.new_scam_report_path)
        return payload


def _first_available(df: pd.DataFrame, candidates: Iterable[str], default=None):
    for column in candidates:
        if column in df.columns:
            series = df[column]
            if series.notna().any():
                return series
    return default


def _to_boolish(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    if series.dtype == object:
        lowered = series.astype(str).str.strip().str.lower()
        if lowered.isin(mapping.keys()).any():
            return lowered.map(mapping).where(~series.isna(), other=pd.NA)
    return series


def _extract_amount(series: pd.Series) -> pd.Series:
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _stable_row_fingerprint(row: pd.Series | dict, columns: Iterable[str]) -> str:
    parts: list[str] = []
    for column in columns:
        value = row.get(column, "")
        if pd.isna(value):
            value = ""
        parts.append(f"{column}={str(value).strip().lower()}")
    return "|".join(parts)


def _fingerprint_columns(df: pd.DataFrame) -> list[str]:
    preferred_columns = [
        "annotation.is_fraud",
        "annotation.fraud_type",
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
        "annotation.key_features.payment_method",
        "annotation.key_features.fraud_channel",
        "annotation.key_features.victim_action",
        "annotation.key_features.request_type",
        "annotation.key_features.impersonated_entity",
        "annotation.key_features.amount_mentioned",
        "annotation.key_features.currency",
        "annotation.key_features.urgency_level",
        "annotation.key_features.amount_normalized",
        "annotation.key_features.has_amount",
        "annotation.psychological_tactics.urgency",
        "annotation.psychological_tactics.fear",
        "annotation.psychological_tactics.authority",
        "annotation.psychological_tactics.reward",
    ]
    available = [column for column in preferred_columns if column in df.columns]
    if available:
        return available
    return list(df.columns)


def _known_annotation_keys(df: pd.DataFrame) -> set[str]:
    known_keys: set[str] = set()

    if "post_metadata.post_id" in df.columns:
        ids = df["post_metadata.post_id"].fillna("").astype(str).str.strip()
        known_keys.update(ids[ids != ""])

    fingerprint_columns = _fingerprint_columns(df)
    if fingerprint_columns:
        for _, row in df.iterrows():
            fingerprint = _stable_row_fingerprint(row, fingerprint_columns)
            if fingerprint:
                known_keys.add(fingerprint)

    return known_keys


def _normalize_base_annotations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["title"] = _first_available(
        df,
        ["title", "post_metadata.title"],
        default=pd.Series([pd.NA] * len(df)),
    )
    df["body"] = _first_available(
        df,
        ["body", "post_metadata.body"],
        default=pd.Series([pd.NA] * len(df)),
    )
    df["is_fraud"] = pd.to_numeric(
        _first_available(df, ["is_fraud", "annotation.is_fraud"]),
        errors="coerce",
    )
    df["amount_numeric"] = _extract_amount(
        _first_available(
            df,
            [
                "amount_numeric",
                "annotation.key_features.amount_mentioned",
                "key_features.amount_mentioned",
                "annotation.amount_mentioned",
            ],
            default=pd.Series([pd.NA] * len(df)),
        )
    )
    return df


def _build_known_signatures(base_df: pd.DataFrame) -> set[tuple[str, ...]]:
    known: set[tuple[str, ...]] = set()
    fraud_rows = base_df[base_df["is_fraud"] == 1].copy()
    for _, row in fraud_rows.iterrows():
        signature = []
        for column in FRAUD_SIGNATURE_COLUMNS:
            value = row.get(column, "unknown")
            if pd.isna(value):
                value = "unknown"
            signature.append(str(value).strip().lower() or "unknown")
        known.add(tuple(signature))
    return known


def _class_count(df: pd.DataFrame, label: int) -> int:
    if "annotation.is_fraud" in df.columns:
        series = pd.to_numeric(df["annotation.is_fraud"], errors="coerce")
    elif "is_fraud" in df.columns:
        series = pd.to_numeric(df["is_fraud"], errors="coerce")
    else:
        return 0
    return int((series == label).sum())


def _build_scraped_candidates(base_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    if not SCRAPED_LABELED_CSV_PATH.exists():
        return pd.DataFrame(columns=base_df.columns), []

    labeled_df = _normalize_base_annotations(pd.read_csv(SCRAPED_LABELED_CSV_PATH))
    if labeled_df.empty:
        return pd.DataFrame(columns=base_df.columns), []

    known_post_ids = {
        str(value)
        for value in _first_available(
            base_df,
            ["post_metadata.post_id", "post_id"],
            default=pd.Series(dtype=str),
        ).dropna()
    }
    known_signatures = _build_known_signatures(base_df)

    candidate_rows: list[dict] = []
    report_rows: list[dict] = []

    for _, post in labeled_df.iterrows():
        post_id = str(post.get("post_metadata.post_id", "")).strip()
        if not post_id or post_id in known_post_ids:
            continue

        is_fraud = pd.to_numeric(post.get("annotation.is_fraud"), errors="coerce")
        if pd.isna(is_fraud) or int(is_fraud) != 1:
            continue
        confidence_bucket = str(post.get("annotation.label_quality.confidence_bucket", "") or "").strip().lower()
        usable = post.get("annotation.label_quality.usable_for_training", "")
        if confidence_bucket != "high" or not _is_truthy(usable):
            continue

        fraud_type = str(post.get("annotation.fraud_type", "unknown") or "unknown")
        payment_method = str(post.get("annotation.key_features.payment_method", "unknown") or "unknown")
        fraud_channel = str(post.get("annotation.key_features.fraud_channel", "unknown") or "unknown")
        request_type = str(post.get("annotation.key_features.request_type", "unknown") or "unknown")
        impersonated_entity = str(post.get("annotation.key_features.impersonated_entity", "unknown") or "unknown")

        signature = (
            fraud_type,
            payment_method,
            fraud_channel,
            request_type,
            impersonated_entity,
        )
        normalized_signature = tuple(value.strip().lower() or "unknown" for value in signature)
        is_new_signature = normalized_signature not in known_signatures
        if not is_new_signature:
            continue

        row = {column: post[column] if column in post.index else pd.NA for column in base_df.columns}
        candidate_rows.append(row)
        report_rows.append(
            {
                "post_id": post_id,
                "subreddit": post.get("post_metadata.subreddit", ""),
                "is_fraud": 1,
                "fraud_type": fraud_type,
                "payment_method": payment_method,
                "fraud_channel": fraud_channel,
                "request_type": request_type,
                "impersonated_entity": impersonated_entity,
                "fraud_confidence": post.get("annotation.fraud_confidence", ""),
            }
        )

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        candidate_df = pd.DataFrame(columns=base_df.columns)
    return candidate_df, report_rows


def _write_new_scam_artifacts(new_scam_df: pd.DataFrame, report_rows: list[dict]) -> None:
    ORIGINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if new_scam_df.empty:
        pd.DataFrame(columns=TRAINING_COLUMNS).to_csv(NOVEL_SCAM_SEEDS_PATH, index=False)
        report = {
            "new_scam_row_count": 0,
            "new_scam_signatures": [],
        }
    else:
        available_columns = [column for column in TRAINING_COLUMNS if column in new_scam_df.columns]
        ordered_columns = [
            "title",
            "body",
            "is_fraud",
            "amount_numeric",
            *[column for column in available_columns if column not in {"title", "body", "is_fraud", "amount_numeric"}],
        ]
        seeds = new_scam_df[ordered_columns].copy()
        seeds.to_csv(NOVEL_SCAM_SEEDS_PATH, index=False)
        report = {
            "new_scam_row_count": int(len(new_scam_df)),
            "new_scam_signatures": report_rows,
        }

    with NOVEL_SCAM_REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def sync_scraped_annotations_into_main_csv(
    main_path: Path = MAIN_CSV_PATH,
    scraped_path: Path = SCRAPED_LABELED_CSV_PATH,
    posts_path: Path = SCRAPED_POSTS_PATH,
    min_created_utc_exclusive: float | None = None,
) -> dict[str, int | float]:
    if not main_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {main_path}")

    if not scraped_path.exists():
        return {
            "base_row_count": 0,
            "scraped_row_count": 0,
            "new_rows_added": 0,
            "final_row_count": 0,
            "latest_added_created_utc": 0.0,
        }

    base_df = pd.read_csv(main_path)
    scraped_df = pd.read_csv(scraped_path)
    base_row_count = int(len(base_df))
    scraped_row_count = int(len(scraped_df))

    if scraped_df.empty:
        return {
            "base_row_count": base_row_count,
            "scraped_row_count": scraped_row_count,
            "new_rows_added": 0,
            "final_row_count": base_row_count,
            "latest_added_created_utc": 0.0,
        }

    base_ids = (
        base_df.get("post_metadata.post_id", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    scraped_ids = (
        scraped_df.get("post_metadata.post_id", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
    )

    known_ids = set(base_ids[base_ids != ""])
    new_rows = scraped_df.loc[(scraped_ids != "") & (~scraped_ids.isin(known_ids))].copy()
    latest_added_created_utc = 0.0

    if not new_rows.empty and posts_path.exists():
        posts_df = pd.read_csv(posts_path, usecols=["post_id", "created_utc"])
        posts_df["post_id"] = posts_df["post_id"].fillna("").astype(str).str.strip()
        posts_df["created_utc"] = pd.to_numeric(posts_df["created_utc"], errors="coerce")
        created_lookup = posts_df.dropna(subset=["created_utc"]).drop_duplicates(subset=["post_id"], keep="last")
        created_map = created_lookup.set_index("post_id")["created_utc"]
        new_rows["post_metadata.created_utc"] = new_rows["post_metadata.post_id"].map(created_map)

        if min_created_utc_exclusive is not None:
            new_rows = new_rows.loc[
                pd.to_numeric(new_rows["post_metadata.created_utc"], errors="coerce") > float(min_created_utc_exclusive)
            ].copy()

    if new_rows.empty:
        return {
            "base_row_count": base_row_count,
            "scraped_row_count": scraped_row_count,
            "new_rows_added": 0,
            "final_row_count": base_row_count,
            "latest_added_created_utc": 0.0,
        }

    latest_added_created_utc = float(
        pd.to_numeric(new_rows.get("post_metadata.created_utc"), errors="coerce").dropna().max()
        if "post_metadata.created_utc" in new_rows.columns
        else 0.0
    )

    merged_columns = list(base_df.columns)
    for column in new_rows.columns:
        if column not in merged_columns:
            merged_columns.append(column)

    merged_df = pd.concat(
        [
            base_df.reindex(columns=merged_columns),
            new_rows.reindex(columns=merged_columns),
        ],
        ignore_index=True,
        sort=False,
    )
    merged_df.to_csv(main_path, index=False)
    return {
        "base_row_count": base_row_count,
        "scraped_row_count": scraped_row_count,
        "new_rows_added": int(len(new_rows)),
        "final_row_count": int(len(merged_df)),
        "latest_added_created_utc": latest_added_created_utc,
    }


def sync_recent_fraud_annotations_into_main_csv(
    main_path: Path = MAIN_CSV_PATH,
    scraped_path: Path = SCRAPED_LABELED_CSV_PATH,
    posts_path: Path = SCRAPED_POSTS_PATH,
    min_created_utc_exclusive: float | None = None,
    recent_window_days: int = 7,
    rebalance_majority_threshold: float = 0.75,
) -> dict[str, int | float | bool]:
    if not main_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {main_path}")

    if not scraped_path.exists():
        return {
            "base_row_count": 0,
            "scraped_row_count": 0,
            "new_rows_added": 0,
            "new_fraud_rows_added": 0,
            "final_row_count": 0,
            "latest_added_created_utc": 0.0,
            "base_fraud_count": 0,
            "base_non_fraud_count": 0,
            "final_fraud_count": 0,
            "final_non_fraud_count": 0,
            "requested_non_fraud_balance_rows": 0,
            "balance_changed": False,
        }

    base_df = pd.read_csv(main_path)
    scraped_df = pd.read_csv(scraped_path)
    base_row_count = int(len(base_df))
    scraped_row_count = int(len(scraped_df))
    base_fraud_count = _class_count(base_df, 1)
    base_non_fraud_count = _class_count(base_df, 0)

    if scraped_df.empty:
        return {
            "base_row_count": base_row_count,
            "scraped_row_count": scraped_row_count,
            "new_rows_added": 0,
            "new_fraud_rows_added": 0,
            "final_row_count": base_row_count,
            "latest_added_created_utc": 0.0,
            "base_fraud_count": base_fraud_count,
            "base_non_fraud_count": base_non_fraud_count,
            "final_fraud_count": base_fraud_count,
            "final_non_fraud_count": base_non_fraud_count,
            "requested_non_fraud_balance_rows": 0,
            "balance_changed": False,
        }

    known_keys = _known_annotation_keys(base_df)

    scraped_df = scraped_df.copy()
    scraped_df["post_metadata.post_id"] = (
        scraped_df.get("post_metadata.post_id", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    scraped_df["annotation.is_fraud"] = pd.to_numeric(
        scraped_df.get("annotation.is_fraud"),
        errors="coerce",
    )
    fingerprint_columns = _fingerprint_columns(scraped_df)
    scraped_df["_row_key"] = scraped_df["post_metadata.post_id"]
    missing_row_key = scraped_df["_row_key"] == ""
    if fingerprint_columns:
        scraped_df.loc[missing_row_key, "_row_key"] = scraped_df.loc[missing_row_key].apply(
            lambda row: _stable_row_fingerprint(row, fingerprint_columns),
            axis=1,
        )
        scraped_df["_fingerprint_key"] = scraped_df.apply(
            lambda row: _stable_row_fingerprint(row, fingerprint_columns),
            axis=1,
        )
    else:
        scraped_df["_fingerprint_key"] = ""

    new_rows = scraped_df.loc[
        (
            (scraped_df["_row_key"] != "")
            | (scraped_df["_fingerprint_key"] != "")
        )
        & (~scraped_df["_row_key"].isin(known_keys))
        & (~scraped_df["_fingerprint_key"].isin(known_keys))
        & (scraped_df["annotation.is_fraud"] == 1)
    ].copy()

    latest_added_created_utc = 0.0
    if not new_rows.empty and posts_path.exists():
        posts_df = pd.read_csv(posts_path, usecols=["post_id", "created_utc"])
        posts_df["post_id"] = posts_df["post_id"].fillna("").astype(str).str.strip()
        posts_df["created_utc"] = pd.to_numeric(posts_df["created_utc"], errors="coerce")
        created_lookup = posts_df.dropna(subset=["created_utc"]).drop_duplicates(subset=["post_id"], keep="last")
        created_map = created_lookup.set_index("post_id")["created_utc"]
        new_rows["post_metadata.created_utc"] = new_rows["post_metadata.post_id"].map(created_map)

        recent_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=recent_window_days)
        ).timestamp()
        new_rows = new_rows.loc[
            pd.to_numeric(new_rows["post_metadata.created_utc"], errors="coerce") >= recent_cutoff
        ].copy()

        if min_created_utc_exclusive is not None:
            new_rows = new_rows.loc[
                pd.to_numeric(new_rows["post_metadata.created_utc"], errors="coerce") > float(min_created_utc_exclusive)
            ].copy()

    if new_rows.empty:
        return {
            "base_row_count": base_row_count,
            "scraped_row_count": scraped_row_count,
            "new_rows_added": 0,
            "new_fraud_rows_added": 0,
            "final_row_count": base_row_count,
            "latest_added_created_utc": 0.0,
            "base_fraud_count": base_fraud_count,
            "base_non_fraud_count": base_non_fraud_count,
            "final_fraud_count": base_fraud_count,
            "final_non_fraud_count": base_non_fraud_count,
            "requested_non_fraud_balance_rows": 0,
            "balance_changed": False,
        }

    latest_added_created_utc = float(
        pd.to_numeric(new_rows.get("post_metadata.created_utc"), errors="coerce").dropna().max()
        if "post_metadata.created_utc" in new_rows.columns
        else 0.0
    )

    merged_columns = list(base_df.columns)
    for column in new_rows.columns:
        if column not in merged_columns:
            merged_columns.append(column)

    merged_df = pd.concat(
        [
            base_df.reindex(columns=merged_columns),
            new_rows.reindex(columns=merged_columns),
        ],
        ignore_index=True,
        sort=False,
    )
    merged_df.to_csv(main_path, index=False)

    # Sync to train and test
    if TRAIN_CSV_PATH.exists() and TEST_CSV_PATH.exists() and not new_rows.empty:
        try:
            # Random split for the new fraud rows (80% train, 20% test)
            new_train, new_test = train_test_split(new_rows, test_size=0.2, random_state=42)
            
            # Sync train
            train_df = pd.read_csv(TRAIN_CSV_PATH)
            train_cols = list(train_df.columns)
            merged_train = pd.concat(
                [train_df, new_train.reindex(columns=train_cols)],
                ignore_index=True, sort=False
            )
            merged_train.to_csv(TRAIN_CSV_PATH, index=False)
            
            # Sync test
            test_df = pd.read_csv(TEST_CSV_PATH)
            test_cols = list(test_df.columns)
            merged_test = pd.concat(
                [test_df, new_test.reindex(columns=test_cols)],
                ignore_index=True, sort=False
            )
            merged_test.to_csv(TEST_CSV_PATH, index=False)
        except Exception as e:
            print(f"Warning: Failed to sync train/test portions: {e}")

    new_fraud_rows_added = int(len(new_rows))
    final_fraud_count = base_fraud_count + new_fraud_rows_added
    final_non_fraud_count = base_non_fraud_count
    total_labeled_rows = final_fraud_count + final_non_fraud_count
    fraud_ratio = (final_fraud_count / total_labeled_rows) if total_labeled_rows else 0.0
    non_fraud_ratio = (final_non_fraud_count / total_labeled_rows) if total_labeled_rows else 0.0
    needs_non_fraud_rebalance = (
        total_labeled_rows > 0
        and fraud_ratio >= rebalance_majority_threshold
        and non_fraud_ratio <= (1.0 - rebalance_majority_threshold)
        and final_fraud_count > final_non_fraud_count
    )
    requested_non_fraud_balance_rows = (
        final_fraud_count - final_non_fraud_count
        if needs_non_fraud_rebalance
        else 0
    )

    return {
        "base_row_count": base_row_count,
        "scraped_row_count": scraped_row_count,
        "new_rows_added": new_fraud_rows_added,
        "new_fraud_rows_added": new_fraud_rows_added,
        "final_row_count": int(len(merged_df)),
        "latest_added_created_utc": latest_added_created_utc,
        "base_fraud_count": base_fraud_count,
        "base_non_fraud_count": base_non_fraud_count,
        "final_fraud_count": final_fraud_count,
        "final_non_fraud_count": final_non_fraud_count,
        "final_fraud_ratio": fraud_ratio,
        "final_non_fraud_ratio": non_fraud_ratio,
        "rebalance_majority_threshold": rebalance_majority_threshold,
        "requested_non_fraud_balance_rows": requested_non_fraud_balance_rows,
        "balance_changed": needs_non_fraud_rebalance,
    }


def append_synthetic_non_fraud_to_main_csv(
    synthetic_path: Path,
    main_path: Path = MAIN_CSV_PATH,
    source_name: str = "adv_ctgan_balance",
) -> dict[str, int]:
    if not main_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {main_path}")
    if not synthetic_path.exists():
        return {"synthetic_rows_added": 0, "final_row_count": int(len(pd.read_csv(main_path)))}

    main_df = pd.read_csv(main_path)
    synthetic_df = pd.read_csv(synthetic_path)
    synthetic_df = synthetic_df.copy()

    if "is_fraud" in synthetic_df.columns:
        synthetic_df = synthetic_df.loc[pd.to_numeric(synthetic_df["is_fraud"], errors="coerce") == 0].copy()
    else:
        return {"synthetic_rows_added": 0, "final_row_count": int(len(main_df))}

    if synthetic_df.empty:
        return {"synthetic_rows_added": 0, "final_row_count": int(len(main_df))}

    main_columns = list(main_df.columns)
    mapped_rows: list[dict] = []
    existing_ids = set(
        main_df.get("post_metadata.post_id", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
    )
    synthetic_index = 0

    for _, row in synthetic_df.iterrows():
        while True:
            synthetic_index += 1
            synthetic_id = f"synthetic_{source_name}_{synthetic_index}"
            if synthetic_id not in existing_ids:
                existing_ids.add(synthetic_id)
                break

        mapped = {column: pd.NA for column in main_columns}
        overlap_columns = set(main_columns).intersection(synthetic_df.columns)
        for column in overlap_columns:
            mapped[column] = row[column]

        mapped["post_metadata.post_id"] = synthetic_id
        if "annotation.is_fraud" in mapped:
            mapped["annotation.is_fraud"] = 0
        if "is_fraud" in mapped:
            mapped["is_fraud"] = 0
        if "post_metadata.title" in mapped and pd.isna(mapped["post_metadata.title"]):
            mapped["post_metadata.title"] = ""
        if "post_metadata.body" in mapped and pd.isna(mapped["post_metadata.body"]):
            mapped["post_metadata.body"] = ""
        if "post_metadata.subreddit" in mapped and pd.isna(mapped["post_metadata.subreddit"]):
            mapped["post_metadata.subreddit"] = source_name
        if "annotation.label_quality.confidence_bucket" in mapped:
            mapped["annotation.label_quality.confidence_bucket"] = "synthetic"
        if "annotation.label_quality.usable_for_training" in mapped:
            mapped["annotation.label_quality.usable_for_training"] = True
        if "annotation.reasoning.primary_evidence" in mapped:
            mapped["annotation.reasoning.primary_evidence"] = f"generated:{source_name}"
        if "annotation.gan_quality.suitable_for_gan" in mapped:
            mapped["annotation.gan_quality.suitable_for_gan"] = True

        mapped_rows.append(mapped)

    augmented_main_df = pd.concat([main_df, pd.DataFrame(mapped_rows)], ignore_index=True, sort=False)
    augmented_main_df.to_csv(main_path, index=False)
    return {
        "synthetic_rows_added": len(mapped_rows),
        "final_row_count": int(len(augmented_main_df)),
    }


def prepare_main_dataset(
    input_path: Path = MAIN_CSV_PATH,
    output_path: Path = ORIGINAL_DATASET_PATH,
) -> PreparedDataset:
    base_df = _normalize_base_annotations(pd.read_csv(input_path))
    labeled_scraped_df = (
        _normalize_base_annotations(pd.read_csv(SCRAPED_LABELED_CSV_PATH))
        if SCRAPED_LABELED_CSV_PATH.exists()
        else pd.DataFrame()
    )
    new_scam_df, report_rows = _build_scraped_candidates(base_df)
    frames_to_concat = [base_df]
    if not new_scam_df.empty:
        frames_to_concat.append(new_scam_df.dropna(axis=1, how="all"))
    combined_df = pd.concat(frames_to_concat, ignore_index=True, sort=False)

    available_columns = [column for column in TRAINING_COLUMNS if column in combined_df.columns]
    ordered_columns = [
        "title",
        "body",
        "is_fraud",
        "amount_numeric",
        *[column for column in available_columns if column not in {"title", "body", "is_fraud", "amount_numeric"}],
    ]
    prepared = combined_df[ordered_columns].copy()

    for column in prepared.columns:
        prepared[column] = _to_boolish(prepared[column])

    prepared = prepared.dropna(subset=["is_fraud"])
    prepared["is_fraud"] = prepared["is_fraud"].astype(int)

    ORIGINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    _write_new_scam_artifacts(new_scam_df, report_rows)

    class_counts = {
        str(label): int(count)
        for label, count in prepared["is_fraud"].value_counts(dropna=False).sort_index().items()
    }
    labeled_scraped_class_counts: dict[str, int] = {}
    labeled_scraped_usable_count = 0
    if not labeled_scraped_df.empty and "is_fraud" in labeled_scraped_df.columns:
        labeled_scraped_clean = labeled_scraped_df.dropna(subset=["is_fraud"]).copy()
        labeled_scraped_clean["is_fraud"] = labeled_scraped_clean["is_fraud"].astype(int)
        labeled_scraped_class_counts = {
            str(label): int(count)
            for label, count in labeled_scraped_clean["is_fraud"].value_counts(dropna=False).sort_index().items()
        }
        if "annotation.label_quality.usable_for_training" in labeled_scraped_clean.columns:
            labeled_scraped_usable_count = int(
                labeled_scraped_clean["annotation.label_quality.usable_for_training"].map(_is_truthy).sum()
            )
    return PreparedDataset(
        output_path=output_path,
        row_count=int(len(prepared)),
        class_counts=class_counts,
        columns=list(prepared.columns),
        new_scam_row_count=int(len(new_scam_df)),
        new_scam_seed_path=NOVEL_SCAM_SEEDS_PATH,
        new_scam_report_path=NOVEL_SCAM_REPORT_PATH,
        base_row_count=int(len(base_df)),
        labeled_scraped_row_count=int(len(labeled_scraped_df)),
        labeled_scraped_class_counts=labeled_scraped_class_counts,
        labeled_scraped_usable_count=labeled_scraped_usable_count,
        new_scam_signatures=report_rows,
    )
