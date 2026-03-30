from __future__ import annotations

import os
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AGENT_ROOT.parent
DATA_ROOT = REPO_ROOT / "9k-dataset"

DATASET_PATH = DATA_ROOT / "data.csv"
TRAIN_PATH = DATA_ROOT / "Data labeling" / "outputs" / "train_3k.csv"
TEST_PATH = DATA_ROOT / "Data labeling" / "outputs" / "test_3k.csv"
BOOTSTRAP_TRAIN_PATH = DATA_ROOT / "1k data" / "xgb_ctgan_1k" / "augmented" / "train_augmented.csv"
BOOTSTRAP_TEST_PATH = DATA_ROOT / "1k data" / "xgb_ctgan_1k" / "augmented" / "test_predictions.csv"

SCRAPER_WORKDIR = DATA_ROOT / "Data Scrapping"
SCRAPER_SCRIPT_PATH = SCRAPER_WORKDIR / "Scrapper.py"
LABEL_WORKDIR = DATA_ROOT / "Data labeling"
LABEL_SCRIPT_PATH = LABEL_WORKDIR / "label.py"
SCRAPED_LABELED_CSV_PATH = DATA_ROOT / "Data labeling" / "outputs" / "scraped_annotations_latest.csv"
SCRAPED_LABELED_JSON_PATH = DATA_ROOT / "Data labeling" / "outputs" / "scraped_annotations_latest.json"
INCREMENTAL_LABEL_DIR = DATA_ROOT / "Data labeling" / "outputs" / "incremental"
INCREMENTAL_POSTS_TO_LABEL_PATH = INCREMENTAL_LABEL_DIR / "posts_to_label.csv"
INCREMENTAL_LABELED_CSV_PATH = INCREMENTAL_LABEL_DIR / "new_annotations.csv"
INCREMENTAL_LABELED_JSON_PATH = INCREMENTAL_LABEL_DIR / "new_annotations.json"
PREPROCESS_SCRIPT_PATH = DATA_ROOT / "scripts" / "data_prep" / "preprocess_dataset.py"
PREPROCESSED_SCRAPED_CSV_PATH = DATA_ROOT / "Data labeling" / "outputs" / "scraped_annotations_preprocessed.csv"
PREPROCESSED_SCRAPED_SUMMARY_PATH = DATA_ROOT / "Data labeling" / "outputs" / "scraped_annotations_preprocessed_summary.json"
PREPROCESSED_INCREMENTAL_CSV_PATH = INCREMENTAL_LABEL_DIR / "new_annotations_preprocessed.csv"
PREPROCESSED_INCREMENTAL_SUMMARY_PATH = INCREMENTAL_LABEL_DIR / "new_annotations_preprocessed_summary.json"

FIN_FRAUD_ROOT = DATA_ROOT
PREPARED_DATASET_PATH = DATASET_PATH
CTGAN_SCRIPT_PATH = DATA_ROOT / "scripts" / "experiments" / "3k_data" / "run_xgb_ctgan_3k.py"
ADV_CTGAN_SCRIPT_PATH = DATA_ROOT / "adversial" / "adversarial_training.py"
EVAL_SCRIPT_PATH = DATA_ROOT / "scripts" / "experiments" / "3k_data" / "run_xgb_ctgan_3k.py"
TRAINING_SCRIPT_PATH = DATA_ROOT / "scripts" / "experiments" / "3k_data" / "run_xgb_ctgan_3k.py"
ROBUSTNESS_SCRIPT_PATH = DATA_ROOT / "adversial" / "robustness_curve.py"
CLASSIFIER_RESULTS_PATH = DATA_ROOT / "artifacts" / "xgb_ctgan_3k" / "summary_metrics_3k.json"
ROBUSTNESS_RESULTS_PATH = DATA_ROOT / "adversial" / "results" / "summary_table.csv"

ADVERSARIAL_ROOT = DATA_ROOT / "adversial"
ADVERSARIAL_TRAINING_SCRIPT_PATH = ADVERSARIAL_ROOT / "adversarial_training.py"
ROBUSTNESS_CURVE_SCRIPT_PATH = ADVERSARIAL_ROOT / "robustness_curve.py"
ADVERSARIAL_RESULTS_DIR = ADVERSARIAL_ROOT / "results"
ADVERSARIAL_SUMMARY_PATH = ADVERSARIAL_RESULTS_DIR / "summary_table.csv"
ADVERSARIAL_ROBUSTNESS_CURVE_PATH = ADVERSARIAL_RESULTS_DIR / "robustness_curve.png"

MEMORY_DIR = AGENT_ROOT / "memory"
MEMORY_PATH = MEMORY_DIR / "agent_memory.json"
OUTPUT_DIR = AGENT_ROOT / "output"
RUN_REPORT_PATH = OUTPUT_DIR / "run_report.json"
EXECUTION_TRACE_PATH = OUTPUT_DIR / "execution_trace.jsonl"
INGESTED_POST_IDS_PATH = OUTPUT_DIR / "ingested_post_ids.json"
AUDIT_LOG_PATH = REPO_ROOT / "agent" / "audit_log.jsonl"

OLLAMA_MODEL = "qwen3.5:4b"
OLLAMA_URL = "http://127.0.0.1:11434"

SMOKE_TEST = os.getenv("AGENT_SMOKE_TEST", "0").strip().lower() in {"1", "true", "yes", "on"}
STOP_AFTER_INGESTION_IF_NO_UPDATE = os.getenv("AGENT_STOP_AFTER_INGESTION_IF_NO_UPDATE", "0").strip().lower() in {"1", "true", "yes", "on"}
SMOKE_TEST_SUBREDDITS = ("Scams",)
SMOKE_TEST_MAX_RESULTS = 2
SMOKE_TEST_TOP_N_COMMENTS = 5
SMOKE_TEST_SLEEP_TIME_SEC = 0.2

MAX_REVIEW_LOOPS = 6

LABEL_NOISE_THRESHOLD = 0.18
MIN_JS_DIVERGENCE_ACCEPT = 0.20
DEFAULT_RATIO_CANDIDATES = (5, 10, 15)
TARGET_F1_THRESHOLD = 0.82
TARGET_NON_FRAUD_F1_THRESHOLD = 0.50  # Minimum acceptable F1 for the minority (non-fraud) class
TARGET_ROBUSTNESS_THRESHOLD = 0.70
MIN_NEW_FRAUD_ROWS_TO_UPDATE = 2
SCRAPER_DEFAULT_LOOKBACK_DAYS = 90
SCRAPER_LABELED_RUN_LOOKBACK_DAYS = 7
LABELED_RUN_BATCH_SIZE = 10
SCRAPER_LABELED_RUN_MAX_RESULTS = LABELED_RUN_BATCH_SIZE
LABELER_LABELED_RUN_MAX_POSTS = LABELED_RUN_BATCH_SIZE
