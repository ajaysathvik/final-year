from __future__ import annotations

from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AGENT_ROOT.parent
DATA_ROOT = REPO_ROOT / "9k-dataset"

DATASET_PATH = DATA_ROOT / "processed" / "data_preprocessed.csv"
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
PREPROCESS_SCRIPT_PATH = DATA_ROOT / "scripts" / "data_prep" / "preprocess_dataset.py"
PREPROCESSED_SCRAPED_CSV_PATH = DATA_ROOT / "Data labeling" / "outputs" / "scraped_annotations_preprocessed.csv"
PREPROCESSED_SCRAPED_SUMMARY_PATH = DATA_ROOT / "Data labeling" / "outputs" / "scraped_annotations_preprocessed_summary.json"

FIN_FRAUD_ROOT = DATA_ROOT
PREPARED_DATASET_PATH = DATA_ROOT / "processed" / "data_preprocessed.csv"
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

OLLAMA_MODEL = "qwen3.5:0.8b"
OLLAMA_URL = "http://127.0.0.1:11434"

MAX_REVIEW_LOOPS = 6

LABEL_NOISE_THRESHOLD = 0.18
SLANG_DRIFT_THRESHOLD = 0.12
MIN_JS_DIVERGENCE_ACCEPT = 0.20
DEFAULT_RATIO_CANDIDATES = (10, 18, 20)
TARGET_F1_THRESHOLD = 0.82
TARGET_ROBUSTNESS_THRESHOLD = 0.70
MIN_NEW_FRAUD_ROWS_TO_UPDATE = 10
