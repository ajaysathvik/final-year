from __future__ import annotations

from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AGENT_ROOT.parent

MAIN_CSV_PATH = REPO_ROOT / "Data labeling" / "outputs" / "main.csv"
TRAIN_CSV_PATH = REPO_ROOT / "Data labeling" / "outputs" / "train_3k.csv"
TEST_CSV_PATH = REPO_ROOT / "Data labeling" / "outputs" / "test_3k.csv"
SCRAPED_POSTS_PATH = REPO_ROOT / "Reddit-dataset" / "Data Scrapping" / "posts.csv"
SCRAPED_COMMENTS_PATH = REPO_ROOT / "Reddit-dataset" / "Data Scrapping" / "comments.csv"
SCRAPER_SCRIPT_PATH = REPO_ROOT / "Reddit-dataset" / "Data Scrapping" / "Scrapper.py"
SCRAPER_WORKDIR = REPO_ROOT / "Reddit-dataset" / "Data Scrapping"
LABEL_SCRIPT_PATH = REPO_ROOT / "Reddit-dataset" / "Data labeling" / "label.py"
LABEL_WORKDIR = REPO_ROOT / "Reddit-dataset" / "Data labeling"
SCRAPED_LABELED_CSV_PATH = REPO_ROOT / "Data labeling" / "outputs" / "scraped_annotations_latest.csv"
SCRAPED_LABELED_JSON_PATH = REPO_ROOT / "Data labeling" / "outputs" / "scraped_annotations_latest.json"
PREPROCESS_DATASET_SCRIPT = REPO_ROOT / "9k-dataset" / "scripts" / "data_prep" / "preprocess_dataset.py"
PREPROCESSED_SCRAPED_LABELED_CSV_PATH = REPO_ROOT / "Data labeling" / "outputs" / "scraped_annotations_preprocessed.csv"
PREPROCESSED_SCRAPED_SUMMARY_PATH = REPO_ROOT / "Data labeling" / "outputs" / "scraped_annotations_preprocessed_summary.json"

FIN_FRAUD_ROOT = REPO_ROOT / "Fin-Fraud_AI"
ORIGINAL_DATASET_DIR = FIN_FRAUD_ROOT / "original_dataset"
ORIGINAL_DATASET_PATH = ORIGINAL_DATASET_DIR / "final1.csv"
NOVEL_SCAM_SEEDS_PATH = ORIGINAL_DATASET_DIR / "new_scam_seeds.csv"
NOVEL_SCAM_REPORT_PATH = ORIGINAL_DATASET_DIR / "new_scam_report.json"

STANDARD_CTGAN_SCRIPT = FIN_FRAUD_ROOT / "CTGAN" / "run_standard_ctgan.py"
ADV_CTGAN_SCRIPT = FIN_FRAUD_ROOT / "adversarial_training" / "adv_ctgan_train.py"
CLASSIFIER_EVAL_SCRIPT = FIN_FRAUD_ROOT / "classifier_models" / "comprehensive_eval.py"
ROBUSTNESS_EVAL_SCRIPT = FIN_FRAUD_ROOT / "adversarial_training" / "robustness_evaluation.py"

CTGAN_OUTPUT_PATH = FIN_FRAUD_ROOT / "CTGAN" / "ctgan_balanced_data.csv"
CTGAN_SYNTHETIC_NOVEL_OUTPUT_PATH = FIN_FRAUD_ROOT / "CTGAN" / "synthetic_new_scams_ctgan.csv"
ADV_CTGAN_OUTPUT_PATH = FIN_FRAUD_ROOT / "adversarial_training" / "adv_balanced_data.csv"
ADV_CTGAN_SYNTHETIC_NOVEL_OUTPUT_PATH = FIN_FRAUD_ROOT / "adversarial_training" / "synthetic_new_scams_adv_ctgan.csv"
ADV_CTGAN_SYNTHETIC_BATCH_PATH = FIN_FRAUD_ROOT / "adversarial_training" / "synthetic_adv_batch.csv"
CLASSIFIER_OUTPUT_PATH = FIN_FRAUD_ROOT / "outputs" / "classifier_comparison.csv"
ROBUSTNESS_OUTPUT_PATH = FIN_FRAUD_ROOT / "outputs" / "robustness_results.csv"
MODELS_DIR = FIN_FRAUD_ROOT / "models"

MEMORY_DIR = AGENT_ROOT / "memory"
MEMORY_PATH = MEMORY_DIR / "agent_memory.json"
OUTPUT_DIR = AGENT_ROOT / "output"
RUN_SUMMARY_PATH = OUTPUT_DIR / "pipeline_run_summary.json"
TOOL_CALL_OUTPUTS_PATH = OUTPUT_DIR / "tool_call_outputs.json"
OUTPUT_MODELS_DIR = OUTPUT_DIR / "models"
OUTPUT_MEMORY_PATH = OUTPUT_DIR / "agent_memory.json"
OUTPUT_PREPARED_DATASET_PATH = OUTPUT_DIR / "final1.csv"
OUTPUT_CTGAN_PATH = OUTPUT_DIR / "ctgan_balanced_data.csv"
OUTPUT_CTGAN_SYNTHETIC_NOVEL_PATH = OUTPUT_DIR / "synthetic_new_scams_ctgan.csv"
OUTPUT_ADV_CTGAN_PATH = OUTPUT_DIR / "adv_balanced_data.csv"
OUTPUT_ADV_CTGAN_SYNTHETIC_NOVEL_PATH = OUTPUT_DIR / "synthetic_new_scams_adv_ctgan.csv"
OUTPUT_CLASSIFIER_RESULTS_PATH = OUTPUT_DIR / "classifier_comparison.csv"
OUTPUT_ROBUSTNESS_RESULTS_PATH = OUTPUT_DIR / "robustness_results.csv"
