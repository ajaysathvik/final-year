"""
Configuration: paths, thresholds, and constants for the fraud detection pipeline.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT  = AGENT_ROOT.parent
DATA_ROOT  = REPO_ROOT / "9k-dataset"

DATASET_PATH = DATA_ROOT / "data.csv"
TRAIN_BALANCED_SCHEMA_PATH = (
    DATA_ROOT / "artifacts" / "xgb_balanced_train_imbalanced_test_3k" / "train_balanced.csv"
)

OUTPUT_DIR   = AGENT_ROOT / "output"
MODELS_DIR   = OUTPUT_DIR / "models"
LOGS_DIR     = OUTPUT_DIR / "logs"
METRICS_DIR  = OUTPUT_DIR / "metrics"
RUNS_DIR     = OUTPUT_DIR / "runs"
PLOTS_DIR    = OUTPUT_DIR / "plots"
KNOWLEDGE_LOG_PATH = LOGS_DIR / "knowledge.jsonl"

# ── Model promotion thresholds ─────────────────────────────────────
F1_THRESHOLD        = float(os.getenv("F1_THRESHOLD", "0.70"))
PRECISION_THRESHOLD = float(os.getenv("PRECISION_THRESHOLD", "0.65"))
FPR_THRESHOLD       = float(os.getenv("FPR_THRESHOLD", "0.15"))
ROBUSTNESS_THRESHOLD = float(os.getenv("ROBUSTNESS_THRESHOLD", "0.60"))

# ── Drift detection ───────────────────────────────────────────────
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.20"))
KS_THRESHOLD  = float(os.getenv("KS_THRESHOLD", "0.05"))
USE_SCRAPED_DRIFT_DATA = os.getenv("USE_SCRAPED_DRIFT_DATA", "true").strip().lower() in {
    "1", "true", "yes", "on"
}

# ── Random drift data generation ──────────────────────────────────
# When GENERATE_RANDOM_DRIFT_DATA=true, the pipeline regenerates
# drift_test_dataset.csv before each run using procedural scenarios so
# drift patterns and class balance shift continuously across runs.
GENERATE_RANDOM_DRIFT_DATA = os.getenv("GENERATE_RANDOM_DRIFT_DATA", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
# How many rows to generate (default 300 is a good balance of diversity/speed)
RANDOM_DRIFT_N_SAMPLES = int(os.getenv("RANDOM_DRIFT_N_SAMPLES", "300"))
# Optional fixed seed (leave unset to get a new distribution every run)
_rds = os.getenv("RANDOM_DRIFT_SEED", "")
RANDOM_DRIFT_SEED: int | None = int(_rds) if _rds.strip() else None

# ── CTGAN ──────────────────────────────────────────────────────────
CTGAN_EPOCHS       = int(os.getenv("CTGAN_EPOCHS", "100"))
CTGAN_SAMPLE_RATIO = float(os.getenv("CTGAN_SAMPLE_RATIO", "1.0"))

# ── Adversarial ────────────────────────────────────────────────────
ADVERSARIAL_NOISE_STD   = float(os.getenv("ADV_NOISE_STD", "0.05"))
ADVERSARIAL_BOUNDARY_K  = int(os.getenv("ADV_BOUNDARY_K", "5"))
ADVERSARIAL_EVASION_STD = float(os.getenv("ADV_EVASION_STD", "0.03"))

# ── Training ───────────────────────────────────────────────────────
TEST_SIZE        = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE     = int(os.getenv("RANDOM_STATE", "42"))
N_ESTIMATORS     = int(os.getenv("N_ESTIMATORS", "200"))

# ── Supervisor ─────────────────────────────────────────────────────
SUPERVISOR_HEALTH_MIN = float(os.getenv("SUPERVISOR_HEALTH_MIN", "0.50"))

# ── Feedback loop limits ───────────────────────────────────────────
MAX_L1_ITERATIONS = int(os.getenv("MAX_L1_ITERATIONS", "2"))  # Eval → Balance
MAX_L2_ITERATIONS = int(os.getenv("MAX_L2_ITERATIONS", "2"))  # Eval → Strategy
MAX_L3_ITERATIONS = int(os.getenv("MAX_L3_ITERATIONS", "2"))  # Training self-loop
MAX_L4_ITERATIONS = int(os.getenv("MAX_L4_ITERATIONS", "2"))  # Supervisor ↔ Policy
MAX_L5_ITERATIONS = int(os.getenv("MAX_L5_ITERATIONS", "1"))  # Eval → KB validation
MAX_KB_LOOPS     = int(os.getenv("MAX_KB_LOOPS", "3"))        # KB → Drift closed loop re-cycles

# ── Feature columns ────────────────────────────────────────────────
# Full training schema derived from train_balanced.csv. The runtime
# pipeline uses every feature column from that file except the target,
# then one-hot encodes non-numeric values during ingestion so the model
# still receives an all-numeric matrix.
TARGET_COL = "annotation.is_fraud"


def _load_schema_feature_cols() -> list[str]:
    if not TRAIN_BALANCED_SCHEMA_PATH.exists():
        return []

    with TRAIN_BALANCED_SCHEMA_PATH.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f), [])

    return [col for col in header if col and col != TARGET_COL]


TRAIN_BALANCED_FEATURE_COLS = _load_schema_feature_cols()

# Legacy numeric-only feature set retained for synthetic drift generation.
NUMERIC_FEATURE_COLS = [
    # Psychological content signals (style of post, not fraud label)
    "annotation.psychological_tactics.urgency",
    "annotation.psychological_tactics.fear",
    "annotation.psychological_tactics.authority",
    "annotation.psychological_tactics.reward",
    # Community-observed signals
    "annotation.community_signals.num_comments",
    "annotation.community_signals.scam_confirmations",
    "annotation.community_signals.not_scam_claims",
    "annotation.community_signals.advice_requests",
    # Content feature
    "annotation.key_features.has_amount",
    # NOTE: annotation.fraud_confidence and annotation.fraud_labels.*
    # are EXCLUDED — they directly encode the target label and cause
    # data leakage (F1=1.0 on every run).
]

# ── Ensure output directories exist ───────────────────────────────
for _d in (OUTPUT_DIR, MODELS_DIR, LOGS_DIR, METRICS_DIR, RUNS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
