"""
Configuration: paths, thresholds, and constants for the fraud detection pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT  = AGENT_ROOT.parent
DATA_ROOT  = REPO_ROOT / "9k-dataset"

DATASET_PATH = DATA_ROOT / "data.csv"

OUTPUT_DIR   = AGENT_ROOT / "output"
MODELS_DIR   = OUTPUT_DIR / "models"
LOGS_DIR     = OUTPUT_DIR / "logs"
METRICS_DIR  = OUTPUT_DIR / "metrics"
KNOWLEDGE_LOG_PATH = LOGS_DIR / "knowledge.jsonl"

# ── Model promotion thresholds ─────────────────────────────────────
F1_THRESHOLD        = float(os.getenv("F1_THRESHOLD", "0.70"))
PRECISION_THRESHOLD = float(os.getenv("PRECISION_THRESHOLD", "0.65"))
FPR_THRESHOLD       = float(os.getenv("FPR_THRESHOLD", "0.15"))
ROBUSTNESS_THRESHOLD = float(os.getenv("ROBUSTNESS_THRESHOLD", "0.60"))

# ── Drift detection ───────────────────────────────────────────────
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.20"))
KS_THRESHOLD  = float(os.getenv("KS_THRESHOLD", "0.05"))

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

# ── Feature columns (numeric ones suitable for ML) ─────────────────
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

TARGET_COL = "annotation.is_fraud"

# ── Ensure output directories exist ───────────────────────────────
for _d in (OUTPUT_DIR, MODELS_DIR, LOGS_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
