# Codex Prompt: Multi-Agent Fraud Detection System with MAPE-K, LangGraph, CTGAN, and Adversarial Training

You are a senior Python engineer, ML systems architect, and agentic AI developer. Build a financial fraud detection platform as a decision-making multi-agent system with adaptive learning, orchestrated using LangGraph.

This is not a single static classifier. It must be a MAPE-K-driven adaptive system with explicit agents, feedback loops, policy-based decisions, drift handling, optional CTGAN augmentation, and adversarial training.

---

## Objective

Implement a working prototype that:

1. Ingests fraud transaction data from CSV
2. Detects feature drift and optionally concept drift
3. Evaluates current model performance
4. Makes policy-driven decisions about retraining or skipping updates
5. Adapts strategy dynamically
6. Handles fraud class imbalance using CTGAN when needed
7. Performs adversarial training using attack-like perturbed fraud samples
8. Retrains and validates a candidate model
9. Runs simulation/robustness checks before promotion
10. Promotes the new model only if it satisfies policy thresholds
11. Stores logs, metrics, decisions, and model metadata in a knowledge layer

---

## Input Data Schema

The system expects a CSV file with the following columns:

| Column             | Type    | Description                                      |
|--------------------|---------|--------------------------------------------------|
| `transaction_id`   | string  | Unique transaction identifier                    |
| `amount`           | float   | Transaction amount in USD                        |
| `merchant_category`| string  | Merchant category code (e.g., "grocery", "travel") |
| `hour_of_day`      | int     | Hour transaction occurred (0–23)                 |
| `day_of_week`      | int     | Day of week (0=Monday, 6=Sunday)                 |
| `distance_from_home`| float  | Distance in km between merchant and home address |
| `velocity_last_1h` | int     | Number of transactions by this card in last 1h   |
| `velocity_last_24h`| int     | Number of transactions by this card in last 24h  |
| `is_foreign`       | int     | 1 if foreign transaction, 0 otherwise            |
| `is_fraud`         | int     | Target label: 1 = fraud, 0 = legitimate          |

**Baseline dataset:** The first batch loaded at system startup serves as the reference/baseline distribution for drift detection. All subsequent batches are compared against this baseline.

**Processing mode:** Batch processing only. Each CSV file represents one time window (e.g., one day of transactions). Simulate streaming by splitting a single large CSV into sequential batches.

---

## Ingestion Agent

The `ingestion_agent` is the first node in the pipeline. It is responsible for loading, validating, and preparing raw data before any downstream agent sees it.

### Supported sources

| Source | Details |
|--------|---------|
| CSV file | Primary mode. Pass path via `--batch-file` CLI arg |
| JSON / Parquet | Accepted; converted internally to DataFrame |
| Database export | SQL query result saved as CSV before ingestion |
| Stream (future) | Kafka integration not in scope; stub only |

### Steps

1. **Load** — Read the file using `pandas.read_csv` (or `read_parquet` / `read_json`). Fix encoding issues silently (use `errors="replace"`).
2. **Schema check** — Verify all required columns are present (see Input Data Schema). Raise `SchemaError` and reject the batch if any column is missing.
3. **Type coercion** — Cast columns to expected dtypes: `amount` → float64, `hour_of_day` / `day_of_week` / `velocity_*` / `is_foreign` / `is_fraud` → int32.
4. **Value range validation** — Enforce:
   - `amount > 0`
   - `hour_of_day` in [0, 23]
   - `day_of_week` in [0, 6]
   - `is_foreign` in {0, 1}
   - `is_fraud` in {0, 1}
   - Drop rows that violate any constraint; log count of dropped rows.
5. **Deduplication** — Drop duplicate `transaction_id` rows, keep first occurrence.
6. **Baseline registration** — If no baseline exists in the knowledge layer yet, store this batch as the baseline and exit the pipeline (no drift to compute on first run).
7. **Split** — Produce a stratified 80/20 train/validation split on `is_fraud`. Store both in `PipelineState`.
8. **Fraud rate check** — Compute `fraud_rate = is_fraud.mean()`. Set `low_fraud_flag = True` if `fraud_rate < 0.05`. Pass flag to `PipelineState` for the policy agent.

### Outputs added to PipelineState

```python
batch_df: pd.DataFrame          # full cleaned batch
train_df: pd.DataFrame          # 80% stratified split
val_df: pd.DataFrame            # 20% stratified split
baseline_df: pd.DataFrame       # reference distribution
fraud_rate: float                # e.g. 0.032
low_fraud_flag: bool             # True if fraud_rate < 0.05
ingestion_meta: dict             # {rows_loaded, rows_dropped, source_file, timestamp}
```

### Error handling

| Condition | Behaviour |
|-----------|-----------|
| Missing required column | Raise `SchemaError`, log, abort pipeline run |
| All rows fail validation | Raise `EmptyBatchError`, log, abort |
| Duplicate transaction IDs | Drop silently, log count |
| File not found | Raise `FileNotFoundError`, log, abort |

### File location

`agents/ingestion_agent.py`

---

## Architecture Constraint (MAPE-K)

- **Monitor** → ingestion + drift detection
- **Analyze** → model evaluation + drift interpretation
- **Plan** → policy agent + strategy agent
- **Execute** → CTGAN + adversarial + training + simulation + deployment
- **Knowledge** → logs, metrics, models, decisions

---

## Mandatory Agents

### `drift_agent`
- Compute PSI for each numeric feature between baseline and current batch
- Compute KS statistic for numeric features
- **Drift is confirmed if:** PSI > 0.2 OR KS p-value < 0.05 for any feature
- Optionally compute concept drift by comparing label distribution shift (use KL divergence on `is_fraud` rate)
- Output: `{feature_name: {psi, ks_stat, ks_pvalue, drifted: bool}}, concept_drift: bool`

### `evaluation_agent`
- Compute on a held-out validation split (20% of current batch, stratified):
  - F1 score (fraud class)
  - Precision (fraud class)
  - Recall (fraud class)
  - ROC-AUC
  - False Positive Rate (FPR = FP / (FP + TN))
- Output: `{f1, precision, recall, roc_auc, fpr, eval_timestamp}`

### `policy_agent`
- Receives: drift results, evaluation metrics, previous model metrics (from knowledge layer)
- Applies the following rules **in order**:

  | Condition | Decision |
  |-----------|----------|
  | F1 < 0.75 AND drift detected | retrain + use_ctgan + use_adversarial |
  | F1 < 0.75 AND no drift | retrain only |
  | F1 >= 0.75 AND drift detected | retrain + use_adversarial |
  | F1 >= 0.85 AND no drift AND simulation passed | skip retrain, deploy as-is |
  | F1 >= 0.75 AND no drift | skip retrain |

- **use_ctgan = True** when: fraud rate in current batch < 5% OR policy rule mandates it
- Output: `{action: "retrain"|"skip", use_ctgan: bool, use_adversarial: bool, reason: str}`

### `strategy_agent`
- Receives policy decision and determines training configuration:
  - `n_estimators`: 200 if use_adversarial else 100
  - `scale_pos_weight`: computed from class ratio in training data
  - `ctgan_samples`: 500 if fraud count < 100, else 200
  - `adversarial_budget`: 0.1 (10% of fraud samples to perturb)
- Output: training config dict

### `augmentation_agent`
- Only runs if `use_ctgan = True`
- Fit CTGAN on **fraud samples only** from the training split
- Generate `ctgan_samples` synthetic fraud rows
- Validate synthetic samples: drop any with `amount < 0` or `velocity_last_1h > 50`
- Output: DataFrame of synthetic fraud samples

### `adversarial_agent`
- Only runs if `use_adversarial = True`
- Takes fraud samples from training data; perturbs `adversarial_budget` fraction of them
- Apply **three attack types**:
  1. **Noise perturbation:** Add Gaussian noise (mean=0, std=0.05 × feature std) to all numeric features
  2. **Boundary attack:** Shift `amount` and `distance_from_home` toward the 25th percentile of legitimate transactions (simulate evasion)
  3. **Evasion mutation:** Set `is_foreign = 0`, `velocity_last_1h = 1` for a random subset (simulate camouflage)
- Do **not** use external adversarial libraries; implement from scratch
- Output: DataFrame of adversarial fraud samples

### `training_agent`
- Training data = `clean_data + ctgan_data (if any) + adversarial_data (if any)`
- Primary model: **XGBoost** (`xgboost.XGBClassifier`)
- Fallback: `sklearn.ensemble.RandomForestClassifier` if XGBoost is not installed
- Train with `scale_pos_weight` from strategy config
- Save model to `./models/candidate_model_{timestamp}.joblib`
- Output: trained model object + training metadata

### `simulation_agent`
- Run 3 robustness checks on the candidate model:
  1. **Holdout eval:** F1 on 20% held-out validation split (already computed by evaluation_agent post-training)
  2. **Adversarial stress test:** Run evaluation_agent on adversarial samples only; require recall ≥ 0.60
  3. **Imbalance stress test:** Simulate 99:1 class ratio by subsampling; require FPR ≤ 0.05
- Simulation **passes** if all 3 checks pass
- Output: `{passed: bool, check_results: {holdout, adversarial_stress, imbalance_stress}}`

### `knowledge_agent`
- Appends one JSONL record per pipeline run to `./logs/pipeline_runs.jsonl`
- Each record must contain:
  ```json
  {
    "run_id": "uuid",
    "timestamp": "ISO8601",
    "batch_file": "path/to/csv",
    "drift_summary": {...},
    "pre_train_metrics": {...},
    "post_train_metrics": {...},
    "policy_decision": {...},
    "simulation_result": {...},
    "model_promoted": true,
    "model_path": "path/or/null",
    "notes": "optional string"
  }
  ```
- Also maintain `./logs/model_registry.jsonl`: one record per promoted model with path, metrics, and promotion timestamp

---

## LangGraph Flow

```
START
→ ingestion_agent
→ drift_agent
→ evaluation_agent            ← (pre-training eval)
→ policy_agent                ← decides retrain/skip + flags
→ [IF action == "skip"] → knowledge_agent → END
→ strategy_agent
→ [IF use_ctgan] → augmentation_agent
→ [IF use_adversarial] → adversarial_agent
→ training_agent
→ evaluation_agent            ← (post-training eval, reused node)
→ simulation_agent
→ policy_agent                ← (promotion decision only)
→ knowledge_agent
→ END
```

**State object** passed between nodes (TypedDict):
```python
class PipelineState(TypedDict):
    batch_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    baseline_df: pd.DataFrame
    fraud_rate: float
    low_fraud_flag: bool
    ingestion_meta: dict
    drift_results: dict
    pre_metrics: dict
    post_metrics: dict
    policy_decision: dict
    strategy_config: dict
    ctgan_samples: pd.DataFrame | None
    adversarial_samples: pd.DataFrame | None
    candidate_model: Any
    simulation_result: dict
    promoted: bool
```

---

## Promotion Logic

Promote the candidate model to `./models/active_model.joblib` **only if ALL conditions are met**:

| Condition | Threshold |
|-----------|-----------|
| F1 (post-training) | ≥ 0.78 |
| Precision (post-training) | ≥ 0.70 |
| FPR (post-training) | ≤ 0.05 |
| Simulation | passed = True |
| vs. previous model | F1_new > F1_previous (strictly greater) |

**"Better than previous model"** is defined as: new model F1 > previous model F1. If no previous model exists, promote automatically if the first four conditions pass.

If promotion is rejected, log the reason and keep the existing active model unchanged.

---

## Tech Stack

- Python 3.10+
- `pandas`, `numpy`, `scipy`
- `scikit-learn`
- `langgraph`
- `ctgan`
- `xgboost` (primary), `scikit-learn RandomForest` (fallback)
- `joblib` (model persistence)
- Optional: `mlflow`, `shap`

---

## Project Structure

```
fraud_detection/
├── agents/
│   ├── ingestion_agent.py
│   ├── drift_agent.py
│   ├── evaluation_agent.py
│   ├── policy_agent.py
│   ├── strategy_agent.py
│   ├── augmentation_agent.py
│   ├── adversarial_agent.py
│   ├── training_agent.py
│   ├── simulation_agent.py
│   └── knowledge_agent.py
├── graph/
│   └── pipeline.py          ← LangGraph graph definition
├── state/
│   └── schema.py            ← PipelineState TypedDict
├── models/                  ← Saved models (gitignored)
├── logs/                    ← JSONL logs (gitignored)
├── data/
│   └── sample_transactions.csv
├── main.py                  ← Entry point
├── requirements.txt
└── README.md
```

---

## Deliverables

1. Full working code, modularized as above
2. LangGraph orchestration in `graph/pipeline.py`
3. README with setup instructions and example run command
4. Sample CSV with at least 1000 rows (≥5% fraud rate) for testing
5. Example JSONL log output showing one complete pipeline run
6. Extension notes: how to add river (online learning) or SHAP explainability

---

## One-Line Goal

Build a drift-aware, adversarially trained, policy-governed, multi-agent fraud detection system with adaptive learning using LangGraph and MAPE-K, with fully specified thresholds, schemas, and agent contracts.
