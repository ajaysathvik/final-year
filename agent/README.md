# Multi-Agent Fraud Detection System

A **drift-aware, adversarially trained, policy-governed, multi-agent fraud detection system** with adaptive learning using **LangGraph** and **MAPE-K** architecture.

## Architecture (MAPE-K)

| Phase       | Agents                                      |
|-------------|---------------------------------------------|
| **Monitor** | `drift_agent` — PSI/KS feature drift        |
| **Analyze** | `evaluation_agent` — F1, precision, recall, ROC-AUC, FPR |
| **Plan**    | `policy_agent` → `strategy_agent`            |
| **Execute** | `augmentation_agent` (CTGAN) → `training_agent` (XGBoost/RF + integrated adversarial augmentation) → `simulation_agent` |
| **Knowledge** | `knowledge_agent` — JSONL logs             |

## LangGraph Flow

```
START → drift_agent → evaluation_agent → policy_agent → strategy_agent
  → [if CTGAN] augmentation_agent
  → training_agent → evaluation_agent → simulation_agent
  → policy_agent (promote/reject) → knowledge_agent → END
```

## Quick Start

### 1. Install dependencies

```bash
cd agent
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
# From the agent/ directory
cd agent
python main.py

# Or specify a custom CSV path
python main.py --data /path/to/your/data.csv
```

### 3. Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `F1_THRESHOLD` | `0.70` | Minimum F1 for model promotion |
| `PRECISION_THRESHOLD` | `0.65` | Minimum precision |
| `FPR_THRESHOLD` | `0.15` | Maximum false positive rate |
| `ROBUSTNESS_THRESHOLD` | `0.60` | Minimum robustness score |
| `PSI_THRESHOLD` | `0.20` | PSI drift threshold |
| `KS_THRESHOLD` | `0.05` | KS test p-value threshold |
| `CTGAN_EPOCHS` | `100` | CTGAN training epochs |
| `N_ESTIMATORS` | `200` | Number of trees |

## Output

After a run, check the `agent/output/` directory:

```
output/
├── models/
│   ├── candidate_xgboost_20260331_121500.joblib
│   └── promoted_model.joblib          # if promoted
├── metrics/
│   └── training_metrics_20260331_121500.json
├── logs/
│   └── knowledge.jsonl                # full JSONL decision log
└── run_summary.json                   # final summary
```

## Project Structure

```
agent/
├── main.py                # CLI entry point
├── graph.py               # LangGraph orchestration (MAPE-K flow)
├── config.py              # Paths, thresholds, feature columns
├── state.py               # TypedDict shared pipeline state
├── requirements.txt       # Python dependencies
├── README.md
└── agents/
    ├── drift_agent.py          # Monitor: PSI + KS drift detection
    ├── evaluation_agent.py     # Analyze: classification metrics
    ├── policy_agent.py         # Plan: retrain/promote decisions
    ├── strategy_agent.py       # Plan: adaptation plan builder
    ├── augmentation_agent.py   # Execute: CTGAN synthetic data
    ├── training_agent.py       # Execute: XGBoost/RandomForest + adversarial augmentation
    ├── simulation_agent.py     # Execute: robustness stress testing
    └── knowledge_agent.py      # Knowledge: JSONL logging + model promotion
```

## Adversarial Training

The training data combines three sources:

```
training_data = clean_data + ctgan_data + adversarial_data
```

When feature drift is detected and labeled scraped rows are available, the
pipeline now appends those scraped rows to the training set before balancing
and retraining:

```
training_data = clean_data + drifted_scraped_data + ctgan_data + adversarial_data
```

Set `USE_SCRAPED_DRIFT_DATA=false` to force the pipeline to train from
`data.csv` only, without appending rows from `agent/output/scraped_data`.

The training agent generates adversarial samples using:
1. **Noise perturbation** — Gaussian noise on all features
2. **Boundary attack** — interpolation between fraud and nearest non-fraud
3. **Evasion mutation** — selective feature mutation (~33% of features)

## Promotion Logic

A candidate model is promoted **only if ALL** conditions are met:
- `F1 ≥ F1_THRESHOLD`
- `Precision ≥ PRECISION_THRESHOLD`
- `FPR ≤ FPR_THRESHOLD`
- Simulation passes (robustness ≥ threshold)
- Better F1 than the previous model

## Extension Notes

- Replace rule-based `policy_agent`/`strategy_agent` with LLM-based reasoning (e.g., via `langchain_ollama`)
- Add concept drift detection (e.g., using `river`)
- Add SHAP explanations for transparency
- Add MLflow for model registry
- Add real-time streaming ingestion
