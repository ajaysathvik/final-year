# Agentic Fraud Workflow

This folder contains a LangGraph-based fraud-detection workflow that replaces a linear pipeline with four specialized agents:

- `IngestionAgent`: scrapes Reddit, labels posts, post-processes outputs, appends new rows into train/test, then checks data quality.
- `BalanceAgent`: searches balancing ratios, runs CTGAN generation, and rejects synthetic batches when JSD is too high.
- `StrategyAgent`: runs adversarial training and robustness-curve generation on the balanced model candidate.
- `EvaluationAgent`: validates F1 and robustness thresholds, then either deploys or issues a correction order.

## Architecture

The graph loops when quality gates fail:

`IngestionAgent -> BalanceAgent -> StrategyAgent -> EvaluationAgent`

Correction paths:

- Low synthetic fidelity: `EvaluationAgent -> BalanceAgent`
- Weak robustness: `EvaluationAgent -> StrategyAgent`
- Pass thresholds: `EvaluationAgent -> Complete`

## Memory

Persistent memory is written to:

- `agent-new/memory/agent_memory.json`
- `agent-new/output/run_report.json`

Each run records:

- dataset snapshots
- agent decisions
- artifact pointers
- final workflow report

## Tools

The workflow includes tool wrappers in `tools.py`:

- dataset profiling
- label review
- balancing ratio search
- CTGAN execution
- synthetic quality scoring with Jensen-Shannon divergence
- adversarial training execution
- evaluation and robustness checks
- attack-surface detection for long-form fraud

## Ollama / Qwen

The decision layer uses `ChatOllama` with `qwen2.5:7b-instruct` by default. If Ollama or LangChain bindings are unavailable, the workflow falls back to deterministic rule-based decisions so the graph still runs.

You can change the model in [`config.py`](/home/norm/Projects/Reddit-dataset/agent-new/config.py).

## Run

Install dependencies:

```bash
pip install -r agent-new/requirements.txt
```

Run the workflow:

```bash
python agent-new/main.py
```

## Notes

- The workflow reuses `Fin-Fraud_AI` for CTGAN and classifier evaluation, and runs the adversarial strategy stage from `9k-dataset/adversial`.
- Thresholds and loop limits are configured in [`config.py`](/home/norm/Projects/Reddit-dataset/agent-new/config.py).
