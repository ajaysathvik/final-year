# Fraud Pipeline Agent

This agent implements the requested flow:

1. pull recent fraud/scam attack signals into memory
2. normalize `Data labeling/outputs/main.csv`
3. write `Fin-Fraud_AI/original_dataset/final1.csv`
4. ask Ollama `qwen3.5:0.8b` whether new attack patterns require classifier and feature updates
5. run standard CTGAN
6. run adversarial CTGAN
7. run classifier evaluation and save models
8. run adversarial robustness evaluation

Run it from the repository root:

```bash
python agent/main.py
```

If external fraud-intel feeds are unavailable, the pipeline still runs and the failure is recorded in `agent/memory/agent_memory.json`.

If Ollama or the `ollama` Python package is unavailable, the agent records that failure in memory and continues with the pipeline.

The final run summary is also saved to:

```text
agent/output/pipeline_run_summary.json
```

To skip the fraud-intel refresh:

```bash
python agent/main.py --skip-intel
```
