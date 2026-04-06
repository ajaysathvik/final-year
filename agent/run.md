# Demo Runs for the Agent

Use these from the `agent/` directory.

## 1. Normal Run

This shows the baseline path with no external drift data appended.

```bash
cd agent
USE_SCRAPED_DRIFT_DATA=false python main.py
```

What this demonstrates:
- Normal pipeline execution on the base dataset
- Drift check without using scraped drift rows
- Standard training, evaluation, simulation, and promotion flow

Expected pattern:
- `drift_detected: false`
- Retraining may still happen if there is no promoted model yet

## 2. Drift Run with Scraped Drift Data

This is the best run to show how the agent reacts when drifted data is available.

```bash
cd agent
USE_SCRAPED_DRIFT_DATA=true python main.py
```

What this demonstrates:
- Drift detection against scraped or prepared drift data
- Drift remediation by appending drift rows into training
- Retraining and re-evaluation after drift is detected

Expected pattern:
- `drift_detected: true`
- `used_drift_remediation: true`
- `drift_rows_added` greater than `0`

## 3. Drift Run with Fresh Random Drift

This is useful when you want a second drift scenario that looks different each time.

```bash
cd agent
GENERATE_RANDOM_DRIFT_DATA=true USE_SCRAPED_DRIFT_DATA=true python main.py
```

What this demonstrates:
- A newly generated drift distribution before the run
- The agent handling a less predictable drift case
- A stronger demo of adaptation than reusing the same saved drift file

Useful optional variants:

```bash
RANDOM_DRIFT_N_SAMPLES=500 GENERATE_RANDOM_DRIFT_DATA=true USE_SCRAPED_DRIFT_DATA=true python main.py
RANDOM_DRIFT_SEED=42 GENERATE_RANDOM_DRIFT_DATA=true USE_SCRAPED_DRIFT_DATA=true python main.py
```

Use the first variant for a larger drift batch.
Use the second variant when you want a repeatable demo.

## Suggested Demo Order

```bash
USE_SCRAPED_DRIFT_DATA=false python main.py
USE_SCRAPED_DRIFT_DATA=true python main.py
GENERATE_RANDOM_DRIFT_DATA=true USE_SCRAPED_DRIFT_DATA=true python main.py
```

This gives you:
- One normal run
- One drift run using prepared drift data
- One drift run using fresh synthetic drift

## Suggestions

- Use the normal run first so people can see the baseline behavior before adaptation starts.
- Use the scraped-drift run second because it is the clearest example of drift detection plus remediation.
- Use the random-drift run third if you want to show the agent is not tied to one fixed drift file.
- After each run, open `agent/output/run_summary.json` and the latest folder in `agent/output/runs/` to show metrics and plots.
- If you want cleaner comparisons in a presentation, note the key values: `drift_detected`, `f1`, `fpr`, `robustness_score`, and `promoted`.
- If you rerun the same drift data again, the knowledge base may mark it as already remediated, so that repeated run may become a validation-style run instead of a fresh drift reaction.
