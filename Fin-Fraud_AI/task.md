# Fin-Fraud AI Paper Task List

This file is the working checklist for turning the current project into a clean, defensible paper. Top items are the ones we should do first because they unblock everything else.

## Deadline Mode: 1-Day Submission Plan

This is the active plan for the next 1 day. We are optimizing for a decent conference-style paper and a usable final project report using the experiments already available in the repository.

### Do First

- [x] Define the implemented agentic workflow clearly in the paper and report.
- [ ] Rewrite `paper.tex` as a proper results paper instead of a proposal-style draft.
- [x] Keep updating `report/report.md` as the living project report.
- [x] Define and compute one lightweight agentic metric using existing outputs.
- [ ] Finalize figures, tables, and wording for submission.

### Do Only If Time Remains

- [x] Run a minimal learning-curve experiment on one main classifier.
- [ ] Add one extra ablation only if it is fast and clean.

### Do Not Prioritize Now

- [ ] Large new architecture changes
- [ ] many new experiments across all models
- [ ] ambitious autonomous-agent implementation work
- [ ] anything that risks creating inconsistencies near submission

## Immediate Priority

- [x] Freeze one dataset version as the paper's official dataset and document the exact counts.
- [x] Official dataset baseline:
  - [x] `original_dataset/final1.csv` is the paper source file.
  - [x] Total rows: `4070`
  - [x] Label distribution: `2551 fraud`, `107 non-fraud`, `1412 uncertain (-1)`
  - [x] After dropping uncertain rows for modeling: `2658` rows
  - [x] Modeling distribution before augmentation: `2551 fraud`, `107 non-fraud`
  - [x] CTGAN / Adv-CTGAN augmented rows: `4570`
  - [x] Modeling distribution after augmentation: `2551 fraud`, `607 non-fraud`, `1412 uncertain (-1)` in stored CSVs
- [x] Reconcile dataset statistics across `paper.tex`, `report/report.md`, code comments, and output tables.
- [x] Decide the final paper claim: "implemented agentic workflow" vs "proposed agentic framework with partial implementation."
- [x] Final claim chosen: `implemented agentic workflow`
- [x] Update `paper.tex` so it reflects completed experiments instead of saying metrics are "not yet finalized."
- [x] Create one source-of-truth results summary from the current CSV outputs.
- [x] Verify that `classifier_comparison.csv`, `robustness_results.csv`, and `ood_results.csv` match the numbers we plan to cite in the paper.

## Paper Framing

- [ ] Rewrite the paper scope so it matches the actual project: Reddit fraud dataset, LLM annotation, synthetic augmentation, adversarial evaluation, and agentic orchestration.
- [ ] Add a clear contribution list in the Introduction.
- [ ] Add explicit research questions for:
  - [ ] raw vs CTGAN vs Adv-CTGAN
  - [ ] robustness under attack
  - [ ] learning curve behavior
  - [ ] agentic evaluation / decision value
- [ ] Make sure the title, abstract, methodology, and conclusion all tell the same story.

## Agentic Framework

- [x] Define the "agentic framework" in a way we can honestly support.
- [ ] Decide the agents to describe in the paper, likely:
  - [x] Data Agent
  - [ ] Labeling / Validation Agent
  - [x] Generation Agent
  - [x] Evaluation Agent
  - [x] Decision Agent
- [x] Write the agent workflow as an end-to-end loop from data ingestion to model recommendation.
- [x] Decide whether we will implement a lightweight orchestration script/module for the agentic layer.
- [x] If yes, add a minimal controller that executes stages and records decisions.

## Agentic Metric

- [x] Define what "agentic metric" means for this paper.
- [x] Pick a measurable formulation, for example a weighted score using clean performance, robustness, OOD stability, and calibration.
- [x] Write the exact formula and justify each component.
- [x] Compute the metric for Raw, CTGAN, and Adv-CTGAN settings.
- [ ] Add one comparison table and one figure for the agentic metric.

## Learning Curve Experiment

- [x] Design the learning curve experiment.
- [x] Choose the sample sizes to test, for example 20%, 40%, 60%, 80%, 100%.
- [ ] Decide whether to vary:
  - [x] amount of real training data
  - [ ] amount of synthetic augmentation
  - [ ] both
- [x] Run the experiment for the main classifier(s).
- [ ] Plot learning curves for F1, AUC, and robustness if feasible.
- [x] Add analysis explaining whether synthetic and adversarial augmentation help more in low-data settings.

## Raw vs GAN vs Adversarial Comparison

- [x] Make one main experiment table comparing:
  - [x] Raw / Original
  - [x] CTGAN
  - [x] Adv-CTGAN
- [x] Pick the primary metrics to highlight in the paper.
- [x] Identify the best classifier per dataset regime.
- [x] Summarize where Adv-CTGAN actually helps and where it does not.

## Robustness Evaluation

- [ ] Verify FGSM and PGD settings and document them clearly.
- [x] Confirm which classifier is being attacked in the robustness script.
- [ ] Decide whether robustness should be reported for only one classifier or all four.
- [x] Add the robustness gap definition clearly in the Methodology.
- [ ] Create one clean robustness figure for the paper.
- [ ] Explain any surprising result, especially where Adv-CTGAN underperforms or shifts by classifier.

## OOD / Robustness Under Shift

- [x] Keep the OOD experiment as a robustness-under-shift result if it strengthens the paper.
- [x] Document the corruption process exactly.
- [x] Decide whether to call this OOD, corruption robustness, or distribution shift robustness.
- [x] Add one concise discussion of what the OOD numbers mean.

## Dataset and Annotation Section

- [ ] Document the Reddit data collection pipeline clearly.
- [ ] Describe the subreddits, filtering logic, and duplicate handling.
- [ ] Describe the LLM annotation schema.
- [ ] Clarify how uncertain labels (`is_fraud = -1`) are handled.
- [ ] Add a short annotation quality / validation note.
- [ ] Include a final feature table describing important columns used for modeling.

## Methodology Cleanup

- [ ] Replace generic GAN language with the actual CTGAN and Adv-CTGAN setup used in code.
- [ ] Describe preprocessing steps exactly as done in the scripts.
- [ ] Document the train/test split and why the same test set is reused across regimes.
- [ ] Clarify which text fields are removed before tabular modeling.
- [ ] Add hyperparameters for the main models.
- [ ] Add a compact algorithm or workflow description for the overall pipeline.

## Results Section Rewrite

- [x] Replace the current placeholder Results section with actual findings.
- [x] Add a subsection for classifier comparison.
- [x] Add a subsection for robustness evaluation.
- [x] Add a subsection for synthetic data quality.
- [x] Add a subsection for OOD / shift robustness.
- [x] Add a subsection for the learning curve once completed.
- [x] Add a subsection for the agentic metric once completed.

## Figures and Tables

- [ ] Decide which current output plots are paper-worthy.
- [ ] Regenerate any figure that looks too report-like or too busy.
- [ ] Add a table for dataset statistics.
- [ ] Add a table for annotation schema.
- [ ] Add a table for model configurations.
- [ ] Add a main results table.
- [ ] Add a robustness table.
- [ ] Add a learning curve plot.
- [ ] Add an agentic framework diagram if the current architecture image is not enough.

## Validation and Consistency

- [x] Check for metric inconsistencies between `outputs/` and `report/report.md`.
- [x] Check whether the robustness numbers in the report match `outputs/robustness_results.csv`.
- [x] Check that all cited values in the paper come from the final CSV files.
- [ ] Standardize naming everywhere:
  - [ ] Original vs Raw
  - [ ] CTGAN vs Standard CTGAN
  - [ ] Adv-CTGAN vs Adversarial CTGAN

## Writing Polish

- [x] Improve the abstract so it includes dataset, methods, and actual results.
- [ ] Tighten the introduction to avoid overclaiming.
- [ ] Make the Related Work more targeted to fraud detection, GAN augmentation, and adversarial ML.
- [ ] Add a limitations section that is honest about the current implementation.
- [ ] Add future work that naturally follows from the missing experiments.
- [ ] Proofread for grammar, tense consistency, and notation.

## Nice-to-Have

- [ ] Add an ablation on synthetic sample count.
- [ ] Add confidence intervals or repeated runs if time permits.
- [ ] Add threshold analysis or cost-sensitive discussion for fraud detection.
- [ ] Add explainability examples for flagged transactions if we want the agentic story to feel stronger.

## Suggested Execution Order

1. Freeze dataset and resolve all count inconsistencies.
2. Align `paper.tex` with the experiments that already exist.
3. Define the agentic framework conservatively and clearly.
4. Define and compute the agentic metric.
5. Run the learning curve experiment.
6. Rewrite the Results and Discussion sections.
7. Clean figures, tables, and final wording.
