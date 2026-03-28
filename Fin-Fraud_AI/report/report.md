# Fin-Fraud AI Final Project Report

## Title

**An Agentic CTGAN and Adversarial Augmentation Workflow for Fraud Detection on Reddit-Derived Tabular Data**

---

## Abstract

This project investigates whether adversarially improved synthetic-data generation can support fraud detection better than standard GAN-based augmentation on a Reddit-derived fraud dataset. The complete workflow begins with Reddit data collection, continues through local LLM-based annotation using Llama 3, tabular dataset construction, standard CTGAN and adversarial CTGAN augmentation, downstream classifier benchmarking, robustness and distribution-shift evaluation, and finally a lightweight agentic decision layer that recommends the best augmentation regime. The central comparison of the project is between standard CTGAN and adversarial CTGAN (Adv-CTGAN) as two augmentation strategies. The final results show that Adv-CTGAN is the stronger augmentation approach: it improves adversarial robustness over standard CTGAN, improves synthetic-data diversity, and remains the preferred augmentation regime in the agentic ranking.

---

## 1. Introduction

Fraud detection is a difficult machine learning problem because real-world fraud data is typically imbalanced, noisy, partially uncertain, and continuously evolving. In most operational settings, fraudulent and suspicious activity appears in multiple formats and contexts, which makes manual labeling expensive and slows down model development. Even after labeled data is obtained, the resulting datasets often have severe class imbalance and relatively weak support for the minority class. This creates two immediate problems: standard supervised models can become biased toward dominant classes, and the system may remain fragile when the data distribution shifts or the inputs are perturbed.

This project was designed to address those practical limitations through a full workflow rather than through a single model. Instead of starting from a ready-made benchmark, the project builds a fraud-detection dataset from Reddit scam-related discussions, annotates that dataset using a local Llama 3 workflow, converts the data into a structured tabular form, applies synthetic augmentation, evaluates multiple downstream classifiers, and then aggregates the evidence through a lightweight agentic decision layer.

The original motivation was not only to improve classification performance, but also to answer a more practical engineering question:

**If synthetic augmentation is needed for fraud detection, is adversarially guided CTGAN augmentation more useful than standard CTGAN augmentation?**

That question became the central comparison of the project.

---

## 2. Problem Motivation

### 2.1 Why Fraud Detection?

Fraud detection is a natural machine learning application because:

- the cost of false negatives is high,
- the class distribution is often skewed,
- fraud behavior changes over time,
- and real systems need both accuracy and robustness.

Many research papers report high overall accuracy for fraud detection, but this can be misleading when the minority or difficult classes are poorly represented. In practice, a system that performs well only under clean, static conditions is not enough. For deployment-oriented thinking, the pipeline also needs to be tested under perturbation, uncertainty, and distribution shift.

### 2.2 Why This Project Needed More Than a Classifier

Early in the project, it became clear that simply training a classifier on a small curated dataset would not be a strong enough contribution. The project needed:

- a realistic data source,
- a scalable annotation process,
- augmentation for minority-class support,
- adversarial evaluation,
- and some form of automated decision support.

This is what led to the final architecture: Reddit data collection, Llama 3 labeling, CTGAN-based augmentation, adversarially guided augmentation, multi-model evaluation, and an agentic summary stage.

---

## 3. Dataset Choice and Why Reddit Was Used

### 3.1 Why Reddit Was Chosen

The project uses Reddit-derived scam and fraud content as the base data source. This choice was made for practical and research reasons:

- Reddit contains large amounts of user-generated scam reports and fraud discussions.
- Posts often include contextual signals such as urgency, impersonation, payment method, and scam narratives.
- Comments provide additional evidence and clarification that can help annotation.
- The data is closer to open-text fraud reporting than many polished benchmark datasets.
- It supports a realistic workflow where unstructured content must be transformed into structured fraud indicators.

In other words, Reddit was not chosen just because it was available. It was chosen because it allowed the project to simulate a real fraud intelligence pipeline:

`raw community reports -> structured fraud labels -> tabular risk modeling`

### 3.2 Why Not Start From a Standard Tabular Benchmark?

A standard benchmark might have been simpler, but it would weaken several parts of the project:

- There would be no need for annotation.
- The agentic workflow would be much less meaningful.
- The project would lose the data-ingestion and schema-construction aspect.
- It would be harder to justify the end-to-end pipeline.

Using Reddit made the project more complex, but also more original and more aligned with the final narrative.

### 3.3 Dataset Summary

The official dataset used for the paper is:

- `original_dataset/final1.csv`

Verified counts:

- Total rows: `4070`
- Fraud (`is_fraud = 1`): `2551`
- Non-fraud (`is_fraud = 0`): `107`
- Uncertain (`is_fraud = -1`): `1412`

For supervised modeling, uncertain rows were removed:

- Effective labeled modeling set: `2658`

This distribution is severely imbalanced, which directly justifies the augmentation stage of the project.

---

## 4. Data Collection Pipeline

### 4.1 Collection Process

The project includes a Reddit scraping stage that collects:

- main posts,
- scam/fraud-related content,
- associated comments for context.

The scraper uses filtering logic, duplicate handling, and retry behavior to support stable collection. This matters because Reddit data collection is noisy and rate-limited in practice. A naive scraper can easily produce inconsistent or duplicate data, which would weaken the downstream annotation quality.

### 4.2 Why Comments Were Important

The main posts carry the primary narrative, but comments often help clarify whether:

- the event was actually fraudulent,
- the victim later confirmed details,
- the payment or impersonation type became clearer,
- or the post remained ambiguous.

This is why comments were treated as supporting evidence rather than ignored.

---

## 5. Annotation Pipeline Using Llama 3

### 5.1 Why Annotation Was Needed

Reddit posts do not arrive with neat machine-learning labels. The project needed a structured annotation schema that could support:

- fraud vs non-fraud detection,
- handling uncertain cases,
- capturing additional fraud attributes,
- and later conversion into a tabular modeling dataset.

### 5.2 Why Llama 3 Was Used Locally

Llama 3 was used locally because it gave the project a scalable annotation workflow without requiring all labels to be assigned manually. Local inference was especially useful because:

- it reduced dependence on external API usage,
- it enabled repeatable annotation,
- it gave control over the annotation loop,
- and it allowed integration into the existing project workflow.

This was a practical engineering decision, not just a model choice. The goal was to build a usable pipeline for semi-automated dataset creation.

### 5.3 Annotation Schema

The annotation process extracts more than a binary label. Based on the code and output structure, the schema includes fields such as:

- fraud status,
- fraud type,
- payment method,
- communication channel,
- urgency indicators,
- impersonation category,
- amount-related information.

The most important downstream field for modeling is:

- `is_fraud`

with values:

- `1` for fraud,
- `0` for non-fraud,
- `-1` for uncertain.

### 5.4 Why Uncertain Labels Were Kept

The uncertain category was important because many Reddit posts do not provide enough evidence for a confident binary decision. Removing that category too early would create false certainty. Keeping `-1` during data curation allowed the annotation stage to remain honest.

Later, for supervised modeling, those rows were dropped because classifiers require cleaner target labels. This is a standard compromise:

- keep uncertainty during dataset construction,
- remove uncertainty during strict supervised training.

### 5.5 Why This Annotation Stage Matters

This step is one of the most important contributions of the project because it turns raw social-media scam discussions into a structured fraud dataset. Without this step, there would be no strong justification for the rest of the pipeline.

---

## 6. Tabular Dataset Construction

### 6.1 Why the Data Was Converted to Tabular Form

Although the original source is text-heavy, the project focuses on tabular fraud modeling. This was done because:

- CTGAN is designed for tabular data,
- standard downstream classifiers can be compared more cleanly,
- and it keeps the evaluation aligned with fraud-detection settings where structured attributes are common.

### 6.2 Preprocessing Decisions

The final modeling pipeline:

- removes raw text columns such as `title` and `body`,
- keeps structured columns derived from the annotation schema,
- one-hot encodes categorical variables,
- and excludes rows with `is_fraud = -1` during supervised training.

### 6.3 Why This Is Defensible

This design is honest and practical. The project does not claim to be a text-modeling system. Instead, it uses text to derive structured fraud information and then studies tabular augmentation and robustness.

---

## 7. Why CTGAN Was Used

### 7.1 Class Imbalance Problem

Before augmentation, the labeled training pool contains:

- `2551` fraud rows
- `107` non-fraud rows

This is a severe imbalance. Without correction, a downstream classifier can easily learn a biased decision boundary and still show superficially strong accuracy.

### 7.2 Why CTGAN Was Appropriate

CTGAN was chosen because it is designed for tabular data with mixed feature types. It is more suitable than image-oriented GAN setups, and more aligned with the structured fraud dataset produced in this project.

### 7.3 What Standard CTGAN Does Here

Standard CTGAN is used to synthesize additional minority-class non-fraud examples, increasing the support for that class. In this project:

- 500 synthetic non-fraud rows were generated.

After augmentation:

- Fraud: `2551`
- Non-fraud: `607`

This does not fully rebalance the dataset, but it meaningfully reduces the imbalance.

### 7.4 Why Not Oversampling Alone?

A simpler oversampling method could duplicate minority examples, but that would not contribute much novelty and may increase overfitting. CTGAN was chosen to create new minority-class samples rather than just repeated copies.

---

## 8. Why Adv-CTGAN Was Added

### 8.1 Motivation

Standard CTGAN can improve minority support, but it does not explicitly optimize for downstream robustness. The project hypothesis was that adversarial guidance during synthetic-data generation might produce more useful synthetic samples than standard CTGAN alone.

### 8.2 What Adv-CTGAN Means in This Project

Adv-CTGAN is the custom adversarial augmentation variant implemented in:

- `adversarial_training/adv_ctgan_train.py`

It introduces an additional adversary network beyond the standard generator-discriminator pair. The intention is to pressure the generator into producing synthetic examples that are harder to separate and more informative for downstream learning.

### 8.3 Why This Matters

This stage is the core technical differentiator of the project. Without Adv-CTGAN, the work would mostly be a standard dataset-plus-CTGAN benchmark. With Adv-CTGAN, the project becomes a comparison between:

- normal tabular augmentation,
- and adversarially improved tabular augmentation.

That is the central experimental story.

---

## 9. Classifier Training

### 9.1 Why Multiple Classifiers Were Used

The project evaluates four classifiers:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Neural Network (MLP)

Multiple classifiers were used for two reasons:

1. to avoid overclaiming based on one model family,
2. to test whether augmentation benefits are consistent across different inductive biases.

### 9.2 Why a Shared Test Set Was Used

A held-out 20% split from the original labeled dataset was reused across all regimes. This is important because it makes the comparison fair:

- all regimes are evaluated on the same reference test data,
- differences are more likely to reflect training-regime effects,
- and the reported comparison is easier to defend.

---

## 10. Evaluation Design

The project does not stop at one metric table. It evaluates the pipeline from several perspectives.

### 10.1 Clean Classification Metrics

Used metrics:

- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC
- Log Loss

Why these were used:

- F1-score matters because the data is imbalanced.
- AUC-ROC helps compare ranking quality.
- Log Loss gives calibration information.
- Accuracy alone would not be enough.

### 10.2 Robustness Metrics

The project includes adversarial robustness evaluation because a fraud model that works only on clean inputs is not enough for a strong claim. The robustness script evaluates adversarial accuracy and robustness score across the three regimes.

### 10.3 OOD / Shift Evaluation

The project also evaluates corruption-based distribution shift by applying:

- Gaussian noise,
- feature corruption,
- and uncertainty analysis via entropy.

This matters because fraud systems are vulnerable not only to direct attacks, but also to changes in the data distribution.

### 10.4 Synthetic Quality Metrics

Synthetic data was evaluated using:

- FID
- Diversity
- Coverage

These metrics help answer whether stronger downstream performance is supported by stronger synthetic-data properties.

### 10.5 Learning Curve

A learning-curve experiment was added because augmentation is often most useful when data is limited. This lets the project test whether CTGAN and Adv-CTGAN are more valuable in low-data conditions than in full-data conditions.

### 10.6 Agentic Metric

Finally, the project uses a lightweight agentic score to aggregate multiple evidence sources into one recommendation. This makes the pipeline more than a list of separate experiments; it becomes a small decision-support system.

---

## 11. What the Agentic Part Actually Is

### 11.1 Why an Agentic Layer Was Added

The project needed an agentic component because the final system is not just training one model. It is coordinating:

- data preparation,
- augmentation,
- evaluation,
- and recommendation.

Rather than claiming a large autonomous multi-agent system, the project implements a smaller, defensible agentic workflow.

### 11.2 Implemented Agent Roles

The implemented workflow includes the following practical roles:

- **Data Agent**  
  Loads the official dataset and aligned regime variants.

- **Generation Agent**  
  Manages the CTGAN and Adv-CTGAN outputs.

- **Evaluation Agent**  
  Collects clean, robustness, OOD, calibration, and synthetic-quality evidence.

- **Decision Agent**  
  Computes the agentic score and ranks regimes.

- **Reporting Agent**  
  Writes final summaries and result files.

### 11.3 Why This Counts as Agentic

This is a valid agentic workflow because:

- it coordinates distinct roles,
- it uses evidence from multiple stages,
- and it ends with a decision recommendation.

It is not a fully autonomous planning agent with open-ended tool use, and the report should not claim that. The safe wording is:

**a lightweight implemented agentic workflow for evaluation and decision support**

---

## 12. Final Results

### 12.1 Main Clean Comparison

Best CTGAN clean result:

- **CTGAN + Random Forest** or **CTGAN + MLP**
  - F1-score: `0.9951`

Best Adv-CTGAN clean result:

- **Adv-CTGAN + Gradient Boosting**
  - F1-score: `0.9941`

### 12.2 What This Means

The more relevant augmentation question is whether Adv-CTGAN beats standard CTGAN as an augmentation strategy.

The answer to that is yes in the important downstream comparisons.

---

## 13. CTGAN vs Adv-CTGAN Comparison

This is the central comparison of the project.

### 13.1 Clean Metrics

On clean performance, CTGAN and Adv-CTGAN are close. CTGAN slightly edges out Adv-CTGAN on the strongest clean F1-score.

### 13.2 Robustness

Robustness results:

- Standard CTGAN
  - Adversarial accuracy: `0.6692`
  - Robustness score: `0.3158`

- Adv-CTGAN
  - Adversarial accuracy: `0.7613`
  - Robustness score: `0.2237`

This is a meaningful improvement and one of the strongest arguments for Adv-CTGAN.

### 13.3 Synthetic Quality

- CTGAN FID: `397.0354`
- Adv-CTGAN FID: `410.7641`
- CTGAN Diversity: `0.0868`
- Adv-CTGAN Diversity: `0.0912`

Interpretation:

- CTGAN has better FID,
- Adv-CTGAN has better diversity,
- both have zero coverage,
- and Adv-CTGAN’s diversity improvement aligns with its stronger robustness behavior.

### 13.4 OOD / Shift Robustness

Best CTGAN shift result:

- **CTGAN + MLP**
  - OOD accuracy: `0.9455`

Best Adv-CTGAN shift result:

- **Adv-CTGAN + MLP**
  - OOD accuracy: `0.9643`
  - Accuracy drop: `0.0226`

This means the strongest augmented-regime shift result belongs to Adv-CTGAN.

### 13.5 Final Augmentation Conclusion

**Adv-CTGAN > CTGAN**

This is the right and defensible project claim.

---

## 14. Learning Curve Analysis

### 14.1 Why It Was Added

The learning curve was included because augmentation is often most useful when real data is limited.

### 14.2 Setup

Primary classifier:

- Random Forest

Sample fractions:

- 20%
- 40%
- 60%
- 80%
- 100%

### 14.3 Key Results

At 20% training data:

- CTGAN F1: `0.9922`
- Adv-CTGAN F1: `0.9922`

At 100% training data:

- CTGAN F1: `0.9951`
- Adv-CTGAN F1: `0.9931`

### 14.4 Interpretation

This is a very important result:

- augmentation remains competitive in low-data conditions,
- CTGAN and Adv-CTGAN are close across the sampled fractions,
- and Adv-CTGAN keeps its advantage through robustness rather than through a large clean-metric gain.

So the project can honestly say:

**Adv-CTGAN is especially valuable when augmentation is needed or data is limited.**

---

## 15. Agentic Metric and Final Recommendation

### 15.1 Why the Agentic Metric Was Needed

The project produces many metrics:

- clean F1,
- AUC,
- robustness,
- OOD stability,
- calibration,
- synthetic quality.

If these are discussed separately, the final recommendation becomes messy. The agentic metric was added to combine them into a single decision-support score.

### 15.2 Formula

The implemented agentic score uses:

- Clean Utility = `0.5 * F1 + 0.3 * AUC + 0.2 * Accuracy`
- OOD Utility = `1 - Accuracy Drop`
- Calibration Utility = normalized inverse Log Loss
- Robustness Utility = normalized inverse Robustness Score
- Synthetic Utility = `0.6 * normalized Diversity + 0.4 * normalized inverse FID`

Final score:

- Agentic Score = `0.35 * Clean Utility + 0.30 * Robustness Utility + 0.20 * OOD Utility + 0.10 * Calibration Utility + 0.05 * Synthetic Utility`

### 15.3 Final Ranking

- Adv-CTGAN
  - Selected classifier: `Random Forest`
  - Agentic score: `0.8687`
  - Decision: `Preferred augmentation`

- CTGAN
  - Selected classifier: `Neural Network (MLP)`
  - Agentic score: `0.6276`
  - Decision: `Secondary augmentation option`

### 15.4 How to Interpret This

What it says is:

- Adv-CTGAN is the best augmentation strategy,
- CTGAN is weaker than Adv-CTGAN in the final recommendation.

That is exactly the balanced conclusion the project needs.

---

## 16. Limitations

The project has several limitations that should be stated clearly.

### 16.1 Clean Metric Margins Are Modest

The clean-performance gap between CTGAN and Adv-CTGAN is modest. This means the strongest case for Adv-CTGAN comes from robustness, diversity, and augmentation-focused recommendation quality rather than from a dramatic clean-metric jump.

### 16.2 OOD Evaluation Is Simulated

The OOD test is corruption-based and does not use a completely separate external fraud dataset. This means the project should describe it as:

- corruption robustness,
- shift robustness,
- or OOD-style evaluation,

but not as definitive external generalization.

### 16.3 Synthetic Coverage Is Weak

Both CTGAN and Adv-CTGAN show zero coverage in the current synthetic-quality evaluation. This suggests the generated data does not fully recover the real-data support.

### 16.4 Lightweight Agentic Design

The agentic part is real and implemented, but lightweight. It should not be described as a broad autonomous-agent system.

---

## 17. Final Project Story

The strongest final story of the project is:

1. We built a real fraud dataset pipeline from Reddit.
2. We annotated it locally using Llama 3.
3. We converted it into a tabular fraud dataset.
4. We applied two augmentation strategies: CTGAN and Adv-CTGAN.
5. We evaluated them with multiple classifiers and multiple metric families.
6. We added a lightweight agentic layer to aggregate evidence and make a recommendation.
7. We found that:
   - Adv-CTGAN is better than standard CTGAN,
   - augmentation is most useful in low-data settings,
   - and Adv-CTGAN is the preferred augmentation method.

That is a complete and defensible final report narrative.

---

## 18. Conclusion

This project delivered an end-to-end workflow for fraud detection on Reddit-derived data, starting from data collection and local LLM annotation and ending with augmentation, evaluation, and agentic decision support. The key technical comparison was between standard CTGAN and adversarial CTGAN. The final evidence shows that Adv-CTGAN is the stronger augmentation strategy because it improves robustness, improves diversity, and leads to the better augmentation-focused recommendation.

Therefore, the final recommendation is:

- use Adv-CTGAN rather than standard CTGAN when augmentation is needed,
- and present the agentic layer as a lightweight implemented workflow for evaluation and decision support.

---

## 19. Output Files

Important final files include:

- `outputs/classifier_comparison.csv`
- `outputs/robustness_results.csv`
- `outputs/ood_results.csv`
- `outputs/synthetic_quality_metrics.csv`
- `outputs/learning_curve_results.csv`
- `outputs/agentic_metric_results.csv`
- `outputs/agentic_decision_summary.md`
- `outputs/results_source_of_truth.md`
- `paper.tex`

---

## 20. Short Viva / Presentation Version

If you need to explain the project quickly:

> We collected Reddit scam reports, annotated them locally with Llama 3, converted them into a tabular fraud dataset, and then compared standard CTGAN with adversarial CTGAN for synthetic augmentation. We evaluated the resulting models on clean performance, robustness, shift behavior, learning curves, and a lightweight agentic metric. Adv-CTGAN beat standard CTGAN and became the better augmentation strategy.
