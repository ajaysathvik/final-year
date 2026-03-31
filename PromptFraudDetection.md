# Codex Prompt: Multi-Agent Fraud Detection System with MAPE-K, LangGraph, CTGAN, and Adversarial Training

You are a senior Python engineer, ML systems architect, and agentic AI developer. Build a financial fraud detection platform as a decision-making multi-agent system with adaptive learning, orchestrated using LangGraph.

This is not a single static classifier. It must be a MAPE-K-driven adaptive system with explicit agents, feedback loops, policy-based decisions, drift handling, optional CTGAN augmentation, and adversarial training.

---

## Objective

Implement a working prototype that:

1. ingests fraud transaction data from CSV,
2. detects feature drift and optionally concept drift,
3. evaluates current model performance,
4. makes policy-driven decisions about retraining or skipping updates,
5. adapts strategy dynamically,
6. handles fraud class imbalance using CTGAN when needed,
7. performs adversarial training using attack-like perturbed fraud samples,
8. retrains and validates a candidate model,
9. runs simulation/robustness checks before promotion,
10. promotes the new model only if it satisfies policy thresholds,
11. stores logs, metrics, decisions, and model metadata in a knowledge layer.

---

## Architecture Constraint (MAPE-K)

- Monitor → ingestion + drift detection  
- Analyze → model evaluation + drift interpretation  
- Plan → policy agent + strategy agent  
- Execute → CTGAN + adversarial + training + simulation + deployment  
- Knowledge → logs, metrics, models, decisions  

---

## Mandatory Agents

### drift_agent
Detect feature drift (PSI/KS/KL) and optionally concept drift.

### evaluation_agent
Compute F1, precision, recall, ROC-AUC, FPR.

### policy_agent
Decide retrain / CTGAN / adversarial / deploy.

### strategy_agent
Define adaptation plan (rule-based initially).

### augmentation_agent
Generate synthetic fraud samples using CTGAN.

### adversarial_agent
Generate adversarial fraud samples (noise, boundary, evasion).

### training_agent
Train model (XGBoost preferred, RandomForest fallback).

### simulation_agent
Test robustness before deployment.

### knowledge_agent
Log all events and metrics (JSONL).

---

## LangGraph Flow

START  
→ drift_agent  
→ evaluation_agent  
→ policy_agent  
→ strategy_agent  
→ (if use_ctgan) augmentation_agent  
→ (if use_adversarial) adversarial_agent  
→ training_agent  
→ evaluation_agent  
→ simulation_agent  
→ policy_agent  
→ knowledge_agent  
→ END  

---

## Adversarial Training Requirement

Training data must be:

clean_data + ctgan_data + adversarial_data

Include at least:
- noise perturbation
- boundary attack 
- evasion-style mutation

---

## Promotion Logic

Promote model only if:
- F1 ≥ threshold
- precision ≥ threshold
- FPR ≤ threshold
- simulation passes
- better than previous model

---

## Tech Stack

- Python
- pandas
- scikit-learn
- LangGraph
- CTGAN
- joblib or MLflow
- optional: xgboost, river, shap

---

## Deliverables

- Full working code (modular)
- LangGraph orchestration
- README + instructions
- Example run
- Logs and outputs
- Extension notes

---

## One-Line Goal

Build a drift-aware, adversarially trained, policy-governed, multi-agent fraud detection system with adaptive learning using LangGraph and MAPE-K.
