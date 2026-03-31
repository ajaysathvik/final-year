"""
LangGraph shared state definition for the fraud detection pipeline.
Matches the Agentic_Architecture.drawio MAPE-K diagram.
"""
from __future__ import annotations
from typing import Any, TypedDict
import pandas as pd


class PipelineState(TypedDict, total=False):
    # ── Data ───────────────────────────────────────────────────
    raw_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    feature_cols: list[str]
    target_col: str

    # ── Drift Agent ────────────────────────────────────────────
    drift_report: dict[str, Any]
    drift_detected: bool

    # ── Balance Agent (CTGAN / SMOTE) ──────────────────────────
    balanced_train_df: pd.DataFrame  # train_df after balancing
    ctgan_samples: pd.DataFrame
    balance_report: dict[str, Any]

    # ── Supervisor Agent ───────────────────────────────────────
    supervisor_health: dict[str, float]
    supervisor_decision: str          # "proceed" | "rebalance" | "escalate"

    # ── Policy Agent ───────────────────────────────────────────
    policy_decision: dict[str, Any]
    should_retrain: bool
    should_rebalance: bool
    should_skip: bool

    # ── Strategy Agent ─────────────────────────────────────────
    strategy_plan: dict[str, Any]     # model selection, hyperparams, adv strategy

    # ── Training Agent ─────────────────────────────────────────
    current_model: Any
    current_model_path: str
    candidate_model: Any
    candidate_model_path: str
    training_metrics: dict[str, float]

    # ── Evaluation Agent ───────────────────────────────────────
    eval_metrics: dict[str, float]    # F1, precision, recall, roc_auc, fpr
    eval_passed: bool
    needs_rebalance: bool             # L1 trigger flag
    needs_strategy_refinement: bool   # L2 trigger flag

    # ── Simulation Agent (incl. adversarial testing) ───────────
    simulation_results: dict[str, Any]
    simulation_passed: bool
    adversarial_results: dict[str, Any]

    # ── Deployment Agent ───────────────────────────────────────
    promoted: bool
    promoted_model_path: str
    deployment_decision: dict[str, Any]

    # ── Feedback loop counters ─────────────────────────────────
    l1_count: int    # Eval → Balance (data correction & rebalancing)
    l2_count: int    # Eval → Strategy (strategy refinement)
    l3_count: int    # Training self-loop
    l4_count: int    # Supervisor ↔ Policy
    l5_count: int    # Eval → KB (simulated validation)
    kb_loop_count: int  # KB → Drift closed-loop re-cycle counter

    # ── Knowledge Base ref (in-memory, not serialized) ─────────
    knowledge_base: Any
    knowledge_log: list[dict[str, Any]]
