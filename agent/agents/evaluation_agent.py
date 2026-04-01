"""
Evaluation Agent — Analyze phase of MAPE-K.
Computes F1, precision, recall, ROC-AUC, and FPR on the test set.
Sets feedback loop flags:
  L1 (needs_rebalance) — if recall is low, data may need rebalancing
  L2 (needs_strategy_refinement) — if F1 is low, strategy needs tuning
  L5 — logs evaluation to KB
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from config import F1_THRESHOLD, PRECISION_THRESHOLD, FPR_THRESHOLD


def _compute_metrics(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute standard classification metrics."""
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel() if len(np.unique(y)) > 1 else (0, 0, 0, 0)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics = {
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4) if len(np.unique(y)) > 1 else 0.0,
        "fpr": round(fpr, 4),
    }
    return metrics


def evaluation_agent(state: dict) -> dict:
    """
    Evaluate the candidate model on the test set.
    Sets L1/L2 feedback flags based on metric thresholds.
    Logs results to Knowledge Base (L5).
    """
    print("\n" + "=" * 60)
    print(" [ EVALUATION AGENT ] Computing model metrics...")
    print("=" * 60)

    kb = state["knowledge_base"]
    test_df: pd.DataFrame = state["test_df"]
    feature_cols = state["feature_cols"]
    target_col = state["target_col"]

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values.astype(int)

    l1_count = state.get("l1_count", 0)
    l2_count = state.get("l2_count", 0)

    # ── KB context: compare against prior eval (KB→Eval communication) ──
    prior_eval = kb.get_latest("evaluation_history")
    if prior_eval:
        prior_f1 = prior_eval.get("data", {}).get("f1", "n/a")
        prior_recall = prior_eval.get("data", {}).get("recall", "n/a")
        print(f"  📖 KB context: prior eval → F1={prior_f1}, recall={prior_recall}")

    # Evaluate candidate model if available, otherwise current model
    model = state.get("candidate_model") or state.get("current_model")
    if model is None:
        print("  ⚠️  No model found. Returning zero metrics.")
        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "roc_auc": 0.0, "fpr": 1.0}
    else:
        metrics = _compute_metrics(model, X_test, y_test)

    print(f"  ✅ Metrics: {metrics}")

    # ── L1 trigger: poor recall / high FPR → needs rebalancing ──
    needs_rebalance = (
        metrics["recall"] < 0.5 or metrics["fpr"] > FPR_THRESHOLD
    )

    # ── L2 trigger: low F1 → strategy needs refinement ──────────
    needs_strategy = metrics["f1"] < F1_THRESHOLD

    # Don't loop if we've already looped enough
    if l1_count > 0 and needs_rebalance:
        print(f"  ℹ️  L1 already iterated {l1_count} time(s).")
    if l2_count > 0 and needs_strategy:
        print(f"  ℹ️  L2 already iterated {l2_count} time(s).")

    # ── L5: log evaluation to KB ────────────────────────────────
    kb.log_event("evaluation", "evaluation_completed", {
        **metrics,
        "l1_count": l1_count,
        "l2_count": l2_count,
        "needs_rebalance": needs_rebalance,
        "needs_strategy_refinement": needs_strategy,
    })

    return {
        **state,
        "eval_metrics": metrics,
        "needs_rebalance": needs_rebalance,
        "needs_strategy_refinement": needs_strategy,
        "l1_count": l1_count + (1 if needs_rebalance else 0),
        "l2_count": l2_count + (1 if needs_strategy else 0),
    }
