"""
Evaluation Agent — Analyze phase of MAPE-K.
Computes F1, precision, recall, ROC-AUC, and FPR on the test set.
Sets feedback loop flags:
  L1 (needs_rebalance) — if recall is low, data may need rebalancing
  L2 (needs_strategy_refinement) — if F1 is degrading, strategy needs refinement
  L5 — logs evaluation to KB

Bug-2 fix: rebuilds test set from latest combined data when scraped data exists.
Bug-5 fix: factors post_augmentation_imbalanced flag into L1 trigger.
Bug-7 fix: detects F1 degradation across iterations → triggers L2 for refinement.
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
from sklearn.model_selection import train_test_split

from config import F1_THRESHOLD, PRECISION_THRESHOLD, FPR_THRESHOLD, TEST_SIZE, RANDOM_STATE


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
    print(" [ EVALUATION AGENT ] Computing test-set metrics...")
    print("=" * 60)

    kb = state["knowledge_base"]
    feature_cols = state["feature_cols"]
    target_col = state["target_col"]

    l1_count = state.get("l1_count", 0)
    l2_count = state.get("l2_count", 0)

    # ── Bug-2 fix: rebuild test set from latest data ─────────────
    scraped_df = state.get("scraped_df")
    raw_df = state.get("raw_df")

    if scraped_df is not None and len(scraped_df) > 0 and raw_df is not None:
        # Combine original + scraped data, then re-split
        required_cols = feature_cols + [target_col]
        combined = pd.concat([
            raw_df[required_cols],
            scraped_df[[c for c in required_cols if c in scraped_df.columns]],
        ], ignore_index=True).dropna(subset=[target_col])
        # Vary seed by loop count so each iteration gets a slightly different hold-out
        split_seed = RANDOM_STATE + l1_count + l2_count
        _, test_df = train_test_split(
            combined, test_size=TEST_SIZE, random_state=split_seed,
            stratify=combined[target_col],
        )
        test_df = test_df.reset_index(drop=True)
        print(f"  🔄 Refreshed test set from latest data ({len(combined)} combined → {len(test_df)} test rows, seed={split_seed})")
    else:
        test_df = state["test_df"]

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values.astype(int)

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

    print(f"  ✅ Test-set metrics: {metrics}")

    # ── Bug-7 fix: detect F1 degradation for L2 trigger ──────────
    prev_f1 = state.get("prev_eval_f1")
    best_f1 = state.get("best_eval_f1", 0.0)
    current_f1 = metrics["f1"]

    f1_degraded = prev_f1 is not None and current_f1 < prev_f1
    if f1_degraded:
        print(f"  ⬇ F1 degraded: {prev_f1:.4f} → {current_f1:.4f}")

    # Track best model across iterations
    new_best_model = state.get("best_model")
    new_best_model_path = state.get("best_model_path", "")
    new_best_f1 = best_f1

    if current_f1 >= best_f1:
        new_best_f1 = current_f1
        new_best_model = model
        new_best_model_path = state.get("candidate_model_path", state.get("current_model_path", ""))
        print(f"  🏆 New best F1: {new_best_f1:.4f}")

    # ── Bug-5 fix: factor post-augmentation imbalance into L1 ────
    post_aug_imbalanced = state.get("post_augmentation_imbalanced", False)

    # ── L1 trigger: poor recall / high FPR / post-augmentation imbalance ──
    needs_rebalance = (
        metrics["recall"] < 0.5
        or metrics["fpr"] > FPR_THRESHOLD
        or (post_aug_imbalanced and metrics["precision"] < PRECISION_THRESHOLD)
    )

    # ── Suppress L1 if balance already determined no action needed ──
    balance_action = state.get("balance_report", {}).get("action", "")
    if needs_rebalance and balance_action == "skipped":
        print("  ℹ️  Suppressing L1: Balance Agent already determined no rebalancing needed.")
        needs_rebalance = False

    # ── L2 trigger: F1 degradation → strategy needs refinement ──
    needs_strategy = (
        metrics["f1"] < F1_THRESHOLD
        or f1_degraded  # Bug-7: degradation triggers L2 refinement
    )

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
        "f1_degraded": f1_degraded,
        "post_augmentation_imbalanced": post_aug_imbalanced,
        "test_set_refreshed": scraped_df is not None and len(scraped_df) > 0,
        "test_set_size": len(test_df),
    })

    return {
        **state,
        "test_df": test_df,  # update test_df in state with refreshed version
        "eval_metrics": metrics,
        "candidate_needs_evaluation": False,
        "needs_rebalance": needs_rebalance,
        "needs_strategy_refinement": needs_strategy,
        "l1_count": l1_count + (1 if needs_rebalance else 0),
        "l2_count": l2_count + (1 if needs_strategy else 0),
        "knowledge_stage": "post_evaluation",
        "prev_eval_f1": current_f1,
        "best_eval_f1": new_best_f1,
        "best_model": new_best_model,
        "best_model_path": new_best_model_path,
    }
