"""
Policy Agent — Plan phase of MAPE-K (Decision / Governance).
Decides: Retrain? Rebalance? Skip Update?
L4 feedback loop with Supervisor.
Receives from Supervisor, routes to Strategy or Balance or End.
"""
from __future__ import annotations

from config import F1_THRESHOLD, MAX_L4_ITERATIONS


def policy_agent(state: dict) -> dict:
    """
    Based on supervisor health assessment and drift status,
    decide on the course of action:
    - should_retrain: train a new model
    - should_rebalance: go back to Balance Agent (L4 loop)
    - should_skip: no action needed
    """
    print("\n" + "=" * 60)
    print(" [ POLICY AGENT ] Decision / Governance...")
    print("=" * 60)

    kb = state["knowledge_base"]
    supervisor_health = state.get("supervisor_health", {})
    supervisor_decision = state.get("supervisor_decision", "proceed")
    eval_metrics = state.get("eval_metrics") or {}
    simulation_passed = state.get("simulation_passed")
    l4_count = state.get("l4_count", 0)
    current_model = state.get("current_model")

    # ── If no in-memory model, try to load the last promoted model from KB ──
    if current_model is None:
        import joblib
        from pathlib import Path
        latest_deploy = kb.get_latest("deployment_records")
        if latest_deploy:
            promoted_path = latest_deploy.get("data", {}).get("promoted_model_path", "")
            if promoted_path and Path(promoted_path).exists():
                try:
                    current_model = joblib.load(promoted_path)
                    print(f"  📦 Loaded previously promoted model from KB: {promoted_path}")
                except Exception as e:
                    print(f"  ⚠️  Could not load promoted model ({e}), treating as no model.")

    # ── KB context: read eval trend (KB→Policy communication) ────
    eval_history = kb.get_history("evaluation_history", limit=3)
    f1_trend = [e.get("data", {}).get("f1", None) for e in eval_history if e.get("data", {}).get("f1") is not None]
    trend_degrading = len(f1_trend) >= 2 and f1_trend[-1] < f1_trend[0]
    if f1_trend:
        print(f"  📖 KB context: F1 trend over last {len(f1_trend)} runs = {f1_trend} "
              f"({'⬇ degrading' if trend_degrading else '➡ stable/improving'})")

    combined_health = supervisor_health.get("combined_health", 0.5)

    # ── Resolve best available F1 ────────────────────────────────
    # eval_metrics is empty before evaluation runs on this cycle.
    # Use it if present; otherwise fall back to the last train_f1 recorded in KB.
    eval_f1 = eval_metrics.get("f1")
    if eval_f1 is not None:
        last_known_f1 = eval_f1
        f1_source = "eval_metrics"
    else:
        latest_train = kb.get_latest("training_records")
        last_known_f1 = latest_train.get("data", {}).get("train_f1", 0.0) if latest_train else 0.0
        f1_source = "kb_training_records" if latest_train else "default_zero"
    print(f"  📊 Last known F1 = {last_known_f1:.4f} (source: {f1_source})")

    # ── Determine action ────────────────────────────────────────
    should_retrain = False
    should_rebalance = False
    should_skip = False
    should_validate_existing = False
    reasons = []

    # If supervisor escalated, consider rebalancing (L4 loop)
    if supervisor_decision == "escalate" and l4_count < MAX_L4_ITERATIONS:
        should_rebalance = True
        reasons.append("supervisor_escalated")
    elif current_model is None:
        should_retrain = True
        reasons.append("no_existing_model")
    elif state.get("drift_detected", False):
        should_retrain = True
        reasons.append("drift_detected")
    elif trend_degrading:
        should_retrain = True
        reasons.append("f1_trend_degrading")
    elif last_known_f1 < F1_THRESHOLD:
        should_retrain = True
        reasons.append(f"last_known_f1={last_known_f1:.4f}<{F1_THRESHOLD}")
    elif eval_metrics.get("f1") is None or simulation_passed is None:
        should_validate_existing = True
        reasons.append("validation_required_for_current_run")
    else:
        should_skip = True
        reasons.append("model_meets_thresholds")

    decision = {
        "should_retrain": should_retrain,
        "should_rebalance": should_rebalance,
        "should_skip": should_skip,
        "should_validate_existing": should_validate_existing,
        "reasons": reasons,
        "supervisor_health": combined_health,
        "current_f1": last_known_f1,
        "l4_count": l4_count,
    }

    print(f"  Retrain: {should_retrain} | Rebalance: {should_rebalance} | Skip: {should_skip}")
    print(f"  Reasons: {reasons}")

    kb.log_event("policy", "policy_decision", decision)

    # Increment L4 counter if we're looping back
    new_l4_count = l4_count + 1 if should_rebalance else l4_count

    return {
        **state,
        "policy_decision": decision,
        "current_model": current_model,
        "should_retrain": should_retrain,
        "should_rebalance": should_rebalance,
        "should_skip": should_skip,
        "should_validate_existing": should_validate_existing,
        "l4_count": new_l4_count,
    }
