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
    eval_metrics = state.get("eval_metrics", {})
    l4_count = state.get("l4_count", 0)
    current_model = state.get("current_model")

    # ── KB context: read eval trend (KB→Policy communication) ────
    eval_history = kb.get_history("evaluation_history", limit=3)
    f1_trend = [e.get("data", {}).get("f1", None) for e in eval_history if e.get("data", {}).get("f1") is not None]
    trend_degrading = len(f1_trend) >= 2 and f1_trend[-1] < f1_trend[0]
    if f1_trend:
        print(f"  📖 KB context: F1 trend over last {len(f1_trend)} runs = {f1_trend} "
              f"({'⬇ degrading' if trend_degrading else '➡ stable/improving'})")

    combined_health = supervisor_health.get("combined_health", 0.5)
    f1 = eval_metrics.get("f1", 0)

    # ── Determine action ────────────────────────────────────────
    should_retrain = False
    should_rebalance = False
    should_skip = False
    reasons = []

    # If supervisor escalated, consider rebalancing (L4 loop)
    if supervisor_decision == "escalate" and l4_count < MAX_L4_ITERATIONS:
        should_rebalance = True
        reasons.append("supervisor_escalated")
    elif current_model is None:
        should_retrain = True
        reasons.append("no_existing_model")
    elif f1 < F1_THRESHOLD:
        should_retrain = True
        reasons.append(f"f1={f1}<{F1_THRESHOLD}")
    elif state.get("drift_detected", False):
        should_retrain = True
        reasons.append("drift_detected")
    else:
        should_skip = True
        reasons.append("model_meets_thresholds")

    # Always retrain if no model exists
    if not should_retrain and not should_rebalance and not should_skip:
        should_retrain = True
        reasons.append("default_retrain")

    decision = {
        "should_retrain": should_retrain,
        "should_rebalance": should_rebalance,
        "should_skip": should_skip,
        "reasons": reasons,
        "supervisor_health": combined_health,
        "current_f1": f1,
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
        "should_retrain": should_retrain,
        "should_rebalance": should_rebalance,
        "should_skip": should_skip,
        "l4_count": new_l4_count,
    }
