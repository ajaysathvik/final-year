"""
Supervisor Agent — Monitor phase of MAPE-K.
Performs anomaly checks and validation on pipeline health.
Sits between Balance Agent and Policy Agent.
L4 feedback loop: Supervisor ↔ Policy.
"""
from __future__ import annotations

from config import SUPERVISOR_HEALTH_MIN


def supervisor_agent(state: dict) -> dict:
    """
    Assess overall pipeline health:
    - Data quality (drift, balance quality)
    - Model readiness
    - Decide: proceed to policy | request rebalance | escalate
    """
    print("\n" + "=" * 60)
    print(" [ SUPERVISOR ] Anomaly check & validation...")
    print("=" * 60)

    kb = state["knowledge_base"]
    drift_detected = state.get("drift_detected", False)
    balance_report = state.get("balance_report", {})
    l4_count = state.get("l4_count", 0)

    # ── Health scoring ──────────────────────────────────────────
    data_quality = 1.0
    if drift_detected:
        data_quality -= 0.3
    if balance_report.get("action") == "failed":
        data_quality -= 0.4
    if balance_report.get("action") == "skipped" and balance_report.get("reason") == "too_few_fraud":
        data_quality -= 0.2

    # Bug-5 fix: detect post-augmentation imbalance from training agent
    post_aug_imbalanced = state.get("post_augmentation_imbalanced", False)
    if post_aug_imbalanced:
        data_quality -= 0.3
        print("  ⚠️ Post-augmentation class imbalance detected — penalizing data quality")

    balance_quality = 1.0
    if balance_report.get("action") == "balanced":
        balance_quality = 0.8  # balanced is good but synthetic data adds uncertainty
    elif balance_report.get("action") == "failed":
        balance_quality = 0.2

    # Check if policy previously requested rebalance (L4 feedback)
    policy_decision = state.get("policy_decision", {})
    if policy_decision.get("should_rebalance") and l4_count > 0:
        data_quality -= 0.1  # penalize if we're looping

    combined_health = round((0.6 * data_quality + 0.4 * balance_quality), 4)

    # ── Decision ────────────────────────────────────────────────
    if combined_health >= SUPERVISOR_HEALTH_MIN:
        decision = "proceed"
    elif balance_report.get("action") == "failed":
        decision = "escalate"
    else:
        decision = "proceed"  # proceed anyway so pipeline doesn't stall

    health = {
        "data_quality": round(data_quality, 4),
        "balance_quality": round(balance_quality, 4),
        "combined_health": combined_health,
        "threshold": SUPERVISOR_HEALTH_MIN,
    }

    print(f"  Health: data={health['data_quality']}, balance={health['balance_quality']}, "
          f"combined={combined_health} (min={SUPERVISOR_HEALTH_MIN})")
    print(f"  Decision: {decision}")

    kb.log_event("supervisor", "health_check", {
        **health,
        "decision": decision,
        "l4_count": l4_count,
    })

    return {
        **state,
        "supervisor_health": health,
        "supervisor_decision": decision,
    }
