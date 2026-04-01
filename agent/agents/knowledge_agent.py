"""
Knowledge Agent — Knowledge layer of MAPE-K.
L5: Logs ALL events from every phase to the knowledge base (comprehensive).
Final pipeline node — also decides whether to trigger KB→Drift closed loop.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from config import KNOWLEDGE_LOG_PATH


def knowledge_agent(state: dict) -> dict:
    """
    L5 — Comprehensive logging of every pipeline phase to the knowledge layer.
    Increments kb_loop_count so graph.py route_after_knowledge can gate re-cycles.
    """
    print("\n" + "=" * 60)
    print(" [ KNOWLEDGE AGENT ] L5 — logging full run to knowledge layer...")
    print("=" * 60)

    kb = state.get("knowledge_base")
    timestamp = datetime.now(timezone.utc).isoformat()
    l5_count = state.get("l5_count", 0)

    # Pre-evaluate KB trigger to accurately log full_run_summary
    should_loop = kb.should_retrigger_drift() if kb else False

    # ── Build comprehensive full-run summary (L5 logs everything) ──
    entry = {
        "timestamp": timestamp,
        "event_type": "full_run_summary",
        "l5_cycle_count": l5_count,
        # ── Monitor ─────────────────────────────────────────────
        "drift_detected": state.get("drift_detected"),
        "drift_features": [
            col for col, v in state.get("drift_report", {}).items()
            if v.get("drifted")
        ],
        # ── Balance ─────────────────────────────────────────────
        "balance_report": _sanitize(state.get("balance_report")),
        # ── Supervisor ──────────────────────────────────────────
        "supervisor_health": _sanitize(state.get("supervisor_health")),
        "supervisor_decision": state.get("supervisor_decision"),
        # ── Policy ──────────────────────────────────────────────
        "policy_decision": _sanitize(state.get("policy_decision")),
        # ── Strategy ────────────────────────────────────────────
        "strategy_plan": _sanitize(state.get("strategy_plan")),
        # ── Training ────────────────────────────────────────────
        "training_metrics": _sanitize(state.get("training_metrics")),
        "candidate_model_path": state.get("candidate_model_path"),
        # ── Evaluation ──────────────────────────────────────────
        "eval_metrics": _sanitize(state.get("eval_metrics")),
        "eval_passed": state.get("eval_passed"),
        # ── Simulation ──────────────────────────────────────────
        "simulation_results": _sanitize(state.get("simulation_results")),
        "simulation_passed": state.get("simulation_passed"),
        # ── Deployment ──────────────────────────────────────────
        "deployment_decision": _sanitize(state.get("deployment_decision")),
        "promoted": state.get("promoted"),
        "promoted_model_path": state.get("promoted_model_path"),
        # ── Feedback loop counters ───────────────────────────────
        "l1_count": state.get("l1_count", 0),
        "l2_count": state.get("l2_count", 0),
        "l3_count": state.get("l3_count", 0),
        "l4_count": state.get("l4_count", 0),
        "l5_count": l5_count,
    }

    # Write to JSONL
    with open(KNOWLEDGE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")

    print(f"  ✅ Logged full_run_summary to {KNOWLEDGE_LOG_PATH}")
    print(f"  Feedback loops — L1={entry['l1_count']} L2={entry['l2_count']} "
          f"L3={entry['l3_count']} L4={entry['l4_count']} L5={entry['l5_count']}")

    if state.get("promoted"):
        print(f"  🏆 Model promoted: {state.get('promoted_model_path')}")
    else:
        print("  ℹ️  Model not promoted.")

    # Log to KB in-memory store as well
    if kb:
        kb.log_event("knowledge", "full_run_summary", entry)

    # ── KB→Drift decision ─────────────────────────────────────
    if should_loop:
        print(f"  🔄 KB→Drift closed loop triggered (L5 iteration {l5_count + 1})")
    else:
        print("  ✅ KB: no re-trigger needed — pipeline complete.")

    log_list = list(state.get("knowledge_log", []))
    log_list.append(entry)

    return {
        **state,
        "knowledge_log": log_list,
        "l5_count": l5_count + (1 if should_loop else 0),
        "kb_loop_count": state.get("kb_loop_count", 0), # retain backward compat
    }


def _sanitize(obj):
    """Ensure obj is JSON-serializable."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)
