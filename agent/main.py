#!/usr/bin/env python3
"""
main.py — CLI entry point for the Multi-Agent Fraud Detection System.

Usage:
    python main.py                          # use default dataset
    python main.py --data /path/to/data.csv # use custom CSV

MAPE-K Pipeline Flow:
    Ingestion → Drift → Balance → Supervisor → Policy → Strategy
              → Training → Evaluation → Simulation → Deployment
              → Knowledge → END
"""
import sys
from pathlib import Path

# Ensure the agent package root is on sys.path
_AGENT_ROOT = Path(__file__).resolve().parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

# Parse args before any other imports that might depend on sys.path
import argparse
import json

from graph import compile_and_run
from config import KNOWLEDGE_LOG_PATH, OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Fraud Detection System (MAPE-K + LangGraph)"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to the input CSV file (default: 9k-dataset/data.csv)"
    )
    args = parser.parse_args()

    final_state = compile_and_run(args.data)

    # Print summary
    print("\n" + "=" * 60)
    print(" RUN SUMMARY")
    print("=" * 60)
    print(f"  Drift detected       : {final_state.get('drift_detected')}")
    print(f"  Balance action       : {_safe(final_state.get('balance_report', {})).get('action', 'N/A')}")
    print(f"  Supervisor decision  : {final_state.get('supervisor_decision')}")
    print(f"  Policy retrain       : {final_state.get('should_retrain')}")
    print(f"  Eval metrics         : {final_state.get('eval_metrics')}")
    print(f"  Simulation passed    : {final_state.get('simulation_passed')}")
    print(f"  Model promoted       : {final_state.get('promoted')}")
    print(f"  Feedback loops       : L1={final_state.get('l1_count', 0)}, "
          f"L2={final_state.get('l2_count', 0)}, L3={final_state.get('l3_count', 0)}, "
          f"L4={final_state.get('l4_count', 0)}, L5={final_state.get('l5_count', 0)}")
    print(f"  Knowledge log        : {KNOWLEDGE_LOG_PATH}")

    # Save final summary
    summary = {
        "drift_detected": final_state.get("drift_detected"),
        "balance_report": _safe(final_state.get("balance_report")),
        "supervisor_decision": final_state.get("supervisor_decision"),
        "policy_decision": _safe(final_state.get("policy_decision")),
        "eval_metrics": final_state.get("eval_metrics"),
        "simulation_passed": final_state.get("simulation_passed"),
        "deployment_decision": _safe(final_state.get("deployment_decision")),
        "promoted": final_state.get("promoted"),
        "feedback_loops": {
            "L1_eval_to_balance": final_state.get("l1_count", 0),
            "L2_eval_to_strategy": final_state.get("l2_count", 0),
            "L3_training_self_loop": final_state.get("l3_count", 0),
            "L4_supervisor_policy": final_state.get("l4_count", 0),
            "L5_eval_to_kb": final_state.get("l5_count", 0),
        },
    }
    summary_path = OUTPUT_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  Summary saved        : {summary_path}")


def _safe(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)


if __name__ == "__main__":
    main()
