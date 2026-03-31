"""
Deployment Agent — Execute phase of MAPE-K.
Decides whether to promote the candidate model based on
simulation results and policy approval.
Maps to N_Deploy in Agentic_Architecture.drawio.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone

from config import MODELS_DIR


def deployment_agent(state: dict) -> dict:
    """
    Promote the candidate model if both simulation and policy approve.
    Otherwise, mark as not promoted.
    """
    print("\n" + "=" * 60)
    print(" [ DEPLOYMENT AGENT ] Deciding model promotion...")
    print("=" * 60)

    kb = state["knowledge_base"]
    simulation_passed = state.get("simulation_passed", False)
    policy_decision = state.get("policy_decision", {})
    candidate_path = state.get("candidate_model_path")
    should_retrain = policy_decision.get("should_retrain", False)

    # Promote only if simulation passed AND policy approved retraining
    promote = simulation_passed and should_retrain and candidate_path is not None

    reasons = []
    if not simulation_passed:
        reasons.append("simulation_failed")
    if not should_retrain:
        reasons.append("policy_did_not_request_retrain")
    if candidate_path is None:
        reasons.append("no_candidate_model")
    if not reasons:
        reasons.append("all_checks_passed")

    deployment_decision = {
        "promoted": promote,
        "reasons": reasons,
        "simulation_passed": simulation_passed,
        "policy_retrain": should_retrain,
        "candidate_path": candidate_path,
    }

    if promote:
        promoted_path = str(MODELS_DIR / "promoted_model.joblib")
        shutil.copy2(candidate_path, promoted_path)
        deployment_decision["promoted_model_path"] = promoted_path
        print(f"  🏆 Model PROMOTED → {promoted_path}")
    else:
        print(f"  ℹ️  Model NOT promoted. Reasons: {reasons}")

    kb.log_event("deployment", "deployment_decision", deployment_decision)

    return {
        **state,
        "promoted": promote,
        "promoted_model_path": deployment_decision.get("promoted_model_path", ""),
        "deployment_decision": deployment_decision,
    }
