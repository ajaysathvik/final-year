"""
Knowledge Base — Central shared-state hub for MAPE-K.
All agents read/write through this class for inter-agent communication.
Stores: logs, models, policies, drift history, metrics.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import KNOWLEDGE_LOG_PATH, METRICS_DIR


class KnowledgeBase:
    """Thread-safe-ish in-memory knowledge store that also persists to JSONL."""

    def __init__(self) -> None:
        self.store: dict[str, list[dict[str, Any]]] = {
            "drift_history": [],
            "evaluation_history": [],
            "policy_decisions": [],
            "strategy_plans": [],
            "training_records": [],
            "simulation_records": [],
            "supervisor_records": [],
            "balance_records": [],
            "deployment_records": [],
            "knowledge_records": [],
        }
        self._log_path = KNOWLEDGE_LOG_PATH
        self._replay_log()  # load history from previous runs

    def _replay_log(self) -> None:
        """Replay the persisted JSONL log into the in-memory store on startup."""
        category_map = {
            "drift": "drift_history",
            "evaluation": "evaluation_history",
            "policy": "policy_decisions",
            "strategy": "strategy_plans",
            "training": "training_records",
            "simulation": "simulation_records",
            "supervisor": "supervisor_records",
            "balance": "balance_records",
            "deployment": "deployment_records",
            "knowledge": "knowledge_records",
        }
        if not self._log_path.exists():
            return
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        agent = entry.get("agent")
                        if not agent and entry.get("event_type") == "full_run_summary":
                            agent = "knowledge"
                            entry = {
                                "timestamp": entry.get("timestamp"),
                                "agent": "knowledge",
                                "event_type": entry.get("event_type"),
                                "data": {
                                    k: v for k, v in entry.items()
                                    if k not in {"timestamp", "agent", "event_type"}
                                },
                            }
                        if not agent:
                            continue
                        cat = category_map.get(agent, "drift_history")
                        self.store[cat].append(entry)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    def log_event(self, agent: str, event_type: str, data: dict[str, Any]) -> None:
        """Append an event to the JSONL log and in-memory store."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent,
            "event_type": event_type,
            "data": _sanitize(data),
        }
        # Persist to JSONL
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")

        # In-memory store
        category = f"{agent}_records" if f"{agent}_records" in self.store else "drift_history"
        # Map agent names to store categories
        category_map = {
            "drift": "drift_history",
            "evaluation": "evaluation_history",
            "policy": "policy_decisions",
            "strategy": "strategy_plans",
            "training": "training_records",
            "simulation": "simulation_records",
            "supervisor": "supervisor_records",
            "balance": "balance_records",
            "deployment": "deployment_records",
            "knowledge": "knowledge_records",
        }
        cat = category_map.get(agent, "drift_history")
        self.store[cat].append(entry)

    def get_latest(self, category: str) -> dict[str, Any] | None:
        """Get the most recent entry from a category."""
        records = self.store.get(category, [])
        return records[-1] if records else None

    def get_history(self, category: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent entries from a category."""
        return self.store.get(category, [])[-limit:]

    def get_latest_evaluation_metrics(self) -> dict[str, Any]:
        """Return the latest held-out evaluation metrics, if any."""
        latest_eval = self.get_latest("evaluation_history")
        if not latest_eval:
            return {}
        return latest_eval.get("data", {}) or {}

    def get_summary(self) -> dict[str, int]:
        """Return count of entries per category."""
        return {k: len(v) for k, v in self.store.items()}

    def save_metrics(self, name: str, metrics: dict[str, Any]) -> Path:
        """Save a metrics dict to a JSON file."""
        path = METRICS_DIR / f"{name}.json"
        path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        return path

    def should_retrigger_drift(self, robustness_threshold: float | None = None) -> bool:
        """
        KB→Drift closed-loop decision.
        Returns True if a new drift-detection cycle is warranted based on:
          1. Latest simulation robustness score fell below threshold, OR
          2. Evaluation F1 has been degrading across the last two KB entries.
        """
        from config import ROBUSTNESS_THRESHOLD, F1_THRESHOLD
        rob_thresh = robustness_threshold if robustness_threshold is not None else ROBUSTNESS_THRESHOLD

        # Check robustness
        latest_sim = self.get_latest("simulation_records")
        if latest_sim:
            score = latest_sim.get("data", {}).get("robustness_score", 1.0)
            if score < rob_thresh:
                return True

        # Check F1 degradation trend
        eval_history = self.get_history("evaluation_history", limit=2)
        f1_vals = [e.get("data", {}).get("f1") for e in eval_history
                   if e.get("data", {}).get("f1") is not None]
        if len(f1_vals) == 2 and f1_vals[-1] < f1_vals[0] and f1_vals[-1] < F1_THRESHOLD:
            return True

        return False


def _sanitize(obj: Any) -> Any:
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
