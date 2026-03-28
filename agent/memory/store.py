from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import MEMORY_DIR, MEMORY_PATH


class AgentMemory:
    def __init__(self, path: Path = MEMORY_PATH):
        self.path = path
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(self._empty_payload())

    def _empty_payload(self) -> dict[str, Any]:
        return {
            "created_at": self._timestamp(),
            "tool_registry": [],
            "latest_fraud_signals": [],
            "llm_assessments": [],
            "dataset_snapshots": [],
            "pipeline_state": {},
            "pipeline_runs": [],
            "runs": [],
            "artifacts": [],
        }

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            payload = self._empty_payload()
            self._write(payload)
            return payload

        try:
            with self.path.open("r", encoding="utf-8") as handle:
                raw = handle.read().strip()
        except OSError:
            payload = self._empty_payload()
            self._write(payload)
            return payload

        if not raw:
            payload = self._empty_payload()
            self._write(payload)
            return payload

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            broken_path = self.path.with_suffix(f"{self.path.suffix}.broken")
            try:
                self.path.replace(broken_path)
            except OSError:
                pass
            payload = self._empty_payload()
            self._write(payload)
            return payload

        if not isinstance(payload, dict):
            payload = self._empty_payload()
            self._write(payload)
            return payload

        payload.setdefault("tool_registry", [])
        payload.setdefault("latest_fraud_signals", [])
        payload.setdefault("llm_assessments", [])
        payload.setdefault("dataset_snapshots", [])
        payload.setdefault("pipeline_state", {})
        payload.setdefault("pipeline_runs", [])
        payload.setdefault("runs", [])
        payload.setdefault("artifacts", [])
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")

    def register_tools(self, tools: list[dict[str, Any]]) -> None:
        payload = self._read()
        payload["tool_registry"] = tools
        payload["updated_at"] = self._timestamp()
        self._write(payload)

    def add_dataset_snapshot(self, snapshot: dict[str, Any]) -> None:
        payload = self._read()
        snapshot["recorded_at"] = self._timestamp()
        payload["dataset_snapshots"].append(snapshot)
        payload["updated_at"] = self._timestamp()
        self._write(payload)

    def set_latest_fraud_signals(self, signals: list[dict[str, Any]]) -> None:
        payload = self._read()
        payload["latest_fraud_signals"] = signals
        payload["updated_at"] = self._timestamp()
        self._write(payload)

    def add_llm_assessment(self, assessment: dict[str, Any]) -> None:
        payload = self._read()
        assessment["recorded_at"] = self._timestamp()
        payload["llm_assessments"].append(assessment)
        payload["updated_at"] = self._timestamp()
        self._write(payload)

    def update_pipeline_state(self, key: str, value: dict[str, Any]) -> None:
        payload = self._read()
        payload.setdefault("pipeline_state", {})
        payload["pipeline_state"][key] = {
            **value,
            "updated_at": self._timestamp(),
        }
        payload["updated_at"] = self._timestamp()
        self._write(payload)

    def add_pipeline_run_summary(self, summary: dict[str, Any]) -> None:
        payload = self._read()
        summary["recorded_at"] = self._timestamp()
        payload.setdefault("pipeline_runs", [])
        payload["pipeline_runs"].append(summary)
        payload["updated_at"] = self._timestamp()
        self._write(payload)

    def record_run(self, run: dict[str, Any]) -> None:
        payload = self._read()
        run["recorded_at"] = self._timestamp()
        payload["runs"].append(run)
        payload["updated_at"] = self._timestamp()
        self._write(payload)

    def upsert_artifact(self, name: str, path: str, metadata: dict[str, Any] | None = None) -> None:
        payload = self._read()
        artifacts = [artifact for artifact in payload["artifacts"] if artifact["name"] != name]
        artifacts.append(
            {
                "name": name,
                "path": path,
                "metadata": metadata or {},
                "updated_at": self._timestamp(),
            }
        )
        payload["artifacts"] = artifacts
        payload["updated_at"] = self._timestamp()
        self._write(payload)

    def get_artifact(self, name: str) -> dict[str, Any] | None:
        payload = self._read()
        for artifact in payload.get("artifacts", []):
            if artifact.get("name") == name:
                return artifact
        return None

    def get_pipeline_state(self, key: str) -> dict[str, Any] | None:
        payload = self._read()
        return payload.get("pipeline_state", {}).get(key)
