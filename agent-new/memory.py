from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import MEMORY_DIR, MEMORY_PATH


class AgentMemory:
    def __init__(self, path: Path = MEMORY_PATH) -> None:
        self.path = path
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(self._empty())

    def _ts(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _empty(self) -> dict[str, Any]:
        return {
            "created_at": self._ts(),
            "runs": [],
            "snapshots": [],
            "decisions": [],
            "artifacts": {},
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            payload = self._empty()
            self._write(payload)
            return payload
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = self._empty()
            self._write(payload)
            return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def add_snapshot(self, snapshot: dict[str, Any]) -> None:
        payload = self._read()
        snapshot["recorded_at"] = self._ts()
        payload["snapshots"].append(snapshot)
        payload["updated_at"] = self._ts()
        self._write(payload)

    def add_decision(self, agent: str, decision: dict[str, Any]) -> None:
        payload = self._read()
        payload["decisions"].append(
            {
                "agent": agent,
                "decision": decision,
                "recorded_at": self._ts(),
            }
        )
        payload["updated_at"] = self._ts()
        self._write(payload)

    def set_artifact(self, name: str, path: str, metadata: dict[str, Any] | None = None) -> None:
        payload = self._read()
        payload["artifacts"][name] = {
            "path": path,
            "metadata": metadata or {},
            "updated_at": self._ts(),
        }
        payload["updated_at"] = self._ts()
        self._write(payload)

    def record_run(self, report: dict[str, Any]) -> None:
        payload = self._read()
        report["recorded_at"] = self._ts()
        payload["runs"].append(report)
        payload["updated_at"] = self._ts()
        self._write(payload)
