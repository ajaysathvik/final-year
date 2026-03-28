#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

echo "Running agentic fraud workflow..."
"$PYTHON_BIN" agent-new/main.py
echo
echo "Execution trace: $ROOT_DIR/agent-new/output/execution_trace.jsonl"
echo "Run report: $ROOT_DIR/agent-new/output/run_report.json"
