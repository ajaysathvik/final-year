"""
finfraudai — Qwen-controlled fraud pipeline backed by train/test CSVs.

Architecture:
- Controller: Qwen on Ollama
- Worker: one simple fraud model (RandomForest)

Flow:
1. Load train/test CSVs
2. Train the active worker model
3. Monitor sequential shadow windows from the test CSV
4. Compute PSI, precision, recall, F1, and economic value
5. Ask Qwen whether to keep monitoring or retrain/redeploy the worker
6. Validate any retrained worker on a held-out gold set before replacing the active worker
"""
from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agent.memory.store import AgentMemory

warnings.filterwarnings("ignore")
console = Console()

AUDIT_LOG_PATH = Path(__file__).resolve().parent / "audit_log.jsonl"
TRAIN_CSV_PATH = Path(__file__).resolve().parents[1] / "Data labeling" / "outputs" / "train_3k.csv"
TEST_CSV_PATH = Path(__file__).resolve().parents[1] / "Data labeling" / "outputs" / "test_3k.csv"
TARGET_COL = "annotation.is_fraud"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen3.5:0.8b"
PSI_REVIEW_THRESHOLD = 0.10
PSI_RETRAIN_THRESHOLD = 0.25
CONCEPT_PRECISION_DROP = 0.05
CONCEPT_RECALL_STABILITY = 0.02
RECENT_WEIGHT = 4.0
FALSE_NEGATIVE_COST = 500.0
FALSE_POSITIVE_COST = 25.0
GOLD_FRACTION = 0.40
SHADOW_WINDOWS = 24
PROMOTION_F1_DELTA = 0.01
PROMOTION_ECONOMIC_VALUE = 0.0


@dataclass
class WorkerBundle:
    pipeline: Pipeline
    gold_metrics: dict[str, float]
    training_frame: pd.DataFrame
    version: int
    training_reason: str


@dataclass
class PipelineContext:
    train_df: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None
    gold_df: pd.DataFrame | None = None
    shadow_windows: list[pd.DataFrame] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    worker: WorkerBundle | None = None
    shadow_history: list[pd.DataFrame] = field(default_factory=list)
    hourly_log: list[dict[str, Any]] = field(default_factory=list)
    memory: AgentMemory = field(default_factory=AgentMemory)


ctx = PipelineContext()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace({
        "": np.nan,
        "unknown": np.nan,
        "Unknown": np.nan,
        "none": np.nan,
        "None": np.nan,
        "null": np.nan,
        "NULL": np.nan,
    })
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def infer_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in df.columns:
        if col == TARGET_COL:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.85:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def split_gold_and_shadow(test_df: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    gold_size = max(1, int(len(test_df) * GOLD_FRACTION))
    gold_df = test_df.iloc[:gold_size].reset_index(drop=True)
    shadow_df = test_df.iloc[gold_size:].reset_index(drop=True)
    windows = [chunk.reset_index(drop=True) for chunk in np.array_split(shadow_df, SHADOW_WINDOWS) if len(chunk) > 0]
    return gold_df, windows


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), ctx.numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ctx.categorical_cols),
        ]
    )


def parse_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=[TARGET_COL]).copy()
    for col in ctx.numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        median = features[col].median()
        features[col] = features[col].fillna(0.0 if pd.isna(median) else median)
    for col in ctx.categorical_cols:
        features[col] = features[col].fillna("missing").astype(str)
    return features


def fit_worker_model(train_df: pd.DataFrame, sample_weight: np.ndarray | None, version: int, reason: str) -> WorkerBundle:
    X_train = parse_feature_frame(train_df)
    y_train = train_df[TARGET_COL].astype(int)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        class_weight="balanced_subsample",
        random_state=42 + version,
    )
    pipeline = Pipeline([("prep", build_preprocessor()), ("model", model)])
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    try:
        pipeline.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        pipeline.fit(X_train, y_train)

    assert ctx.gold_df is not None
    gold_metrics, _, _ = evaluate_bundle_metrics(pipeline, ctx.gold_df)
    return WorkerBundle(
        pipeline=pipeline,
        gold_metrics=gold_metrics,
        training_frame=train_df.reset_index(drop=True),
        version=version,
        training_reason=reason,
    )


def evaluate_bundle_metrics(pipeline: Pipeline, df: pd.DataFrame) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    X = parse_feature_frame(df)
    y = df[TARGET_COL].values
    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else 0.0,
        "pred_positive_rate": float(preds.mean()),
    }
    return metrics, preds, probs


def safe_log(value: float) -> float:
    return math.log(max(value, 1e-8))


def numeric_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    expected_hist, _ = np.histogram(expected, bins=quantiles)
    actual_hist, _ = np.histogram(actual, bins=quantiles)
    expected_pct = expected_hist / max(expected_hist.sum(), 1) + 1e-8
    actual_pct = actual_hist / max(actual_hist.sum(), 1) + 1e-8
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    expected_freq = expected.fillna("missing").astype(str).value_counts(normalize=True)
    actual_freq = actual.fillna("missing").astype(str).value_counts(normalize=True)
    cats = expected_freq.index.union(actual_freq.index)
    psi = 0.0
    for cat in cats:
        exp = float(expected_freq.get(cat, 0.0)) + 1e-8
        act = float(actual_freq.get(cat, 0.0)) + 1e-8
        psi += (act - exp) * safe_log(act / exp)
    return float(psi)


def dataset_psi(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> tuple[float, list[tuple[str, float]]]:
    feature_scores: list[tuple[str, float]] = []
    for col in ctx.numeric_cols:
        feature_scores.append((col, numeric_psi(reference_df[col], current_df[col])))
    for col in ctx.categorical_cols:
        feature_scores.append((col, categorical_psi(reference_df[col], current_df[col])))
    overall = float(np.mean([score for _, score in feature_scores])) if feature_scores else 0.0
    feature_scores.sort(key=lambda item: item[1], reverse=True)
    return overall, feature_scores


def retraining_frame() -> tuple[pd.DataFrame, np.ndarray]:
    assert ctx.train_df is not None
    frames = [ctx.train_df]
    weights = [np.ones(len(ctx.train_df), dtype=float)]
    if ctx.shadow_history:
        history = pd.concat(ctx.shadow_history, ignore_index=True)
        frames.append(history)
        history_weight = np.ones(len(history), dtype=float)
        if len(ctx.shadow_history) >= 4:
            recent_rows = sum(len(frame) for frame in ctx.shadow_history[-4:])
            history_weight[-recent_rows:] = RECENT_WEIGHT
        else:
            history_weight[:] = RECENT_WEIGHT
        weights.append(history_weight)
    return pd.concat(frames, ignore_index=True), np.concatenate(weights)


def economic_value(current_preds: np.ndarray, candidate_preds: np.ndarray, truth: np.ndarray) -> float:
    cur_tn, cur_fp, cur_fn, cur_tp = confusion_matrix(truth, current_preds, labels=[0, 1]).ravel()
    cand_tn, cand_fp, cand_fn, cand_tp = confusion_matrix(truth, candidate_preds, labels=[0, 1]).ravel()
    fn_reduction = cur_fn - cand_fn
    fp_increase = cand_fp - cur_fp
    return float(fn_reduction * FALSE_NEGATIVE_COST - max(fp_increase, 0) * FALSE_POSITIVE_COST)


def build_controller_prompt(snapshot: dict[str, Any]) -> str:
    return (
        "You are Qwen, the controller agent for a fraud detection system.\n"
        "You do not classify transactions directly. You supervise one simpler worker model.\n"
        "Your job is to decide whether the worker should keep running, retrain on recent labeled data, "
        "or be replaced by a newly validated worker model.\n\n"
        "Decision policy:\n"
        "- PSI < 0.10: usually keep monitoring.\n"
        "- 0.10 <= PSI <= 0.25: moderate drift, retraining may be useful.\n"
        "- PSI > 0.25: major drift, retraining is strongly preferred.\n"
        "- Concept drift exists when precision drops materially while recall is stable.\n"
        "- Replace the active worker only if the candidate passes gold-set validation and has non-negative economic value.\n"
        "- Be conservative. Do not redeploy on noise.\n\n"
        f"Snapshot:\n{json.dumps(snapshot, indent=2)}\n\n"
        "Return exactly one JSON object with these keys:\n"
        "{\n"
        '  "controller_action": "monitor|retrain_worker|replace_worker",\n'
        '  "reason": "short reason",\n'
        '  "confidence": 0.0,\n'
        '  "feature_focus": [],\n'
        '  "window_weight_hours": 4,\n'
        '  "requires_validation": true\n'
        "}\n"
        "Do not return markdown. Do not add extra text."
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates: list[str] = []
    if text:
        candidates.append(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def ask_qwen_controller(snapshot: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": build_controller_prompt(snapshot),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_predict": 400,
        },
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        return {
            "status": "unavailable",
            "controller_action": "monitor",
            "reason": f"Ollama unavailable: {exc}",
            "confidence": 0.0,
            "feature_focus": [],
            "window_weight_hours": 4,
            "requires_validation": True,
            "raw_response": "",
        }

    try:
        parsed_body = json.loads(body)
    except json.JSONDecodeError:
        parsed_body = {"response": body}
    raw_response = str(parsed_body.get("response", "")).strip()
    decision = extract_json_object(raw_response)
    if decision is None:
        return {
            "status": "malformed",
            "controller_action": "monitor",
            "reason": "Ollama response could not be parsed as decision JSON.",
            "confidence": 0.0,
            "feature_focus": [],
            "window_weight_hours": 4,
            "requires_validation": True,
            "raw_response": raw_response,
        }
    decision["status"] = "ok"
    decision["raw_response"] = raw_response
    return decision


def heuristic_controller(snapshot: dict[str, Any]) -> dict[str, Any]:
    psi_value = snapshot["psi_value"]
    precision_drop = snapshot["precision_drop"]
    recall_delta = snapshot["recall_delta"]
    if precision_drop > CONCEPT_PRECISION_DROP and recall_delta <= CONCEPT_RECALL_STABILITY:
        action = "retrain_worker"
        reason = "concept drift: precision dropped while recall stayed stable"
    elif psi_value > PSI_RETRAIN_THRESHOLD:
        action = "retrain_worker"
        reason = "major PSI drift"
    elif psi_value >= PSI_REVIEW_THRESHOLD:
        action = "retrain_worker"
        reason = "moderate PSI drift"
    else:
        action = "monitor"
        reason = "metrics stable"
    return {
        "status": "heuristic",
        "controller_action": action,
        "reason": reason,
        "confidence": 0.6,
        "feature_focus": [snapshot["top_drift_feature"]] if snapshot.get("top_drift_feature") else [],
        "window_weight_hours": 4,
        "requires_validation": True,
        "raw_response": "",
    }


def train_initial_worker() -> None:
    assert ctx.train_df is not None
    initial = fit_worker_model(ctx.train_df, None, version=1, reason="initial_train")
    ctx.worker = initial


def validate_candidate(candidate: WorkerBundle, current_metrics: dict[str, float], current_preds: np.ndarray, window_df: pd.DataFrame) -> tuple[bool, str, dict[str, float], float]:
    cand_metrics, cand_preds, _ = evaluate_bundle_metrics(candidate.pipeline, window_df)
    econ = economic_value(current_preds, cand_preds, window_df[TARGET_COL].values)
    gold_f1_gain = candidate.gold_metrics["f1"] - ctx.worker.gold_metrics["f1"] if ctx.worker else 0.0
    if candidate.gold_metrics["f1"] + 1e-9 < (ctx.worker.gold_metrics["f1"] if ctx.worker else 0.0):
        return False, f"gold-set F1 regressed ({gold_f1_gain:.3f})", cand_metrics, econ
    if cand_metrics["f1"] + PROMOTION_F1_DELTA < current_metrics["f1"]:
        return False, f"window F1 worse ({cand_metrics['f1']:.3f} < {current_metrics['f1']:.3f})", cand_metrics, econ
    if econ < PROMOTION_ECONOMIC_VALUE:
        return False, f"economic value negative ({econ:.1f})", cand_metrics, econ
    return True, f"validated: gold_gain={gold_f1_gain:.3f}, econ={econ:.1f}", cand_metrics, econ


def controller_cycle(hour: int, window_df: pd.DataFrame) -> dict[str, Any]:
    assert ctx.worker is not None
    current_metrics, current_preds, _ = evaluate_bundle_metrics(ctx.worker.pipeline, window_df)
    reference_features = parse_feature_frame(ctx.worker.training_frame)
    current_features = parse_feature_frame(window_df)
    psi_value, feature_scores = dataset_psi(reference_features, current_features)
    top_feature, top_feature_psi = feature_scores[0] if feature_scores else ("n/a", 0.0)
    precision_drop = ctx.worker.gold_metrics["precision"] - current_metrics["precision"]
    recall_delta = abs(ctx.worker.gold_metrics["recall"] - current_metrics["recall"])

    snapshot = {
        "hour": hour,
        "worker_version": ctx.worker.version,
        "worker_training_reason": ctx.worker.training_reason,
        "psi_value": round(psi_value, 6),
        "top_drift_feature": top_feature,
        "top_feature_psi": round(top_feature_psi, 6),
        "current_metrics": {k: round(v, 4) for k, v in current_metrics.items()},
        "gold_metrics": {k: round(v, 4) for k, v in ctx.worker.gold_metrics.items()},
        "precision_drop": round(precision_drop, 4),
        "recall_delta": round(recall_delta, 4),
        "shadow_history_windows": len(ctx.shadow_history),
        "retrain_thresholds": {
            "psi_review": PSI_REVIEW_THRESHOLD,
            "psi_retrain": PSI_RETRAIN_THRESHOLD,
            "precision_drop": CONCEPT_PRECISION_DROP,
            "recall_stability": CONCEPT_RECALL_STABILITY,
        },
    }
    controller_decision = ask_qwen_controller(snapshot)
    if controller_decision.get("status") != "ok":
        controller_decision = heuristic_controller(snapshot)

    deployed = False
    candidate_metrics: dict[str, float] | None = None
    validation_reason = "no retraining requested"
    candidate_economic_value = 0.0

    if controller_decision["controller_action"] in {"retrain_worker", "replace_worker"}:
        combined_train, sample_weight = retraining_frame()
        next_version = ctx.worker.version + 1
        candidate = fit_worker_model(
            combined_train,
            sample_weight,
            version=next_version,
            reason=controller_decision["reason"],
        )
        approved, validation_reason, candidate_metrics, candidate_economic_value = validate_candidate(
            candidate,
            current_metrics,
            current_preds,
            window_df,
        )
        if approved:
            ctx.worker = candidate
            deployed = True

    record = {
        "cycle_id": hour,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "controller_model": OLLAMA_MODEL,
        "controller_status": controller_decision.get("status"),
        "controller_action": controller_decision["controller_action"],
        "controller_reason": controller_decision["reason"],
        "controller_confidence": round(float(controller_decision.get("confidence", 0.0)), 4),
        "worker_version": ctx.worker.version if ctx.worker else None,
        "worker_training_reason": ctx.worker.training_reason if ctx.worker else None,
        "psi_value": round(psi_value, 6),
        "top_drift_feature": top_feature,
        "top_feature_psi": round(top_feature_psi, 6),
        "worker_precision": round(current_metrics["precision"], 4),
        "worker_recall": round(current_metrics["recall"], 4),
        "worker_f1": round(current_metrics["f1"], 4),
        "worker_pr_auc": round(current_metrics["pr_auc"], 4),
        "precision_drop": round(precision_drop, 4),
        "recall_delta": round(recall_delta, 4),
        "candidate_f1": round(candidate_metrics["f1"], 4) if candidate_metrics else None,
        "candidate_precision": round(candidate_metrics["precision"], 4) if candidate_metrics else None,
        "candidate_recall": round(candidate_metrics["recall"], 4) if candidate_metrics else None,
        "candidate_economic_value": round(candidate_economic_value, 2),
        "deployed": deployed,
        "validation_reason": validation_reason,
    }
    ctx.hourly_log.append(record)
    ctx.shadow_history.append(window_df.reset_index(drop=True))
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    ctx.memory.add_llm_assessment({
        "stage": "qwen_controller",
        "hour": hour,
        "model": OLLAMA_MODEL,
        "snapshot": snapshot,
        "decision": controller_decision,
        "deployed": deployed,
        "validation_reason": validation_reason,
    })
    return record


def print_workflow() -> None:
    console.print(Panel.fit(
        "Qwen Controller -> Simple Worker Model\n"
        "Monitor: worker metrics and PSI on each shadow window\n"
        "Analyze: Qwen reads drift signals and decides the next action\n"
        "Synthesize: retrain the worker on train CSV plus recent shadow windows\n"
        "Validate: compare the retrained worker on gold and current shadow window\n"
        "Deploy: replace the worker only if validation passes",
        title="True Agentic Workflow",
        border_style="cyan",
    ))


def print_summary() -> None:
    table = Table(title="Qwen Controller Log", show_lines=True, header_style="bold magenta")
    for col in ["Hour", "PSI", "Top Drift", "Action", "F1", "Cand F1", "Deploy", "Worker Ver", "Controller"]:
        table.add_column(col, justify="center")
    for rec in ctx.hourly_log:
        table.add_row(
            str(rec["cycle_id"]),
            f"{rec['psi_value']:.3f}",
            str(rec["top_drift_feature"]),
            str(rec["controller_action"]),
            f"{rec['worker_f1']:.3f}",
            f"{rec['candidate_f1']:.3f}" if rec["candidate_f1"] is not None else "—",
            "YES" if rec["deployed"] else "—",
            str(rec["worker_version"]),
            str(rec["controller_status"]),
        )
    console.print(table)
    console.print(f"\nAudit log: [green]{AUDIT_LOG_PATH}[/green]")


def main() -> None:
    console.rule("[bold cyan]finfraudai — Qwen Controller + Worker Pipeline[/bold cyan]")
    print_workflow()

    ctx.memory.register_tools([
        {
            "name": "qwen_controller",
            "description": "Qwen on Ollama supervises the worker fraud model and decides retraining or replacement.",
            "inputs": ["shadow metrics", "PSI", "gold metrics", "top drift features"],
            "outputs": ["controller decision JSON"],
        }
    ])

    ctx.train_df = load_dataset(TRAIN_CSV_PATH)
    ctx.test_df = load_dataset(TEST_CSV_PATH)
    ctx.numeric_cols, ctx.categorical_cols = infer_feature_types(ctx.train_df)
    ctx.gold_df, ctx.shadow_windows = split_gold_and_shadow(ctx.test_df)

    console.print("\n[bold]Phase 1:[/bold] Train the simple worker model from CSVs\n")
    console.print(
        f"  Train rows={len(ctx.train_df)} | Gold rows={len(ctx.gold_df)} | "
        f"Shadow rows={sum(len(window) for window in ctx.shadow_windows)} | "
        f"Numeric={len(ctx.numeric_cols)} | Categorical={len(ctx.categorical_cols)}"
    )
    train_initial_worker()
    assert ctx.worker is not None
    console.print(
        f"  Controller: [cyan]{OLLAMA_MODEL}[/cyan]\n"
        f"  Worker: [green]RandomForest v{ctx.worker.version}[/green] | "
        f"Gold F1={ctx.worker.gold_metrics['f1']:.3f} | Gold PR-AUC={ctx.worker.gold_metrics['pr_auc']:.3f}"
    )

    console.rule("[bold cyan]Phase 2: Qwen-Controlled Shadow Monitoring[/bold cyan]")
    for hour, window_df in enumerate(ctx.shadow_windows, start=1):
        rec = controller_cycle(hour, window_df)
        flag = "🚨" if rec["deployed"] else "  "
        console.print(
            f"  {flag} Window {hour:>2}/{len(ctx.shadow_windows)} | PSI={rec['psi_value']:.3f} | "
            f"TopDrift={rec['top_drift_feature']} | Action={rec['controller_action']} | "
            f"F1={rec['worker_f1']:.3f} | CandF1={rec['candidate_f1'] if rec['candidate_f1'] is not None else '—'} | "
            f"Worker=v{rec['worker_version']} | Controller={rec['controller_status']}"
        )
        if rec["deployed"]:
            console.print(Panel(
                f"[bold red]WORKER UPDATED[/bold red]\n"
                f"Controller reason: {rec['controller_reason']}\n"
                f"Validation: {rec['validation_reason']}\n"
                f"Top drifting feature: {rec['top_drift_feature']} ({rec['top_feature_psi']:.3f})",
                title="Deployment",
                border_style="bold red",
                expand=False,
            ))

    console.rule("[bold cyan]Dashboard Summary[/bold cyan]")
    print_summary()


if __name__ == "__main__":
    main()
