"""
Reporting helpers for per-run artifacts and cross-run comparison plots.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from config import KNOWLEDGE_LOG_PATH, PLOTS_DIR, RUNS_DIR


def generate_run_reports(summary: dict[str, Any]) -> dict[str, str]:
    """
    Persist a dedicated artifact bundle for the current run and regenerate
    comparison plots across all historical runs recorded in the knowledge log.
    """
    run_id = _resolve_run_id(summary)
    summary = {
        **summary,
        "run_id": run_id,
        "timestamp": _resolve_timestamp(summary),
    }

    run_dir = RUNS_DIR / run_id
    run_plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_plots_dir.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_plot_readme(PLOTS_DIR / "README.txt")

    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    _write_json_if_present(run_dir / "evaluation_metrics.json", summary.get("eval_metrics"))
    _write_json_if_present(run_dir / "training_metrics.json", summary.get("training_metrics"))
    _write_json_if_present(run_dir / "simulation_results.json", summary.get("simulation_results"))

    scalar_snapshot = _build_scalar_snapshot(summary)
    _write_scalar_csv(run_dir / "scalar_metrics.csv", scalar_snapshot)

    per_run_paths = {}
    eval_plot = run_plots_dir / "evaluation_metrics_bar.png"
    if _plot_single_run_eval(summary, eval_plot):
        per_run_paths["evaluation_plot"] = str(eval_plot)

    robustness_plot = run_plots_dir / "robustness_curve.png"
    if _plot_single_run_robustness(summary, robustness_plot):
        per_run_paths["robustness_plot"] = str(robustness_plot)

    historical_runs = _load_historical_runs()
    historical_runs = _merge_current_run(historical_runs, summary)
    historical_runs.sort(key=lambda item: (_resolve_timestamp(item), item.get("run_id", "")))

    comparison_csv = PLOTS_DIR / "run_comparison_metrics.csv"
    _write_comparison_csv(comparison_csv, historical_runs)

    comparison_paths = {}
    eval_cmp = PLOTS_DIR / "run_comparison_eval_metrics.png"
    if _plot_eval_comparison(historical_runs, eval_cmp):
        comparison_paths["eval_comparison_plot"] = str(eval_cmp)

    fpr_cmp = PLOTS_DIR / "run_comparison_fpr.png"
    if _plot_fpr_comparison(historical_runs, fpr_cmp):
        comparison_paths["fpr_comparison_plot"] = str(fpr_cmp)

    robust_cmp = PLOTS_DIR / "run_comparison_robustness_metrics.png"
    if _plot_robustness_comparison(historical_runs, robust_cmp):
        comparison_paths["robustness_comparison_plot"] = str(robust_cmp)

    attack_cmp = PLOTS_DIR / "run_comparison_attack_curves.png"
    if _plot_attack_curve_comparison(historical_runs, attack_cmp):
        comparison_paths["attack_curve_plot"] = str(attack_cmp)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "plots_dir": str(PLOTS_DIR),
        "comparison_csv": str(comparison_csv),
        **per_run_paths,
        **comparison_paths,
    }


def _write_individual_run_bundle(summary: dict[str, Any]) -> dict[str, str]:
    run_id = _resolve_run_id(summary)
    run_plot_dir = PLOTS_DIR / run_id
    run_plot_dir.mkdir(parents=True, exist_ok=True)
    _write_individual_plot_readme(run_plot_dir / "README.txt", run_id)

    normalized = {
        **summary,
        "run_id": run_id,
        "timestamp": _resolve_timestamp(summary),
    }

    (run_plot_dir / "run_summary.json").write_text(
        json.dumps(normalized, indent=2, default=str),
        encoding="utf-8",
    )
    _write_json_if_present(run_plot_dir / "evaluation_metrics.json", normalized.get("eval_metrics"))
    _write_json_if_present(run_plot_dir / "training_metrics.json", normalized.get("training_metrics"))
    _write_json_if_present(run_plot_dir / "simulation_results.json", normalized.get("simulation_results"))
    _write_scalar_csv(run_plot_dir / "scalar_metrics.csv", _build_scalar_snapshot(normalized))

    path_info = {
        "individual_run_dir": str(run_plot_dir),
    }

    eval_plot = run_plot_dir / "evaluation_metrics_bar.png"
    if _plot_single_run_eval(normalized, eval_plot) or _copy_existing_run_plot(run_id, eval_plot.name, eval_plot):
        path_info["evaluation_plot"] = str(eval_plot)

    robustness_plot = run_plot_dir / "robustness_curve.png"
    if _plot_single_run_robustness(normalized, robustness_plot) or _copy_existing_run_plot(run_id, robustness_plot.name, robustness_plot):
        path_info["robustness_plot"] = str(robustness_plot)

    return path_info


def _copy_existing_run_plot(run_id: str, filename: str, output_path: Path) -> bool:
    source_path = RUNS_DIR / run_id / "plots" / filename
    if not source_path.exists():
        return False
    shutil.copy2(source_path, output_path)
    return True


def _load_historical_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    if not KNOWLEDGE_LOG_PATH.exists():
        return runs

    with open(KNOWLEDGE_LOG_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(entry)

            if entry.get("event_type") != "full_run_summary":
                continue

            payload = dict(entry.get("data", {}))
            if not payload:
                payload = {
                    k: v for k, v in entry.items()
                    if k not in {"agent", "event_type"}
                }
            payload.setdefault("timestamp", entry.get("timestamp"))
            payload["run_id"] = _resolve_run_id(payload)
            runs.append(payload)

    unique_runs: dict[str, dict[str, Any]] = {}
    for run in runs:
        unique_runs[run["run_id"]] = run
    if unique_runs:
        return list(unique_runs.values())
    return _reconstruct_runs_from_events(entries)


def _merge_current_run(runs: list[dict[str, Any]], current: dict[str, Any]) -> list[dict[str, Any]]:
    merged = {run["run_id"]: run for run in runs if run.get("run_id")}
    merged[current["run_id"]] = current
    return list(merged.values())


def _resolve_run_id(summary: dict[str, Any]) -> str:
    for value in (
        summary.get("run_id"),
        summary.get("timestamp"),
        (summary.get("training_metrics") or {}).get("timestamp"),
    ):
        if not value:
            continue
        text = str(value)
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 14:
            return f"run_{digits[:14]}"
    return f"run_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"


def _resolve_timestamp(summary: dict[str, Any]) -> str:
    for value in (
        summary.get("timestamp"),
        (summary.get("training_metrics") or {}).get("timestamp"),
        summary.get("run_id"),
    ):
        if not value:
            continue
        return str(value)
    return datetime.utcnow().isoformat()


def _reconstruct_runs_from_events(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build approximate per-run summaries from the sequential event log when
    explicit full_run_summary entries are not available.
    """
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    for entry in entries:
        agent = entry.get("agent")
        event_type = entry.get("event_type")
        data = entry.get("data", {}) or {}
        timestamp = entry.get("timestamp")

        if not current:
            current = {"timestamp": timestamp}

        if agent == "policy" and event_type == "policy_decision":
            current["policy_decision"] = data
        elif agent == "training" and event_type == "model_trained":
            current["training_metrics"] = data
            current["timestamp"] = timestamp
        elif agent == "evaluation" and event_type == "evaluation_completed":
            current["eval_metrics"] = data
            current.setdefault("timestamp", timestamp)
        elif agent == "simulation" and event_type == "simulation_completed":
            current["simulation_results"] = data
        elif agent == "deployment" and event_type == "deployment_decision":
            current["deployment_decision"] = data
            current["promoted"] = data.get("promoted")
            current["simulation_passed"] = data.get("simulation_passed")
            current["timestamp"] = timestamp or current.get("timestamp")
            current["run_id"] = _resolve_run_id(current)
            runs.append(current)
            current = {}

    unique_runs: dict[str, dict[str, Any]] = {}
    for run in runs:
        unique_runs[run["run_id"]] = run
    return list(unique_runs.values())


def _build_scalar_snapshot(summary: dict[str, Any]) -> dict[str, float]:
    scalar_metrics: dict[str, float] = {}
    for prefix, section in (
        ("eval", summary.get("eval_metrics")),
        ("training", summary.get("training_metrics")),
        ("simulation", summary.get("simulation_results")),
    ):
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if key in {"threshold"}:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scalar_metrics[f"{prefix}.{key}"] = float(value)
    return scalar_metrics


def _write_json_if_present(path: Path, payload: Any) -> None:
    if payload is None:
        return
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_scalar_csv(path: Path, values: dict[str, float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in sorted(values.items()):
            writer.writerow([metric, value])


def _write_comparison_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    rows = []
    for run in runs:
        row: dict[str, Any] = {
            "run_id": run.get("run_id"),
            "timestamp": run.get("timestamp"),
        }
        row.update(_build_scalar_snapshot(run))
        eval_metrics = run.get("eval_metrics") or {}
        noise_stress = (run.get("simulation_results") or {}).get("noise_stress_test", {})
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[f"eval.{key}"] = value
        for key, value in noise_stress.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[f"simulation.{key}"] = value
        rows.append(row)

    fieldnames = ["run_id", "timestamp"]
    extra_fields = sorted({
        key for row in rows for key in row.keys()
        if key not in {"run_id", "timestamp"}
    })
    fieldnames.extend(extra_fields)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_single_run_eval(summary: dict[str, Any], output_path: Path) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    metrics = summary.get("eval_metrics") or {}
    ordered_keys = ["f1", "precision", "recall", "roc_auc", "fpr"]
    values = [metrics.get(key) for key in ordered_keys]
    if not any(value is not None for value in values):
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f77b4", "#4c78a8", "#54a24b", "#72b7b2", "#e45756"]
    ax.bar(ordered_keys, [float(v or 0.0) for v in values], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Evaluation Metrics ({summary.get('run_id', 'run')})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_single_run_robustness(summary: dict[str, Any], output_path: Path) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    sim = summary.get("simulation_results") or {}
    attack_scores = sim.get("noise_stress_test") or {}
    if not attack_scores:
        return False

    eps_vals = []
    scores = []
    for key, value in sorted(attack_scores.items(), key=lambda item: float(item[0].split("_", 1)[1])):
        try:
            eps_vals.append(float(key.split("_", 1)[1]))
            scores.append(float(value))
        except (IndexError, ValueError, TypeError):
            continue

    if not eps_vals:
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eps_vals, scores, marker="o", linewidth=2, color="#1f77b4")
    ax.set_xlabel("Attack Epsilon")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Robustness Curve ({summary.get('run_id', 'run')})")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_eval_comparison(runs: list[dict[str, Any]], output_path: Path) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    if len(runs) < 2:
        return False

    run_labels = [run["run_id"] for run in runs]
    metrics = ["f1", "precision", "recall", "roc_auc"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric, color in zip(metrics, ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]):
        values = [(run.get("eval_metrics") or {}).get(metric) for run in runs]
        if any(value is not None for value in values):
            ax.plot(run_labels, [float("nan") if value is None else float(value) for value in values], marker="o", linewidth=2, label=metric.upper(), color=color)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Run")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics Across Runs")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=4)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_fpr_comparison(runs: list[dict[str, Any]], output_path: Path) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    if len(runs) < 2:
        return False
    values = [(run.get("eval_metrics") or {}).get("fpr") for run in runs]
    if not any(value is not None for value in values):
        return False

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        [run["run_id"] for run in runs],
        [float(value or 0.0) for value in values],
        marker="o",
        linewidth=2,
        color="#e45756",
    )
    ax.set_xlabel("Run")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("False Positive Rate Across Runs")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_robustness_comparison(runs: list[dict[str, Any]], output_path: Path) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    if len(runs) < 2:
        return False

    metric_specs = [
        ("training.train_f1", "Train F1", "#1f77b4"),
        ("simulation.bootstrap_mean_f1", "Bootstrap Mean F1", "#ff7f0e"),
        ("simulation.worst_noise_f1", "Worst Attacked F1", "#2ca02c"),
        ("simulation.robustness_score", "Robustness Score", "#9467bd"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    run_labels = [run["run_id"] for run in runs]
    for metric_key, label, color in metric_specs:
        values = [_build_scalar_snapshot(run).get(metric_key) for run in runs]
        if any(value is not None for value in values):
            ax.plot(run_labels, [float("nan") if value is None else float(value) for value in values], marker="o", linewidth=2, label=label, color=color)
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Run")
    ax.set_ylabel("Score")
    ax.set_title("Training and Robustness Metrics Across Runs")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_attack_curve_comparison(runs: list[dict[str, Any]], output_path: Path) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False
    eligible = [run for run in runs if (run.get("simulation_results") or {}).get("noise_stress_test")]
    if len(eligible) < 2:
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    for run in eligible:
        attack_scores = (run.get("simulation_results") or {}).get("noise_stress_test", {})
        eps_vals = []
        scores = []
        for key, value in sorted(attack_scores.items(), key=lambda item: float(item[0].split("_", 1)[1])):
            try:
                eps_vals.append(float(key.split("_", 1)[1]))
                scores.append(float(value))
            except (IndexError, ValueError, TypeError):
                continue
        if eps_vals:
            ax.plot(eps_vals, scores, marker="o", linewidth=1.8, label=run["run_id"])

    if not ax.lines:
        plt.close(fig)
        return False

    ax.set_xlabel("Attack Epsilon")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("FGSM-Style Robustness Curves Across Runs")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        return None


def _write_plot_readme(path: Path) -> None:
    path.write_text(
        (
            "Plots are generated automatically when matplotlib is installed in the active Python "
            "environment.\n"
            "Cross-run comparison files are stored in this folder.\n"
            "Per-run artifacts are stored in subfolders named run_<YYYYMMDDHHMMSS>/.\n"
        ),
        encoding="utf-8",
    )


def _write_individual_plot_readme(path: Path, run_id: str) -> None:
    path.write_text(
        (
            f"Per-run plot bundle for {run_id}.\n"
            "This folder contains only the plots and scalar/JSON data for this run.\n"
        ),
        encoding="utf-8",
    )
