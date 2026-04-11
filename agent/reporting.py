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

    _write_json_if_present(
        run_dir / "evaluation_metrics.json", summary.get("eval_metrics")
    )
    _write_json_if_present(
        run_dir / "training_metrics.json", summary.get("training_metrics")
    )
    _write_json_if_present(
        run_dir / "simulation_results.json", summary.get("simulation_results")
    )

    scalar_snapshot = _build_scalar_snapshot(summary)
    _write_scalar_csv(run_dir / "scalar_metrics.csv", scalar_snapshot)

    per_run_paths = {}
    eval_plot = run_plots_dir / "evaluation_metrics_bar.png"
    if _plot_single_run_eval(summary, eval_plot):
        per_run_paths["evaluation_plot"] = str(eval_plot)

    robustness_plot = run_plots_dir / "robustness_curve.png"
    if _plot_single_run_robustness(summary, robustness_plot):
        per_run_paths["robustness_plot"] = str(robustness_plot)

    lr_plot = run_plots_dir / "learning_rate_curve.png"
    if _plot_single_run_learning_rate(summary, lr_plot):
        per_run_paths["learning_rate_plot"] = str(lr_plot)

    historical_runs = _load_historical_runs()
    historical_runs = _merge_current_run(historical_runs, summary)
    historical_runs.sort(
        key=lambda item: (_resolve_timestamp(item), item.get("run_id", ""))
    )

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

    robust_criteria = PLOTS_DIR / "run_comparison_robustness_criteria.png"
    if _plot_robustness_criteria(historical_runs, robust_criteria):
        comparison_paths["robustness_criteria_plot"] = str(robust_criteria)

    attack_cmp = PLOTS_DIR / "run_comparison_attack_curves.png"
    if _plot_attack_curve_comparison(historical_runs, attack_cmp):
        comparison_paths["attack_curve_plot"] = str(attack_cmp)

    lr_cmp = PLOTS_DIR / "run_comparison_learning_rate.png"
    if _plot_learning_rate_comparison(historical_runs, lr_cmp):
        comparison_paths["learning_rate_comparison_plot"] = str(lr_cmp)

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
    _write_json_if_present(
        run_plot_dir / "evaluation_metrics.json", normalized.get("eval_metrics")
    )
    _write_json_if_present(
        run_plot_dir / "training_metrics.json", normalized.get("training_metrics")
    )
    _write_json_if_present(
        run_plot_dir / "simulation_results.json", normalized.get("simulation_results")
    )
    _write_scalar_csv(
        run_plot_dir / "scalar_metrics.csv", _build_scalar_snapshot(normalized)
    )

    path_info = {
        "individual_run_dir": str(run_plot_dir),
    }

    eval_plot = run_plot_dir / "evaluation_metrics_bar.png"
    if _plot_single_run_eval(normalized, eval_plot) or _copy_existing_run_plot(
        run_id, eval_plot.name, eval_plot
    ):
        path_info["evaluation_plot"] = str(eval_plot)

    robustness_plot = run_plot_dir / "robustness_curve.png"
    if _plot_single_run_robustness(
        normalized, robustness_plot
    ) or _copy_existing_run_plot(run_id, robustness_plot.name, robustness_plot):
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
                    k: v for k, v in entry.items() if k not in {"agent", "event_type"}
                }
            payload.setdefault("timestamp", entry.get("timestamp"))
            payload["run_id"] = _resolve_run_id(payload)
            runs.append(payload)

    unique_runs: dict[str, dict[str, Any]] = {}
    for run in runs:
        unique_runs[run["run_id"]] = run
    if unique_runs:
        return _assign_descriptive_run_ids(list(unique_runs.values()))
    return _reconstruct_runs_from_events(entries)


def _merge_current_run(
    runs: list[dict[str, Any]], current: dict[str, Any]
) -> list[dict[str, Any]]:
    current_timestamp = _resolve_timestamp(current)
    current_base_label = _derive_run_label(current)

    merged: dict[str, dict[str, Any]] = {}
    for run in runs:
        run_id = run.get("run_id")
        if not run_id:
            continue

        same_timestamp = _resolve_timestamp(run) == current_timestamp
        same_condition = _derive_run_label(run) == current_base_label
        if same_timestamp and same_condition:
            continue

        merged[run_id] = run

    merged[current["run_id"]] = current
    return list(merged.values())


def _resolve_run_id(summary: dict[str, Any]) -> str:
    """Return a descriptive run_id for the summary.

    Existing descriptive IDs are preserved. Legacy numeric IDs such as
    run_1 are upgraded to condition-based labels like normal_data_run.
    """
    existing = summary.get("run_id")
    if existing and not _is_legacy_numeric_run_id(str(existing)):
        return str(existing)
    return _ensure_unique_run_id(_derive_run_label(summary))


def _resolve_timestamp(summary: dict[str, Any]) -> str:
    for value in (
        summary.get("timestamp"),
        (summary.get("training_metrics") or {}).get("timestamp"),
    ):
        if not value:
            continue
        return str(value)
    return datetime.utcnow().isoformat()


def _reconstruct_runs_from_events(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build approximate per-run summaries from the sequential event log when
    explicit full_run_summary entries are not available.

    Run IDs are assigned descriptive labels based on run conditions, with
    numeric suffixes added only when the same condition appears multiple times.
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
        elif agent == "strategy" and event_type == "strategy_plan":
            current["strategy_plan"] = data
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
            runs.append(current)
            current = {}

    return _assign_descriptive_run_ids(runs)


def _is_legacy_numeric_run_id(run_id: str) -> bool:
    return run_id.startswith("run_") and run_id[4:].isdigit()


def _derive_run_label(summary: dict[str, Any]) -> str:
    run_context = summary.get("run_context") or {}
    training_metrics = summary.get("training_metrics") or {}
    policy_decision = summary.get("policy_decision") or {}

    if run_context.get("generate_random_drift_data") is True:
        return "random_drift_run"
    if run_context.get("use_scraped_drift_data") is False:
        return "normal_data_run"
    if (
        policy_decision.get("should_validate_existing")
        and summary.get("training_metrics") is None
    ):
        return "validation_run"
    if (
        summary.get("drift_detected")
        or training_metrics.get("used_drift_remediation")
        or float(training_metrics.get("drift_rows_added", 0) or 0) > 0
    ):
        return "scraped_drift_run"
    return "normal_data_run"


def _ensure_unique_run_id(base_id: str) -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = {p.name for p in RUNS_DIR.iterdir() if p.is_dir()}
    if base_id not in existing_ids:
        return base_id

    suffix = 2
    while f"{base_id}_{suffix}" in existing_ids:
        suffix += 1
    return f"{base_id}_{suffix}"


def _assign_descriptive_run_ids(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not runs:
        return []

    counts: dict[str, int] = {}
    labeled_runs: list[dict[str, Any]] = []

    for run in sorted(
        runs, key=lambda item: (_resolve_timestamp(item), str(item.get("run_id", "")))
    ):
        normalized = dict(run)
        base_id = _derive_run_label(normalized)
        counts[base_id] = counts.get(base_id, 0) + 1
        normalized["run_id"] = (
            base_id if counts[base_id] == 1 else f"{base_id}_{counts[base_id]}"
        )
        labeled_runs.append(normalized)

    return labeled_runs


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
        noise_stress = (run.get("simulation_results") or {}).get(
            "noise_stress_test", {}
        )
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[f"eval.{key}"] = value
        for key, value in noise_stress.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[f"simulation.{key}"] = value
        bootstrap_std = row.get("simulation.bootstrap_std")
        if isinstance(bootstrap_std, (int, float)):
            row["simulation.bootstrap_variance"] = float(bootstrap_std) ** 2

        worst_eps = _resolve_worst_case_epsilon(run)
        if worst_eps is not None:
            row["simulation.worst_noise_epsilon"] = worst_eps
        rows.append(row)

    fieldnames = ["run_id", "timestamp"]
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in {"run_id", "timestamp"}
        }
    )
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
    ax.bar(
        ordered_keys,
        [float(v or 0.0) for v in values],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_title(f"Evaluation Metrics - {summary.get('run_id', 'run')}")
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
    for key, value in sorted(
        attack_scores.items(), key=lambda item: float(item[0].split("_", 1)[1])
    ):
        try:
            eps_vals.append(float(key.split("_", 1)[1]))
            scores.append(float(value))
        except (IndexError, ValueError, TypeError):
            continue

    if not eps_vals:
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eps_vals, scores, marker="o", linewidth=2, color="#1f77b4")
    ax.set_xlabel("Attack Epsilon", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Robustness Curve - {summary.get('run_id', 'run')}")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_eval_comparison(runs: list[dict[str, Any]], output_path: Path) -> bool:
    """Grouped bar chart with zoomed y-axis to highlight small metric changes."""
    try:
        import numpy as np  # noqa: PLC0415
    except ModuleNotFoundError:
        np = None  # type: ignore[assignment]

    plt = _load_pyplot()
    if plt is None:
        return False
    if len(runs) < 2:
        return False

    run_labels = [_display_run_label(run) for run in runs]
    metrics = ["f1", "precision", "recall", "roc_auc"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    # Collect all values per metric
    metric_values: dict[str, list[float]] = {}
    for metric in metrics:
        values = [(run.get("eval_metrics") or {}).get(metric) for run in runs]
        if any(v is not None for v in values):
            metric_values[metric] = [float(v) if v is not None else 0.0 for v in values]

    if not metric_values:
        return False

    n_runs = len(run_labels)
    n_metrics = len(metric_values)
    bar_width = 0.7 / n_metrics

    fig, ax = plt.subplots(figsize=(max(10, n_runs * 2), 6))

    if np is not None:
        x = np.arange(n_runs)
    else:
        x = list(range(n_runs))

    all_vals = [v for vals in metric_values.values() for v in vals if v > 0]
    if all_vals:
        y_min = max(0, min(all_vals) - 0.03)
        y_max = min(1.0, max(all_vals) + 0.02)
    else:
        y_min, y_max = 0, 1.0

    for idx, (metric, vals) in enumerate(metric_values.items()):
        offset = (idx - (n_metrics - 1) / 2) * bar_width
        if np is not None:
            positions = x + offset
        else:
            positions = [xi + offset for xi in x]
        bars = ax.bar(
            positions,
            vals,
            width=bar_width * 0.88,
            label=metric.upper(),
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.4,
        )
        # Annotate each bar with its value
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (y_max - y_min) * 0.01,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    if np is not None:
        ax.set_xticks(x)
    else:
        ax.set_xticks(list(range(n_runs)))
    ax.set_xticklabels(run_labels, rotation=30, ha="right")
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.15)
    ax.set_xlabel("Experiment Configuration", fontsize=11)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_title("Evaluation Metrics Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
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
        [_display_run_label(run) for run in runs],
        [float(value or 0.0) for value in values],
        marker="o",
        linewidth=2,
        color="#e45756",
    )
    ax.set_xlabel("Experiment Configuration", fontsize=11)
    ax.set_ylabel("False Positive Rate", fontsize=11)
    ax.set_title("False Positive Rate Comparison")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_robustness_comparison(runs: list[dict[str, Any]], output_path: Path) -> bool:
    """
    Robustness comparison with *persistence lines*: solid segments denote runs
    where a new model was trained; dashed segments denote runs where the
    previous model was re-used (training_metrics is None).

    For Train F1, the last successful value is carried forward on re-used runs
    so the line remains continuous.
    """
    from matplotlib.lines import Line2D  # noqa: PLC0415

    plt = _load_pyplot()
    if plt is None:
        return False
    if len(runs) < 2:
        return False

    metric_specs = [
        ("training.train_f1", "Train F1", "#1f77b4"),
        ("simulation.bootstrap_mean_f1", "Bootstrap Mean F1", "#ff7f0e"),
        ("simulation.worst_noise_f1", "Worst Attacked F1", "#2ca02c"),
        ("simulation.robustness_score", "Average Robustness Score", "#9467bd"),
    ]

    # Determine which runs re-used the previous model (no new training).
    model_reused = [(run.get("training_metrics") is None) for run in runs]

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    run_labels = [_display_run_label(run) for run in runs]
    x_indices = list(range(len(runs)))

    for metric_key, label, color in metric_specs:
        raw_values = [_build_scalar_snapshot(run).get(metric_key) for run in runs]

        # For Train F1, carry forward the last known value on re-used runs.
        if metric_key == "training.train_f1":
            filled_values: list[float | None] = []
            last_known: float | None = None
            for i, val in enumerate(raw_values):
                if val is not None:
                    last_known = float(val)
                    filled_values.append(last_known)
                elif model_reused[i] and last_known is not None:
                    filled_values.append(last_known)
                else:
                    filled_values.append(None)
            values = filled_values
        else:
            values = [float(v) if v is not None else None for v in raw_values]

        if not any(v is not None for v in values):
            continue

        # Draw segment-by-segment: solid for trained, dashed for re-used.
        for i in range(len(runs) - 1):
            y0, y1 = values[i], values[i + 1]
            if y0 is None or y1 is None:
                continue
            # A segment is "re-used" if the *destination* run re-used the model.
            is_reused_segment = model_reused[i + 1]
            ax.plot(
                [x_indices[i], x_indices[i + 1]],
                [y0, y1],
                linestyle="--" if is_reused_segment else "-",
                linewidth=2,
                color=color,
            )

        # Draw markers: filled circle for trained, open circle for re-used.
        for i, val in enumerate(values):
            if val is None:
                continue
            ax.plot(
                x_indices[i],
                val,
                marker="o",
                markersize=7,
                color=color,
                markerfacecolor="white" if model_reused[i] else color,
                markeredgewidth=1.8 if model_reused[i] else 1.2,
                markeredgecolor=color,
            )

        # Invisible full line just for the primary legend entry.
        ax.plot([], [], marker="o", linewidth=2, color=color, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xticks(x_indices)
    ax.set_xticklabels(run_labels)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Experiment Configuration", fontsize=11)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_title("Training and Robustness Metrics Comparison")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=30)

    # Primary legend (metric names).
    primary_legend = ax.legend(frameon=False, loc="lower left")
    ax.add_artist(primary_legend)

    # Secondary legend explaining line styles.
    style_handles = [
        Line2D(
            [0],
            [0],
            color="grey",
            linewidth=2,
            linestyle="-",
            marker="o",
            markersize=6,
            label="New Model Trained",
        ),
        Line2D(
            [0],
            [0],
            color="grey",
            linewidth=2,
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.8,
            markersize=6,
            label="Previous Model Re-used",
        ),
    ]
    ax.legend(
        handles=style_handles,
        frameon=True,
        fancybox=True,
        framealpha=0.85,
        edgecolor="#ccc",
        fontsize=8,
        loc="upper right",
    )
    # Re-add primary legend (adding second legend removes the first).
    ax.add_artist(primary_legend)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_robustness_criteria(runs: list[dict[str, Any]], output_path: Path) -> bool:
    plt = _load_pyplot()
    if plt is None:
        return False

    selected_runs = _select_robustness_criteria_runs(runs)

    robust_rows = []
    for run in selected_runs:
        sim = run.get("simulation_results") or {}
        training = run.get("training_metrics") or {}

        # Extract all F1 scores from the multi-epsilon FGSM attack curve
        attack_curve: dict[str, float] = sim.get("noise_stress_test") or {}
        curve_vals: list[float] = []
        for k, v in sorted(attack_curve.items(),
                            key=lambda kv: float(kv[0].split("_", 1)[1])):
            try:
                curve_vals.append(float(v))
            except (ValueError, TypeError):
                continue

        if curve_vals:
            import statistics  # noqa: PLC0415
            mean_rob = statistics.mean(curve_vals)
            var_rob = statistics.variance(curve_vals) if len(curve_vals) > 1 else 0.0
            worst_f1 = min(curve_vals)
            worst_eps = _resolve_worst_case_epsilon(run)
        else:
            # Fallback to stored scalar if curve unavailable
            tb_score = training.get("tabularbench_robustness_score") or sim.get("robustness_score")
            mean_rob = float(tb_score) if tb_score is not None else None
            var_rob = (float(sim["bootstrap_std"]) ** 2
                       if sim.get("bootstrap_std") is not None else None)
            worst_f1 = sim.get("worst_noise_f1")
            worst_eps = _resolve_worst_case_epsilon(run)

        if any(v is not None for v in (mean_rob, var_rob, worst_f1)):
            robust_rows.append(
                {
                    "run_id": _display_run_label(run),
                    "mean_robustness": float(mean_rob) if mean_rob is not None else None,
                    "variance": float(var_rob) if var_rob is not None else None,
                    "worst_f1": float(worst_f1) if worst_f1 is not None else None,
                    "worst_eps": worst_eps,
                    "n_epsilons": len(curve_vals),
                    "is_robust": bool(training.get("tabularbench_is_robust", sim.get("passed", True))),
                    "asr": float(training.get("tabularbench_asr", 0.0)),
                }
            )

    if len(robust_rows) < 2:
        return False

    run_labels = [row["run_id"] for row in robust_rows]
    x = list(range(len(run_labels)))

    # ── Light theme ───────────────────────────────────────────────
    BG    = "white"
    AX_BG = "#f8f9fa"
    SPINE = "#dee2e6"
    TEXT  = "#212529"
    BLUE  = "#3d7eba"
    RED   = "#e05c5c"
    GREEN = "#2d9f6e"

    metric_panels = [
        ("mean_robustness", "Mean Robustness Score\nacross Attack Epsilons (\u2191 Better)", BLUE,  True),
        ("variance",        "Variance of Robustness\nacross Attack Epsilons (\u2193 Better)", RED,   False),
        ("worst_f1",        "Worst-Case Attacked F1\n(Minimum across Epsilons, \u2191 Better)", GREEN, True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Robustness Criteria \u2014 Experimental Configuration Comparison",
        fontsize=14, fontweight="bold", y=1.03,
    )

    def _style_ax(ax):
        ax.set_facecolor(AX_BG)
        ax.tick_params(labelsize=9)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(SPINE)
        for sp in ["top", "right"]:
            ax.spines[sp].set_color("none")

    plotted = False
    for ax, (metric_key, title, color, higher_better) in zip(axes, metric_panels):
        values = [row[metric_key] for row in robust_rows]
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            ax.set_visible(False)
            continue

        _style_ax(ax)
        best_value = max(valid_values) if higher_better else min(valid_values)

        bar_colors = []
        for v in values:
            if v is None:
                bar_colors.append("#dee2e6")
            elif abs(v - best_value) < 1e-12:
                bar_colors.append(color)
            else:
                bar_colors.append("#adb5bd")

        heights = [float(v) if v is not None else 0.0 for v in values]
        bars = ax.bar(
            x, heights, color=bar_colors, edgecolor=SPINE,
            linewidth=0.6, width=0.55, zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=20, ha="right")
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_ylabel("Score" if metric_key != "variance" else "Variance", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color=SPINE, zorder=0)
        plotted = True

        if metric_key == "variance":
            top = max(valid_values) * 1.35 if max(valid_values) > 0 else 1.0
            ax.set_ylim(0, top)
            label_offset = top * 0.03
            label_fmt = "{:.2e}"
        elif metric_key == "robustness_score":
            ymin = max(0.0, min(valid_values) - 0.02)
            ymax = min(1.0, max(valid_values) + 0.04)
            ax.set_ylim(ymin, ymax)
            label_offset = (ymax - ymin) * 0.025
            label_fmt = "{:.4f}"
        else:
            ymin = max(0.0, min(valid_values) - 0.03)
            ymax = min(1.0, max(valid_values) + 0.06)
            if ymax <= ymin:
                ymax = min(1.0, ymin + 0.06)
            ax.set_ylim(ymin, ymax)
            label_offset = (ymax - ymin) * 0.025
            label_fmt = "{:.4f}"

        for idx, (bar, v) in enumerate(zip(bars, values)):
            if v is None:
                continue
            text_label = label_fmt.format(float(v))
            if metric_key == "worst_f1" and robust_rows[idx]["worst_eps"] is not None:
                text_label = f"{text_label}\n@ \u03b5={robust_rows[idx]['worst_eps']:.2f}"
            is_winner = abs(float(v) - best_value) < 1e-12
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,
                text_label + ("\n\u2605" if is_winner else ""),
                ha="center", va="bottom",
                fontsize=8,
                color=color if is_winner else TEXT,
                fontweight="bold" if is_winner else "normal",
            )

        if metric_key == "mean_robustness":
            for idx, row in enumerate(robust_rows):
                asr = row.get("asr", 0.0)
                n_eps = row.get("n_epsilons", 0)
                status_icon = "\u2714" if row.get("is_robust", True) else "\u2718"
                label_inner = (f"n={n_eps}\nASR {asr:.1f}%\n{status_icon}"
                               if n_eps > 0 else f"ASR {asr:.1f}%\n{status_icon}")
                ax.text(
                    x[idx], heights[idx] / 2 if heights[idx] > 0 else 0.01,
                    label_inner,
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=5,
                )

    if not plotted:
        plt.close(fig)
        return False

    from matplotlib.patches import Patch  # noqa: PLC0415
    legend_elements = [
        Patch(facecolor=BLUE,    edgecolor=SPINE, label="Best (Mean Robustness)"),
        Patch(facecolor=GREEN,   edgecolor=SPINE, label="Best (Worst-Case F1)"),
        Patch(facecolor=RED,     edgecolor=SPINE, label="Best (Variance)"),
        Patch(facecolor="#adb5bd", edgecolor=SPINE, label="Non-winner"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=4,
        edgecolor=SPINE,
        fontsize=8, bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return True


def _display_run_label(run: dict[str, Any]) -> str:
    run_id = str(run.get("run_id", ""))
    mapping = {
        "normal_data_run": "baseline",
        "scraped_drift_run": "baseline with drift",
        "random_drift_run": "baseline with drift",
        "validation_run": "validation",
    }
    for prefix, label in mapping.items():
        if run_id == prefix or run_id.startswith(f"{prefix}_"):
            return label
    return run_id or "Run"


def _select_robustness_criteria_runs(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Keep only the latest baseline and baseline-with-drift runs for the
    robustness criteria comparison plot.
    """
    latest_by_label: dict[str, dict[str, Any]] = {}
    sort_key = lambda run: (_resolve_timestamp(run), str(run.get("run_id", "")))

    for run in sorted(runs, key=sort_key):
        label = _display_run_label(run)
        if label not in {"baseline", "baseline with drift"}:
            continue
        latest_by_label[label] = run

    ordered = []
    for label in ("baseline", "baseline with drift"):
        run = latest_by_label.get(label)
        if run is not None:
            ordered.append(run)
    return ordered


def _plot_attack_curve_comparison(
    runs: list[dict[str, Any]], output_path: Path
) -> bool:
    """Grouped bar chart: x-axis = attack epsilon level, bars = runs."""
    try:
        import numpy as np  # noqa: PLC0415
    except ModuleNotFoundError:
        np = None  # type: ignore[assignment]

    plt = _load_pyplot()
    if plt is None:
        return False
    eligible = [
        run
        for run in runs
        if (run.get("simulation_results") or {}).get("noise_stress_test")
    ]
    if len(eligible) < 2:
        return False

    # Collect all epsilon levels and per-run scores
    all_eps: set[float] = set()
    run_data: list[tuple[str, dict[float, float]]] = []
    for run in eligible:
        attack_scores = (run.get("simulation_results") or {}).get(
            "noise_stress_test", {}
        )
        eps_map: dict[float, float] = {}
        for key, value in attack_scores.items():
            try:
                eps = float(key.split("_", 1)[1])
                eps_map[eps] = float(value)
                all_eps.add(eps)
            except (IndexError, ValueError, TypeError):
                continue
        if eps_map:
            run_data.append((_display_run_label(run), eps_map))

    if not run_data:
        plt.close("all")
        return False

    eps_levels = sorted(all_eps)
    n_runs = len(run_data)
    n_eps = len(eps_levels)
    bar_width = 0.7 / n_runs

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    fig, ax = plt.subplots(figsize=(max(8, n_eps * n_runs * 0.5 + 2), 5))

    if np is not None:
        x = np.arange(n_eps)
        offsets = [(i - (n_runs - 1) / 2) * bar_width for i in range(n_runs)]
        for idx, (run_id, eps_map) in enumerate(run_data):
            heights = [eps_map.get(eps, 0.0) for eps in eps_levels]
            ax.bar(
                x + offsets[idx],
                heights,
                width=bar_width * 0.9,
                label=run_id,
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x)
    else:
        # Fallback without numpy: simple side-by-side using plain floats
        offsets = [(i - (n_runs - 1) / 2) * bar_width for i in range(n_runs)]
        for idx, (run_id, eps_map) in enumerate(run_data):
            heights = [eps_map.get(eps, 0.0) for eps in eps_levels]
            xs = [j + offsets[idx] for j in range(n_eps)]
            ax.bar(
                xs,
                heights,
                width=bar_width * 0.9,
                label=run_id,
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(list(range(n_eps)))

    ax.set_xticklabels([str(e) for e in eps_levels])
    ax.set_xlabel("Attack Epsilon", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("FGSM-Style Robustness Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=8, ncol=min(n_runs, 5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _resolve_worst_case_epsilon(run: dict[str, Any]) -> float | None:
    attack_scores = (run.get("simulation_results") or {}).get("noise_stress_test") or {}
    worst_pair: tuple[float, float] | None = None
    for key, value in attack_scores.items():
        try:
            eps = float(key.split("_", 1)[1])
            score = float(value)
        except (IndexError, ValueError, TypeError):
            continue
        if worst_pair is None or score < worst_pair[1]:
            worst_pair = (eps, score)
    return None if worst_pair is None else worst_pair[0]


def _plot_single_run_learning_rate(summary: dict[str, Any], output_path: Path) -> bool:
    """
    Plot a per-run training progression chart using real F1 values from each
    L3 self-loop iteration.  Falls back gracefully when only one iteration ran.

    Primary axis  : Real Train-F1 per L3 iteration (measured, not interpolated)
    Secondary axis: n_estimators used at each iteration (bar)
    """
    plt = _load_pyplot()
    if plt is None:
        return False

    tm = summary.get("training_metrics") or {}
    f1_history: list[float] = tm.get("f1_history") or []
    train_f1 = tm.get("train_f1")

    # Fall back: if the history list wasn't persisted use the single final value.
    if not f1_history and train_f1 is not None:
        f1_history = [float(train_f1)]

    if not f1_history:
        return False

    n_est_final: int = int(tm.get("n_estimators", 0))
    model_type: str = str(tm.get("model_type", "Model"))
    l3_iterations = len(f1_history)  # 1 = no L3 loops, 2+ = self-loop ran

    # Reconstruct per-iteration n_estimators.  The final value already
    # includes 50 * (l3_iterations - 1) boosts; reverse-engineer the base.
    n_boosts = l3_iterations - 1
    base_n_est = n_est_final - (50 * n_boosts)
    iterations = list(range(l3_iterations))
    n_est_per_iter = [base_n_est + 50 * i for i in iterations]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color_f1 = "#1f77b4"
    marker = "o" if l3_iterations > 1 else "s"
    ax1.plot(
        iterations,
        f1_history,
        marker=marker,
        linewidth=2,
        color=color_f1,
        label="Train F1 (real)",
    )
    ax1.set_xlabel("Training Iteration (L3 Loop)")
    ax1.set_ylabel("Train F1", color=color_f1)
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(iterations)
    ax1.grid(True, linestyle="--", alpha=0.4)

    if l3_iterations == 1:
        ax1.annotate(
            f"Single run (no L3 iterations)\nF1 = {f1_history[0]:.4f}",
            xy=(0, f1_history[0]),
            xytext=(0.15, f1_history[0] - 0.12),
            textcoords="data",
            fontsize=8,
            color="#555",
        )

    # Secondary axis: n_estimators per iteration
    ax2 = ax1.twinx()
    color_n = "#ff7f0e"
    ax2.bar(iterations, n_est_per_iter, alpha=0.25, color=color_n, label="n_estimators")
    ax2.set_ylabel("n_estimators", color=color_n)
    ax2.tick_params(axis="y", labelcolor=color_n)

    run_id = summary.get("run_id", "run")
    ax1.set_title(f"L3 Training Progression — {model_type} ({run_id})")

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_learning_rate_comparison(
    runs: list[dict[str, Any]], output_path: Path
) -> bool:
    """
    Cross-run chart comparing learning-related hyperparameters.

    Primary axis  : Train F1 across runs (line)
    Secondary axis: learning_rate across runs (bar)
    """
    plt = _load_pyplot()
    if plt is None:
        return False
    if len(runs) < 2:
        return False

    run_labels: list[str] = []
    train_f1_vals: list[float] = []
    lr_vals: list[float] = []

    for run in runs:
        tm = run.get("training_metrics") or {}
        train_f1 = tm.get("train_f1")
        strategy = run.get("strategy_plan") or {}
        hyper = strategy.get("hyperparameters") or {}
        lr = hyper.get("learning_rate")

        if train_f1 is None or lr is None:
            continue

        run_labels.append(_display_run_label(run))
        train_f1_vals.append(float(train_f1))
        lr_vals.append(float(lr))

    if len(run_labels) < 2:
        return False

    x = list(range(len(run_labels)))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_lr = "#ff7f0e"
    color_f1 = "#1f77b4"

    ax1.bar(x, lr_vals, alpha=0.35, color=color_lr, label="Learning Rate")
    ax1.set_ylabel("Learning Rate", color=color_lr)
    ax1.tick_params(axis="y", labelcolor=color_lr)
    ax1.set_xticks(x)
    ax1.set_xticklabels(run_labels, rotation=30, ha="right")

    for xi, val in zip(x, lr_vals):
        ax1.text(
            xi,
            val + max(lr_vals) * 0.02,
            f"lr={val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555",
        )

    ax2 = ax1.twinx()
    ax2.plot(
        x, train_f1_vals, marker="o", linewidth=2, color=color_f1, label="Train F1"
    )
    ax2.set_ylabel("Train F1", color=color_f1)
    ax2.tick_params(axis="y", labelcolor=color_f1)
    ax2.set_ylim(0, 1.05)

    ax1.set_xlabel("Experiment Configuration", fontsize=11)
    ax1.set_title("Learning Rate vs. Train F1 Comparison")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="upper left")

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
            "Per-run artifacts are stored in descriptive subfolders such as "
            "normal_data_run, scraped_drift_run, or random_drift_run.\n"
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
