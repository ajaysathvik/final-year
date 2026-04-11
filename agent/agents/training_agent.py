"""
Training Agent — Execute phase of MAPE-K.
Trains a model on balanced data using the Strategy Agent's plan.
L3 feedback loop: self-loop for iterative retraining.
On success, returns control to Strategy so Strategy remains the central hub.

Evaluation: TabularBench-style robustness metrics computed inline —
  Standard Accuracy, Robust Accuracy (under CAA), Attack Success Rate (ASR),
  Feature Sensitivity Map — plotted and saved to artifacts/training_robustness/.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

from config import MODELS_DIR, RANDOM_STATE, MAX_L3_ITERATIONS, F1_THRESHOLD

try:
    from advanced_adversarial import AdvancedAdversarialTrainer

    ADVANCED_ADV_AVAILABLE = True
except ImportError:
    ADVANCED_ADV_AVAILABLE = False


# ── TabularBench-style CAA evaluation (inline, no external imports) ──────────

def _caa_attack_inline(
    X_in: np.ndarray,
    y_in: np.ndarray,
    model,
    feature_cols: list[str],
    epsilon: float = 0.1,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """
    Constrained Adaptive Attack (CAA) — TabularBench framework logic,
    implemented inline without importing tabularbench_caa.py.

    Returns:
        X_adv       — perturbed samples
        asr         — attack success rate (0-100)
        sensitivity — {feature_name: contribution_fraction}
    """
    X_adv = X_in.copy().astype(np.float32)
    fraud_indices = np.where(y_in == 1)[0]

    success_flags = np.zeros(len(fraud_indices), dtype=np.float32)
    feature_flips: dict[str, int] = {f: 0 for f in feature_cols}

    try:
        y_pred_initial = model.predict(X_in)
    except Exception:
        return X_adv, 0.0, {}

    for idx_i, idx in enumerate(fraud_indices):
        # Only attack samples the model already identifies as fraud
        if y_pred_initial[idx] == 0:
            continue

        x_curr = X_adv[idx].copy()
        for f_idx, f_name in enumerate(feature_cols):
            perturbation = np.random.uniform(-epsilon * 50, epsilon * 50)
            x_test = x_curr.copy()
            x_test[f_idx] += perturbation
            try:
                pred = model.predict(x_test.reshape(1, -1))[0]
            except Exception:
                continue
            if pred == 0:
                X_adv[idx] = x_test
                feature_flips[f_name] += 1
                success_flags[idx_i] = 1
                break

    total_successes = sum(feature_flips.values())
    if total_successes > 0:
        sensitivity = {
            k: v / total_successes
            for k, v in feature_flips.items()
            if v > 0
        }
    else:
        sensitivity = {}

    asr = float(np.mean(success_flags) * 100)
    return X_adv, asr, sensitivity


def _compute_tabularbench_metrics(
    model,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_cols: list[str],
    epsilon: float = 0.1,
) -> dict:
    """
    Compute TabularBench-framework robustness metrics:
      - standard_accuracy  : clean accuracy on eval set
      - robust_accuracy    : accuracy after CAA perturbation
      - asr                : attack success rate (%)
      - sensitivity_map    : feature → contribution fraction
      - is_robust          : bool — drop < 20 pp considered robust
    """
    try:
        y_pred_clean = model.predict(X_eval)
        std_acc = float(accuracy_score(y_eval, y_pred_clean) * 100)
    except Exception:
        std_acc = 0.0

    rng_state = np.random.get_state()
    np.random.seed(RANDOM_STATE)
    X_adv, asr, sensitivity = _caa_attack_inline(
        X_eval, y_eval, model, feature_cols, epsilon=epsilon
    )
    np.random.set_state(rng_state)

    try:
        y_pred_adv = model.predict(X_adv)
        rob_acc = float(accuracy_score(y_eval, y_pred_adv) * 100)
    except Exception:
        rob_acc = 0.0

    is_robust = (std_acc - rob_acc) <= 20.0

    robustness_score = round(rob_acc / 100.0, 4)

    return {
        "standard_accuracy": round(std_acc, 2),
        "robust_accuracy": round(rob_acc, 2),
        "robustness_score": robustness_score,
        "asr": round(asr, 2),
        "accuracy_drop": round(std_acc - rob_acc, 2),
        "is_robust": is_robust,
        "sensitivity_map": sensitivity,
    }


# ── History helpers ──────────────────────────────────────────────────────────

HISTORY_FILE_NAME = "tabularbench_history.json"


def _load_history(output_dir: Path) -> list[dict]:
    """Load accumulated run history from JSON; return empty list if none."""
    path = output_dir / HISTORY_FILE_NAME
    if path.exists():
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(history: list[dict], output_dir: Path) -> None:
    """Persist accumulated run history to JSON."""
    import json
    path = output_dir / HISTORY_FILE_NAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ── Cumulative multi-run plot ─────────────────────────────────────────────────

def _plot_tabularbench_metrics(
    tb_metrics: dict,
    model_type: str,
    timestamp: str,
    output_dir: Path,
) -> str:
    """
    Append current run to history, then generate a cumulative 3-panel plot:
      Panel 1 — Standard vs Robust Accuracy trend over all runs (line)
      Panel 2 — ASR & Accuracy Drop trend over all runs (line)
      Panel 3 — Latest Feature Sensitivity Map (barh, top-5)
    Plot saved as 'tabularbench_cumulative.png' (overwritten each run so
    latest always at a fixed path) plus a timestamped snapshot copy.
    Returns the fixed cumulative path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load, append, save history ────────────────────────────────
    history = _load_history(output_dir)
    history.append({
        "run": len(history) + 1,
        "timestamp": timestamp,
        "model_type": model_type,
        "standard_accuracy": tb_metrics["standard_accuracy"],
        "robust_accuracy": tb_metrics["robust_accuracy"],
        "robustness_score": tb_metrics["robustness_score"],
        "asr": tb_metrics["asr"],
        "accuracy_drop": tb_metrics["accuracy_drop"],
        "is_robust": tb_metrics["is_robust"],
        "sensitivity_map": tb_metrics.get("sensitivity_map", {}),
    })
    _save_history(history, output_dir)

    # ── Extract series ────────────────────────────────────────────
    run_labels = []
    for h in history:
        if h["run"] == 1:
            lbl = "baseline"
        elif h["run"] == 2:
            lbl = "baseline with drift"
        else:
            lbl = f"Run {h['run']}"
        run_labels.append(f"{lbl}\n{h['timestamp'][4:13]}")
    x = list(range(1, len(history) + 1))
    std_accs = [h["standard_accuracy"] for h in history]
    rob_accs = [h["robust_accuracy"] for h in history]
    asrs = [h["asr"] for h in history]
    drops = [h["accuracy_drop"] for h in history]
    robust_flags = [h["is_robust"] for h in history]

    # Latest sensitivity
    sensitivity = tb_metrics.get("sensitivity_map", {})
    sorted_sens = dict(sorted(sensitivity.items(), key=lambda kv: kv[1], reverse=True))
    top_features = list(sorted_sens.keys())[:5]
    top_vals = [sorted_sens[k] * 100 for k in top_features]
    clean_labels = [f.split(".")[-1].replace("_", " ").title() for f in top_features]

    # ── Light theme ───────────────────────────────────────────────
    BG     = "white"
    AX_BG  = "#f8f9fa"
    SPINE  = "#dee2e6"
    GREEN  = "#2d9f6e"
    RED    = "#e05c5c"
    BLUE   = "#3d7eba"
    YELLOW = "#d4880a"
    TEXT   = "#212529"

    n_panels = 3
    fig, axes = plt.subplots(1, n_panels, figsize=(20, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"TabularBench Robustness — Cumulative ({len(history)} run{'s' if len(history) != 1 else ''})",
        color=TEXT, fontsize=14, fontweight="bold", y=1.02,
    )

    def _style_ax(ax):
        ax.set_facecolor(AX_BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(SPINE)
        for sp in ["top", "right"]:
            ax.spines[sp].set_color("none")

    # ── Panel 1: Accuracy trend ───────────────────────────────────
    ax1 = axes[0]
    _style_ax(ax1)
    ax1.plot(x, std_accs, marker="o", color=GREEN,  linewidth=2, label="Standard Acc", markersize=6)
    ax1.plot(x, rob_accs, marker="s", color=RED,    linewidth=2, label="Robust Acc (CAA)", markersize=6)
    for xi, (sa, ra, ok) in enumerate(zip(std_accs, rob_accs, robust_flags), start=1):
        ax1.annotate(f"{sa:.1f}%", (xi, sa), textcoords="offset points", xytext=(0, 7),
                     ha="center", color=GREEN, fontsize=7)
        ax1.annotate(f"{ra:.1f}%", (xi, ra), textcoords="offset points", xytext=(0, -12),
                     ha="center", color=RED, fontsize=7)
    ax1.set_ylim(max(0, min(rob_accs) - 10), 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(run_labels, fontsize=7)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_title("Accuracy Trend (Standard vs Robust)", pad=12, fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4, color=SPINE)

    # ── Panel 2: ASR & Drop trend ─────────────────────────────────
    ax2 = axes[1]
    _style_ax(ax2)
    ax2_twin = ax2.twinx()
    ax2_twin.set_facecolor(AX_BG)
    ax2_twin.tick_params(labelsize=8)
    ax2_twin.spines["right"].set_color(SPINE)
    ax2_twin.spines["top"].set_color("none")

    ax2.plot(x, asrs,  marker="^", color=YELLOW, linewidth=2, label="ASR (%)", markersize=6)
    ax2_twin.plot(x, drops, marker="D", color=BLUE, linewidth=2, label="Acc Drop (pp)", markersize=5, linestyle="--")
    for xi, (asr_v, drop_v) in enumerate(zip(asrs, drops), start=1):
        ax2.annotate(f"{asr_v:.1f}", (xi, asr_v), textcoords="offset points", xytext=(0, 7),
                     ha="center", color=YELLOW, fontsize=7)
        ax2_twin.annotate(f"{drop_v:.2f}", (xi, drop_v), textcoords="offset points", xytext=(0, -12),
                          ha="center", color=BLUE, fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(run_labels, fontsize=7)
    ax2.set_ylabel("ASR (%)", fontsize=10)
    ax2_twin.set_ylabel("Accuracy Drop (pp)", fontsize=10)
    ax2.set_title("Attack Success Rate & Accuracy Drop", pad=12, fontsize=11)
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, color=SPINE)

    # ── Panel 3: Latest feature sensitivity barh ──────────────────
    ax3 = axes[2]
    _style_ax(ax3)
    if top_features:
        bars = ax3.barh(clean_labels, top_vals, color=BLUE, edgecolor=SPINE, height=0.5)
        ax3.invert_yaxis()
        ax3.set_xlabel("Contribution to Successful Attacks (%)", fontsize=9)
        for i, v in enumerate(top_vals):
            ax3.text(v + 0.4, i, f"{v:.1f}%", va="center", color=TEXT, fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No sensitivity data\n(no CAA successes)",
                 transform=ax3.transAxes, ha="center", va="center", color=TEXT, fontsize=10)
    ax3.set_title(f"Feature Sensitivity Map\n(Latest run — {model_type})", pad=12, fontsize=11)
    ax3.grid(axis="x", linestyle="--", alpha=0.4, color=SPINE)

    # ── Robustness status footer ──────────────────────────────────
    current = history[-1]
    status_color = GREEN if current["is_robust"] else RED
    status_text  = "✔ ROBUST" if current["is_robust"] else "✘ NOT ROBUST"
    fig.text(
        0.5, -0.01,
        f"Latest run — Std: {current['standard_accuracy']:.2f}%  "
        f"Robust: {current['robust_accuracy']:.2f}%  "
        f"Score: {current.get('robustness_score', 0.0):.4f}  "
        f"ASR: {current['asr']:.2f}%  |  {status_text}",
        ha="center", color=status_color, fontsize=9, style="italic",
    )

    plt.tight_layout()

    # Fixed path (always current)
    fixed_path = output_dir / "tabularbench_cumulative.png"
    plt.savefig(str(fixed_path), facecolor=BG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fixed_path)


# ── Adversarial sample generation (unchanged) ────────────────────────────────

def _generate_adversarial_samples(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    adv_strategy: dict,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Generate adversarial samples using advanced techniques (APFE, ASC, TLA, FraudGAN)."""
    noise_cfg = adv_strategy.get("noise_perturbation", {})
    epsilon = float(noise_cfg.get("std", 0.05))

    use_advanced = adv_strategy.get("use_advanced", False)
    advanced_config = adv_strategy.get("advanced_config", {})

    if train_df.empty or not feature_cols:
        return pd.DataFrame(), {"generated": 0, "reason": "no_train_features"}

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(int)

    rng = np.random.RandomState(RANDOM_STATE)

    if use_advanced and ADVANCED_ADV_AVAILABLE:
        enable_apfe = advanced_config.get("enable_apfe", True)
        enable_asc = advanced_config.get("enable_asc", True)
        enable_tla = advanced_config.get("enable_tla", True)
        enable_fraudgan = advanced_config.get("enable_fraudgan", True)
        n_clusters = advanced_config.get("n_clusters", 5)
        triplet_weight = advanced_config.get("triplet_weight", 0.1)

        trainer = AdvancedAdversarialTrainer(
            epsilon=epsilon,
            enable_apfe=enable_apfe,
            enable_asc=enable_asc,
            enable_tla=enable_tla,
            enable_fraudgan=enable_fraudgan,
            n_clusters=n_clusters,
            triplet_weight=triplet_weight,
        )

        X_adv, y_adv = trainer.generate_adversarial_training_set(
            X_train, y_train, method="all"
        )

        method = "advanced_adversarial"
        report = {
            "generated": int(len(X_adv) - len(X_train)),
            "base_rows": int(len(train_df)),
            "method": method,
            "epsilon": epsilon,
            "augmented_total": int(len(X_adv)),
            "techniques_used": trainer.get_technique_summary(),
            "apfe_enabled": enable_apfe,
            "asc_enabled": enable_asc,
            "tla_enabled": enable_tla,
            "fraudgan_enabled": enable_fraudgan,
        }

        adv_df = pd.DataFrame(X_adv, columns=feature_cols)
        adv_df[target_col] = y_adv

        return adv_df, report

    from xgboost import XGBClassifier

    # Train preliminary model for CAA guided generation
    prelim_model = XGBClassifier(
        n_estimators=50, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss"
    )
    prelim_model.fit(X_train, y_train)

    X_adv, _, _ = _caa_attack_inline(X_train, y_train, prelim_model, feature_cols, epsilon=epsilon)

    adv_df = pd.DataFrame(X_adv, columns=feature_cols)
    adv_df[target_col] = y_train

    report = {
        "generated": int(len(adv_df)),
        "base_rows": int(len(train_df)),
        "method": "caa_attack",
        "epsilon": epsilon,
        "augmented_total": int(len(train_df) + len(adv_df)),
    }
    return adv_df, report


# ── Main training agent ───────────────────────────────────────────────────────

def training_agent(state: dict) -> dict:
    """
    Train a classifier using the balanced dataset and strategy plan.
    Supports L3 self-loop for iterative improvement.
    Increments l3_count when self-looping.
    After training, runs TabularBench-style robustness evaluation and plots results.
    """
    print("\n" + "=" * 60)
    print(" [ TRAINING AGENT ] Training candidate model...")
    print("=" * 60)

    kb = state["knowledge_base"]
    strategy = state.get("strategy_plan", {})
    feature_cols = state["feature_cols"]
    target_col = state["target_col"]
    l3_count = state.get("l3_count", 0)
    f1_history: list[float] = list(state.get("f1_history") or [])
    strategy_decision = state.get("strategy_decision")

    # Prefer drift-remediated data, then balanced data built from it.
    base_train_data = state.get("balanced_train_df")
    if base_train_data is None:
        base_train_data = state.get("remediated_train_df", state["train_df"])
    train_data = base_train_data
    use_adversarial_training = bool(strategy.get("use_adversarial_training", False))
    adv_samples = state.get("adversarial_samples")
    adv_report = state.get("adversarial_report", {})
    used_adversarial_samples = False
    drift_remediation = state.get("drift_remediation", {})
    used_drift_remediation = bool(drift_remediation.get("applied", False))

    if use_adversarial_training:
        adv_samples, adv_report = _generate_adversarial_samples(
            train_df=base_train_data,
            feature_cols=feature_cols,
            target_col=target_col,
            adv_strategy=strategy.get("adversarial_strategy", {}),
        )
        if isinstance(adv_samples, pd.DataFrame) and not adv_samples.empty:
            train_data = pd.concat([base_train_data, adv_samples], ignore_index=True)
            used_adversarial_samples = True

    # ── Bug-5 fix: detect post-augmentation class imbalance ─────
    post_augmentation_imbalanced = False
    y_check = train_data[target_col].values.astype(int)
    n_fraud_post = int(y_check.sum())
    n_non_fraud_post = int(len(y_check) - n_fraud_post)
    if n_non_fraud_post > 0:
        post_aug_ratio = n_fraud_post / n_non_fraud_post
    else:
        post_aug_ratio = float("inf")

    if post_aug_ratio > 3.0 or post_aug_ratio < 0.3:
        post_augmentation_imbalanced = True
        print(
            f"  ⚠️ Post-augmentation imbalance detected: "
            f"fraud={n_fraud_post}, non-fraud={n_non_fraud_post}, ratio={post_aug_ratio:.2f}"
        )

    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values.astype(int)

    print(
        f"  Training rows: {len(train_data)} "
        f"(fraud={int(y_train.sum())}, non-fraud={int(len(y_train) - y_train.sum())})"
    )
    if used_drift_remediation:
        print(
            f"  ℹ️  Drift remediation enabled with "
            f"{drift_remediation.get('rows_added', 0)} appended scraped rows"
        )
    if use_adversarial_training:
        if used_adversarial_samples:
            print(
                f"  ℹ️  CAA adversarial strengthening enabled "
                f"(epsilon={adv_report.get('epsilon', 'n/a')}, generated={len(adv_samples)})"
            )
        else:
            print(
                f"  ℹ️  Adversarial strengthening requested but skipped: {adv_report.get('reason', 'no_samples_generated')}"
            )
    if l3_count > 0:
        print(f"  ℹ️  L3 self-loop iteration #{l3_count}")

    # ── Build model from strategy plan ──────────────────────────
    hyper = strategy.get("hyperparameters", {})
    model_type = strategy.get("model_type", "XGBoost")
    n_estimators = hyper.get("n_estimators", 200)

    # On L3 retrain, boost estimators
    if l3_count > 0:
        n_estimators = n_estimators + (50 * l3_count)
        print(f"  ℹ️  L3 boost: n_estimators increased to {n_estimators}")

    if model_type == "XGBoost":
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=hyper.get("max_depth", 6),
                learning_rate=hyper.get("learning_rate", 0.1),
                scale_pos_weight=max(
                    1, int((y_train == 0).sum() / max(1, (y_train == 1).sum()))
                ),
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                use_label_encoder=False,
            )
            print(f"  Using XGBoost (n_estimators={n_estimators})...")
        except ImportError:
            model_type = "LightGBM"  # fall through
            print("  ⚠️  XGBoost not available, falling back to LightGBM.")

    elif model_type == "LightGBM":
        try:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=hyper.get("max_depth", 6),
                learning_rate=hyper.get("learning_rate", 0.1),
                class_weight=hyper.get("class_weight", "balanced"),
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
            print(f"  Using LightGBM (n_estimators={n_estimators})...")
        except ImportError:
            raise ImportError("LightGBM not available.")
    elif model_type == "CatBoost":
        try:
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                iterations=n_estimators,
                depth=hyper.get("max_depth", 6),
                learning_rate=hyper.get("learning_rate", 0.1),
                auto_class_weights="Balanced",
                random_state=RANDOM_STATE,
                verbose=0,
            )
            print(f"  Using CatBoost (iterations={n_estimators})...")
        except ImportError:
            raise ImportError("CatBoost not available.")
    else:
        raise ValueError(f"Unsupported model_type from strategy plan: {model_type}")

    model.fit(X_train, y_train)

    # Quick training-set check
    train_preds = model.predict(X_train)
    train_f1 = round(float(f1_score(y_train, train_preds, zero_division=0)), 4)
    print(f"  ✅ Training complete. Train-set F1={train_f1} ({model_type})")

    # Save model
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = str(MODELS_DIR / f"candidate_{model_type.lower()}_{timestamp}.joblib")
    joblib.dump(model, model_path)
    print(f"  💾 Saved model: {model_path}")

    # ── TabularBench-style robustness evaluation ─────────────────
    print("\n  [ TABULARBENCH EVAL ] Running CAA robustness evaluation...")
    adv_epsilon = float(
        strategy.get("adversarial_strategy", {})
        .get("noise_perturbation", {})
        .get("std", 0.1)
    )
    tb_metrics = _compute_tabularbench_metrics(
        model=model,
        X_eval=X_train,
        y_eval=y_train,
        feature_cols=feature_cols,
        epsilon=adv_epsilon,
    )
    print(f"  Standard Accuracy : {tb_metrics['standard_accuracy']:.2f}%")
    print(f"  Robust Accuracy   : {tb_metrics['robust_accuracy']:.2f}%")
    print(f"  Robustness Score  : {tb_metrics['robustness_score']:.4f} (TabularBench)")
    print(f"  Accuracy Drop     : {tb_metrics['accuracy_drop']:.2f} pp")
    print(f"  Attack Success Rate (ASR): {tb_metrics['asr']:.2f}%")
    print(f"  Robustness Status : {'✔ ROBUST' if tb_metrics['is_robust'] else '✘ NOT ROBUST'}")
    if tb_metrics["sensitivity_map"]:
        print("  Top Feature Vulnerabilities:")
        for feat, contrib in list(
            sorted(tb_metrics["sensitivity_map"].items(), key=lambda x: x[1], reverse=True)
        )[:5]:
            print(f"    - {feat}: {contrib * 100:.1f}%")

    # ── Plot & save ───────────────────────────────────────────────
    plot_output_dir = Path(MODELS_DIR).parent / "plots"
    plot_path = _plot_tabularbench_metrics(
        tb_metrics=tb_metrics,
        model_type=model_type,
        timestamp=timestamp,
        output_dir=plot_output_dir,
    )
    _history_count = len(_load_history(plot_output_dir))  # already includes this run
    print(f"  📊 Cumulative robustness plot updated ({_history_count} run(s)): {plot_path}")

    # L3 decision: if train F1 is too low, increment counter for self-loop
    new_l3_count = l3_count
    if train_f1 < F1_THRESHOLD and l3_count < MAX_L3_ITERATIONS:
        new_l3_count = l3_count + 1
        print(
            f"  🔄 L3: Train-set F1={train_f1} < {F1_THRESHOLD}, will retrain (count={new_l3_count})"
        )

    # Append the real F1 measured this iteration to the history list.
    f1_history.append(train_f1)

    metrics = {
        "model_type": model_type,
        "train_f1": train_f1,
        "f1_history": f1_history,
        "total_rows": len(train_data),
        "n_fraud": int(y_train.sum()),
        "n_non_fraud": int(len(y_train) - y_train.sum()),
        "n_estimators": n_estimators,
        "l3_count": new_l3_count,
        "used_drift_remediation": used_drift_remediation,
        "drift_rows_added": int(drift_remediation.get("rows_added", 0))
        if used_drift_remediation
        else 0,
        "used_adversarial_samples": used_adversarial_samples,
        "adversarial_sample_count": int(len(adv_samples))
        if used_adversarial_samples
        else 0,
        "adversarial_method": adv_report.get("method")
        if used_adversarial_samples
        else None,
        "adversarial_epsilon": adv_report.get("epsilon")
        if used_adversarial_samples
        else None,
        "timestamp": timestamp,
        # TabularBench robustness metrics
        "tabularbench_standard_accuracy": tb_metrics["standard_accuracy"],
        "tabularbench_robust_accuracy": tb_metrics["robust_accuracy"],
        "tabularbench_robustness_score": tb_metrics["robustness_score"],
        "tabularbench_accuracy_drop": tb_metrics["accuracy_drop"],
        "tabularbench_asr": tb_metrics["asr"],
        "tabularbench_is_robust": tb_metrics["is_robust"],
        "tabularbench_sensitivity_map": tb_metrics["sensitivity_map"],
        "tabularbench_plot_path": plot_path,
    }

    if use_adversarial_training and kb:
        kb.log_event("adversarial", "adversarial_samples_generated", adv_report)

    kb.log_event("training", "model_trained", metrics)
    kb.log_event("training", "tabularbench_eval", tb_metrics)
    kb.save_metrics(f"training_metrics_{timestamp}", metrics)

    return {
        **state,
        "candidate_model": model,
        "candidate_model_path": model_path,
        "training_metrics": metrics,
        "candidate_needs_evaluation": True,
        "simulation_results": {},
        "simulation_passed": None,
        "needs_strategy_refinement": False,
        "needs_rebalance": False,
        "l3_count": new_l3_count,
        "f1_history": f1_history,
        "adversarial_samples": adv_samples
        if use_adversarial_training
        else pd.DataFrame(),
        "adversarial_report": adv_report if use_adversarial_training else {},
        "adversarial_trained": used_adversarial_samples,
        "augmented_train_df": train_data,  # Bug-5: store full augmented data for balance re-check
        "post_augmentation_imbalanced": post_augmentation_imbalanced,  # Bug-5: flag for L1
        "tabularbench_metrics": tb_metrics,
    }
