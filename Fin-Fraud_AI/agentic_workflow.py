import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "outputs", ".mplconfig"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None
import pandas as pd

from classifier_models.evaluation_core import OUT_DIR, min_max_normalize


def load_required_csv(filename):
    path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required output file: {path}")
    return pd.read_csv(path)


def build_model_level_scores():
    clean_df = load_required_csv("classifier_comparison.csv")
    ood_df = load_required_csv("ood_results.csv")

    merged = clean_df.merge(ood_df, on=["Dataset", "Classifier"], how="inner")
    merged["Clean Utility"] = (
        0.5 * merged["F1-Score"] + 0.3 * merged["AUC-ROC"] + 0.2 * merged["Accuracy"]
    )
    merged["OOD Utility"] = 1 - merged["Accuracy Drop"]
    merged["Calibration Utility"] = min_max_normalize(merged["Log Loss"], invert=True)
    return merged


def build_dataset_level_scores(model_scores):
    robustness_df = load_required_csv("robustness_results.csv")
    quality_df = load_required_csv("synthetic_quality_metrics.csv")

    best_model = (
        model_scores.sort_values(["Dataset", "Clean Utility", "OOD Utility"], ascending=False)
        .groupby("Dataset", as_index=False)
        .first()
        .rename(columns={"Classifier": "Selected Classifier"})
    )

    robustness_map = {
        "Original (Imbalanced)": "Original",
        "Standard CTGAN": "CTGAN",
        "Adv-CTGAN (Custom)": "Adv-CTGAN",
    }
    robustness_df["Dataset"] = robustness_df["Dataset"].map(robustness_map)
    robustness_df["Robustness Utility"] = min_max_normalize(
        robustness_df["Robustness Score"], invert=True
    )

    quality_df["Dataset"] = quality_df["Model"].map(
        {"CTGAN": "CTGAN", "Adv-CTGAN": "Adv-CTGAN"}
    )
    quality_df["Synthetic Utility"] = (
        0.6 * min_max_normalize(quality_df["Diversity"])
        + 0.4 * min_max_normalize(quality_df["FID"], invert=True)
    )

    dataset_scores = best_model.merge(
        robustness_df[
            [
                "Dataset",
                "Clean Accuracy",
                "FGSM Accuracy",
                "PGD Accuracy",
                "Robustness Score",
                "Robustness Utility",
            ]
        ],
        on="Dataset",
        how="left",
    ).merge(
        quality_df[["Dataset", "FID", "Diversity", "Coverage", "Synthetic Utility"]],
        on="Dataset",
        how="left",
    )

    dataset_scores["Synthetic Utility"] = dataset_scores["Synthetic Utility"].fillna(0.0)
    dataset_scores["Agentic Score"] = (
        0.35 * dataset_scores["Clean Utility"]
        + 0.30 * dataset_scores["Robustness Utility"]
        + 0.20 * dataset_scores["OOD Utility"]
        + 0.10 * dataset_scores["Calibration Utility"]
        + 0.05 * dataset_scores["Synthetic Utility"]
    ).round(4)

    dataset_scores["Decision"] = dataset_scores["Agentic Score"].rank(
        ascending=False, method="dense"
    )
    dataset_scores["Decision"] = dataset_scores["Decision"].map(
        {
            1.0: "Deploy now",
            2.0: "Use as robustness-aware backup",
            3.0: "Keep as baseline only",
        }
    )
    return dataset_scores.sort_values("Agentic Score", ascending=False)


def save_outputs(model_scores, dataset_scores):
    model_csv = os.path.join(OUT_DIR, "agentic_model_scores.csv")
    dataset_csv = os.path.join(OUT_DIR, "agentic_metric_results.csv")
    dataset_md = os.path.join(OUT_DIR, "agentic_decision_summary.md")

    model_scores.to_csv(model_csv, index=False)
    dataset_scores.to_csv(dataset_csv, index=False)

    top_row = dataset_scores.iloc[0]
    lines = [
        "# Agentic Decision Summary",
        "",
        "The agentic workflow uses four evidence channels:",
        "- clean classification utility",
        "- adversarial robustness utility",
        "- corruption-shift stability",
        "- calibration quality",
        "",
        f"Recommended deployment regime: **{top_row['Dataset']}**",
        f"- Selected classifier: `{top_row['Selected Classifier']}`",
        f"- Agentic score: `{top_row['Agentic Score']:.4f}`",
        f"- Decision: `{top_row['Decision']}`",
        "",
        "Dataset-level ranking:",
    ]
    for _, row in dataset_scores.iterrows():
        lines.append(
            f"- {row['Dataset']}: score `{row['Agentic Score']:.4f}`, "
            f"classifier `{row['Selected Classifier']}`, decision `{row['Decision']}`"
        )

    with open(dataset_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    plot_path = os.path.join(OUT_DIR, "agentic_metric_comparison.png")
    if plt is not None and sns is not None:
        sns.set_theme(style="whitegrid", palette="muted")
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(data=dataset_scores, x="Dataset", y="Agentic Score")
        ax.set_title("Agentic Metric by Dataset Regime", fontweight="bold")
        ax.set_ylim(0, 1.05)
        for patch, score in zip(ax.patches, dataset_scores["Agentic Score"]):
            ax.annotate(
                f"{score:.3f}",
                (patch.get_x() + patch.get_width() / 2, score),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plot_path = ""

    return model_csv, dataset_csv, dataset_md, plot_path


def main():
    print("=" * 60)
    print("  Agentic Workflow Evaluation")
    print("=" * 60)
    model_scores = build_model_level_scores()
    dataset_scores = build_dataset_level_scores(model_scores)

    print("\nDataset-level agentic ranking:")
    print(
        dataset_scores[
            [
                "Dataset",
                "Selected Classifier",
                "Clean Utility",
                "Robustness Utility",
                "OOD Utility",
                "Calibration Utility",
                "Synthetic Utility",
                "Agentic Score",
                "Decision",
            ]
        ].to_string(index=False)
    )

    model_csv, dataset_csv, dataset_md, plot_path = save_outputs(model_scores, dataset_scores)
    print(f"\nSaved -> {model_csv}")
    print(f"Saved -> {dataset_csv}")
    print(f"Saved -> {dataset_md}")
    if plot_path:
        print(f"Saved -> {plot_path}")
    else:
        print("Plotting skipped because matplotlib/seaborn are not installed.")


if __name__ == "__main__":
    main()
