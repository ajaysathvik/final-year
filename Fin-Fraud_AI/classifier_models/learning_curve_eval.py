import os
import sys
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "..", "outputs", ".mplconfig"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classifier_models.evaluation_core import (
    CLASSIFIERS,
    DATASETS,
    OUT_DIR,
    clone_classifier,
    get_master_test_split,
    prepare_scaled_data,
    safe_predict_proba,
)

SAMPLE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]
PRIMARY_CLASSIFIER = "XGBoost"


def main():
    print("=" * 60)
    print("  Learning Curve Evaluation")
    print("=" * 60)
    print(f"Primary classifier: {PRIMARY_CLASSIFIER}")

    _, X_test_raw, _, y_test = get_master_test_split()
    base_clf = CLASSIFIERS[PRIMARY_CLASSIFIER]

    rows = []
    for ds_name, ds_path in DATASETS.items():
        X_train, y_train, X_test, _ = prepare_scaled_data(ds_path, X_test_raw)
        n_samples = len(y_train)

        print(f"\n--- Dataset: {ds_name} ({n_samples} training rows) ---")
        for fraction in SAMPLE_FRACTIONS:
            sample_n = max(20, int(round(n_samples * fraction)))
            X_subset = X_train[:sample_n]
            y_subset = y_train[:sample_n]

            clf = clone_classifier(base_clf)
            clf.fit(X_subset, y_subset)

            y_pred = clf.predict(X_test)
            y_prob = safe_predict_proba(clf, X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob)

            print(
                f"  Fraction {fraction:.1f} | rows={sample_n:4d} | "
                f"Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}"
            )

            rows.append(
                {
                    "Dataset": ds_name,
                    "Classifier": PRIMARY_CLASSIFIER,
                    "Sample Fraction": fraction,
                    "Training Rows": sample_n,
                    "Accuracy": round(acc, 4),
                    "F1-Score": round(f1, 4),
                    "AUC-ROC": round(auc, 4),
                }
            )

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "learning_curve_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    if plt is not None and sns is not None:
        sns.set_theme(style="whitegrid", palette="muted")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metric_names = ["Accuracy", "F1-Score", "AUC-ROC"]

        for ax, metric in zip(axes, metric_names):
            sns.lineplot(
                data=results_df,
                x="Sample Fraction",
                y=metric,
                hue="Dataset",
                marker="o",
                linewidth=2,
                ax=ax,
            )
            ax.set_title(f"{metric} Learning Curve", fontweight="bold")
            ax.set_ylim(0.85, 1.01)
            ax.set_xticks(SAMPLE_FRACTIONS)
            ax.set_xlabel("Training Fraction")

        plt.suptitle(
            f"Learning Curves for {PRIMARY_CLASSIFIER} Across Data Regimes",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, "learning_curve_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {plot_path}")
        plt.close()
    else:
        print("Plotting skipped because matplotlib/seaborn are not installed.")


if __name__ == "__main__":
    main()
