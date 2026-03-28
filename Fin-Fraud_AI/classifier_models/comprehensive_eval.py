import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "..", "outputs", ".mplconfig"))

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from classifier_models.evaluation_core import (
    CLASSIFIERS,
    DATASETS,
    OUT_DIR,
    clone_classifier,
    compute_metrics,
    get_master_test_split,
    prepare_scaled_data,
    safe_predict_proba,
)


def main():
    print("=" * 60)
    print("  XGBoost Evaluation")
    print("=" * 60)
    print("\nLoading held-out test set from original data...")
    _, X_test_raw, _, y_test = get_master_test_split()
    print(f"Test set: {len(y_test)} samples (fraud=1: {(y_test==1).sum()}, not-fraud=0: {(y_test==0).sum()})\n")

    all_results = []
    model_dir = os.path.join(ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    for ds_name, ds_path in DATASETS.items():
        print(f"\n--- Dataset: {ds_name} ---")
        X_train, y_train, X_test, _ = prepare_scaled_data(ds_path, X_test_raw)

        for clf_name, clf in CLASSIFIERS.items():
            print(f"  Training {clf_name}...", end=" ", flush=True)
            clf_instance = clone_classifier(clf)
            clf_instance.fit(X_train, y_train)

            safe_ds_name = ds_name.lower().replace(" ", "_").replace("-", "_")
            safe_clf_name = clf_name.lower().replace(" ", "_")
            model_filename = f"{safe_clf_name}_{safe_ds_name}.pkl"
            with open(os.path.join(model_dir, model_filename), "wb") as handle:
                pickle.dump(clf_instance, handle)

            y_pred = clf_instance.predict(X_test)
            y_prob = safe_predict_proba(clf_instance, X_test)
            metrics = compute_metrics(y_test, y_pred, y_prob)
            all_results.append({"Dataset": ds_name, "Classifier": clf_name, **metrics})
            print(f"Acc={metrics['Accuracy']}  F1={metrics['F1-Score']}  AUC={metrics['AUC-ROC']}")

    print(f"\n{'='*60}")
    print("  FULL RESULTS TABLE")
    print(f"{'='*60}")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    csv_path = os.path.join(OUT_DIR, "classifier_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    if plt is not None and sns is not None:
        sns.set_theme(style="whitegrid", palette="muted")
        metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

        fig, axes = plt.subplots(1, len(metric_cols), figsize=(24, 6))
        for ax, metric in zip(axes, metric_cols):
            sns.barplot(data=results_df, x="Dataset", y=metric, hue="Classifier", ax=ax)
            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.08)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=15)
            ax.legend(fontsize=7, title=None)
        plt.suptitle("XGBoost Performance Across Datasets", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        out1 = os.path.join(OUT_DIR, "classifier_metrics_comparison.png")
        plt.savefig(out1, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out1}")
        plt.close()

        pivot_f1 = results_df.pivot(index="Classifier", columns="Dataset", values="F1-Score")
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            pivot_f1,
            annot=True,
            fmt=".4f",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={"label": "F1-Score"},
        )
        plt.title("XGBoost F1-Score Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out2 = os.path.join(OUT_DIR, "f1_heatmap.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out2}")
        plt.close()

        pivot_auc = results_df.pivot(index="Classifier", columns="Dataset", values="AUC-ROC")
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            pivot_auc,
            annot=True,
            fmt=".4f",
            cmap="RdYlGn",
            linewidths=0.5,
            cbar_kws={"label": "AUC-ROC"},
        )
        plt.title("XGBoost AUC-ROC Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out3 = os.path.join(OUT_DIR, "aucroc_heatmap.png")
        plt.savefig(out3, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out3}")
        plt.close()
    else:
        print("Plotting skipped because matplotlib/seaborn are not installed.")

    print("\nDone! All results saved to outputs/")


if __name__ == "__main__":
    main()
