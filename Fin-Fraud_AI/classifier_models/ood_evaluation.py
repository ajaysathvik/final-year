"""
Out-of-Distribution (OOD) Testing
==================================
Evaluates how well classifiers trained on each dataset generalize
to corruption-based distribution shift.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "..", "outputs", ".mplconfig"))

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import accuracy_score

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
    get_master_test_split,
    prepare_scaled_data,
)


def generate_ood_data(X_test, noise_level=1.0, corruption_frac=0.3, seed=42):
    rng = np.random.RandomState(seed)
    X_ood = X_test.copy()
    X_ood = X_ood + rng.normal(0, noise_level, size=X_ood.shape)

    n_corrupt = int(X_ood.shape[1] * corruption_frac)
    corrupt_cols = rng.choice(X_ood.shape[1], n_corrupt, replace=False)
    for col in corrupt_cols:
        X_ood[:, col] = rng.permutation(X_ood[:, col])

    return X_ood.astype(np.float32)


def prediction_entropy(probs):
    ent = []
    for prob in probs:
        prob = np.clip(prob, 1e-10, 1 - 1e-10)
        ent.append(scipy_entropy([prob, 1 - prob]))
    return float(np.mean(ent))


def main():
    print("=" * 60)
    print("  Out-of-Distribution (OOD) Testing")
    print("=" * 60)

    _, X_test_raw, _, y_test = get_master_test_split()
    print(f"Test set: {len(y_test)} samples\n")

    rows = []
    for ds_name, ds_path in DATASETS.items():
        print(f"\n--- Dataset: {ds_name} ---")
        X_train, y_train, X_test, _ = prepare_scaled_data(ds_path, X_test_raw)
        X_ood = generate_ood_data(X_test, noise_level=1.0, corruption_frac=0.3)

        for clf_name, clf in CLASSIFIERS.items():
            clf_instance = clone_classifier(clf)
            clf_instance.fit(X_train, y_train)

            y_pred_clean = clf_instance.predict(X_test)
            clean_acc = accuracy_score(y_test, y_pred_clean)

            y_pred_ood = clf_instance.predict(X_ood)
            ood_acc = accuracy_score(y_test, y_pred_ood)

            try:
                ood_probs = clf_instance.predict_proba(X_ood)[:, 1]
                ent = prediction_entropy(ood_probs)
            except Exception:
                ent = float("nan")

            acc_drop = clean_acc - ood_acc
            print(
                f"  {clf_name:25s} | Clean: {clean_acc:.4f} | "
                f"OOD: {ood_acc:.4f} | Drop: {acc_drop:.4f} | Entropy: {ent:.4f}"
            )

            rows.append(
                {
                    "Dataset": ds_name,
                    "Classifier": clf_name,
                    "Clean Accuracy": round(clean_acc, 4),
                    "OOD Accuracy": round(ood_acc, 4),
                    "Accuracy Drop": round(acc_drop, 4),
                    "OOD Entropy": round(ent, 4),
                }
            )

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "ood_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    if plt is not None and sns is not None:
        sns.set_theme(style="whitegrid", palette="muted")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.barplot(data=results_df, x="Dataset", y="OOD Accuracy", hue="Classifier", ax=axes[0])
        axes[0].set_title("OOD Accuracy by Dataset", fontweight="bold")
        axes[0].set_ylim(0, 1.1)

        sns.barplot(data=results_df, x="Dataset", y="OOD Entropy", hue="Classifier", ax=axes[1])
        axes[1].set_title("Prediction Entropy on OOD Data", fontweight="bold")

        plt.suptitle("Out-of-Distribution Generalization", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, "ood_comparison.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out_png}")
        plt.close()

        pivot_drop = results_df.pivot(index="Classifier", columns="Dataset", values="Accuracy Drop")
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            pivot_drop,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            linewidths=0.5,
            cbar_kws={"label": "Accuracy Drop"},
        )
        plt.title("Accuracy Drop on OOD Data", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_heat = os.path.join(OUT_DIR, "ood_accuracy_drop_heatmap.png")
        plt.savefig(out_heat, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out_heat}")
        plt.close()
    else:
        print("Plotting skipped because matplotlib/seaborn are not installed.")


if __name__ == "__main__":
    main()
