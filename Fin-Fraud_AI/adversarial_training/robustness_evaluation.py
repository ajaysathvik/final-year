import os
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "..", "outputs", ".mplconfig"))

import numpy as np
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
    OUT_DIR,
    clone_classifier,
    get_master_test_split,
    get_selected_datasets,
    prepare_scaled_data,
)

DATASETS = get_selected_datasets()

EPSILON = 0.2
PGD_STEPS = 40
PGD_STEP_SIZE = 0.02
MODEL_NAME = "XGBoost"


def tree_style_attack(X, epsilon, rng, numeric_idx=None):
    if epsilon == 0.0 or X.size == 0:
        return X.copy().astype(np.float32)

    numeric_idx = numeric_idx or list(range(X.shape[1]))
    if not numeric_idx:
        return X.copy().astype(np.float32)

    X_adv = X.copy()
    noise = epsilon * np.sign(rng.randn(X.shape[0], len(numeric_idx)))
    X_adv[:, numeric_idx] = X_adv[:, numeric_idx] + noise
    X_adv[:, numeric_idx] = np.clip(X_adv[:, numeric_idx], 0.0, None)
    return X_adv.astype(np.float32)


def fgsm_attack(X, epsilon=EPSILON):
    rng = np.random.RandomState(42)
    return tree_style_attack(X, epsilon=epsilon, rng=rng, numeric_idx=list(range(X.shape[1])))


def pgd_attack(X, epsilon=EPSILON, step_size=PGD_STEP_SIZE, steps=PGD_STEPS):
    rng = np.random.RandomState(42)
    X_start = X.copy().astype(np.float32)
    X_adv = X_start.copy()

    for _ in range(steps):
        candidate = tree_style_attack(
            X_adv,
            epsilon=step_size,
            rng=rng,
            numeric_idx=list(range(X.shape[1])),
        )
        delta = np.clip(candidate - X_start, -epsilon, epsilon)
        X_adv = np.clip(X_start + delta, 0.0, None).astype(np.float32)
    return X_adv


def evaluate_dataset(name, train_path, test_X_raw, test_y):
    print(f"\n--- Evaluating {name} ---")
    X_train, y_train, X_test, _ = prepare_scaled_data(train_path, test_X_raw)

    model = clone_classifier(CLASSIFIERS[MODEL_NAME])
    model.fit(X_train, y_train)

    clean_pred = model.predict(X_test)
    clean_acc = float((clean_pred == test_y).mean())

    X_fgsm = fgsm_attack(X_test)
    fgsm_pred = model.predict(X_fgsm)
    fgsm_acc = float((fgsm_pred == test_y).mean())
    fgsm_perturb = float(np.mean(np.linalg.norm(X_fgsm - X_test, axis=1)))

    X_pgd = pgd_attack(X_test)
    pgd_pred = model.predict(X_pgd)
    pgd_acc = float((pgd_pred == test_y).mean())
    pgd_perturb = float(np.mean(np.linalg.norm(X_pgd - X_test, axis=1)))

    robustness_score = clean_acc - min(fgsm_acc, pgd_acc)
    print(f"Clean Accuracy: {clean_acc:.4f}")
    print(f"FGSM Accuracy:  {fgsm_acc:.4f} | Perturbation: {fgsm_perturb:.4f}")
    print(f"PGD Accuracy:   {pgd_acc:.4f} | Perturbation: {pgd_perturb:.4f}")
    print(f"Robustness Score: {robustness_score:.4f}")

    return {
        "Dataset": name,
        "Classifier": MODEL_NAME,
        "Clean Accuracy": round(clean_acc, 4),
        "FGSM Accuracy": round(fgsm_acc, 4),
        "PGD Accuracy": round(pgd_acc, 4),
        "Robustness Score": round(robustness_score, 4),
        "FGSM Perturbation": round(fgsm_perturb, 4),
        "PGD Perturbation": round(pgd_perturb, 4),
    }


def main():
    print("Loading Master Test Set (Held-out from Original Data)")
    _, X_test_raw, _, y_test = get_master_test_split()

    results = []
    name_map = {
        "Original": "Original (Imbalanced)",
        "CTGAN": "Standard CTGAN",
        "Adv-CTGAN": "Adv-CTGAN (Custom)",
    }

    for ds_key, ds_path in DATASETS.items():
        results.append(evaluate_dataset(name_map[ds_key], ds_path, X_test_raw.copy(), y_test))

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "robustness_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")

    if plt is not None and sns is not None:
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        melt_acc = results_df.melt(
            id_vars=["Dataset", "Classifier"],
            value_vars=["Clean Accuracy", "FGSM Accuracy", "PGD Accuracy"],
            var_name="Condition",
            value_name="Accuracy",
        )
        sns.barplot(data=melt_acc, x="Dataset", y="Accuracy", hue="Condition", ax=axes[0])
        axes[0].set_title("XGBoost Accuracy: Clean vs Adversarial Attacks", fontweight="bold")
        axes[0].set_ylim(0, 1.1)

        melt_rob = results_df.melt(
            id_vars=["Dataset", "Classifier"],
            value_vars=["Robustness Score", "FGSM Perturbation", "PGD Perturbation"],
            var_name="Metric",
            value_name="Value",
        )
        sns.barplot(data=melt_rob, x="Dataset", y="Value", hue="Metric", ax=axes[1])
        axes[1].set_title("XGBoost Robustness Score and Perturbation", fontweight="bold")

        plt.suptitle("Adversarial Robustness Evaluation", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, "robustness_comparison.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out_png}")
        plt.close()
    else:
        print("Plotting skipped because matplotlib/seaborn are not installed.")


if __name__ == "__main__":
    main()
