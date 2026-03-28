from pathlib import Path

import matplotlib

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "artifacts" / "controlled_real_ratio_experiment_3000" / "baseline"
MPL_DIR = BASE_DIR / "artifacts" / "controlled_real_ratio_experiment_3000" / "mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


PREDICTIONS_FILE = ARTIFACT_DIR / "test_predictions.csv"
OUTPUT_FILE = ARTIFACT_DIR / "pr_auc_curve.png"


def main():
    df = pd.read_csv(PREDICTIONS_FILE)
    y_true = df["y_true"].astype(int)
    y_score = df["y_score"].astype(float)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6.5, 5))
    plt.plot(recall, precision, color="#2a9d8f", linewidth=2, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve: XGBoost Baseline (3000, Train Ratio ~1:18.5)")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=180)
    plt.close()

    print(f"PR AUC: {pr_auc:.6f}")
    print(f"Average Precision: {avg_precision:.6f}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
