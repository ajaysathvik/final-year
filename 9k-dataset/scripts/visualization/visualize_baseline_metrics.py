import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "artifacts" / "xgb_balanced_train_imbalanced_test"
os.environ["MPLCONFIGDIR"] = str(ARTIFACT_DIR / "mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


METRICS_FILE = ARTIFACT_DIR / "metrics.json"
CONFUSION_MATRIX_FILE = ARTIFACT_DIR / "confusion_matrix_heatmap.png"
METRICS_BAR_FILE = ARTIFACT_DIR / "overall_metrics_bar.png"
CLASS_METRICS_FILE = ARTIFACT_DIR / "class_metrics_bar.png"


def load_metrics():
    with METRICS_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_confusion_matrix(metrics):
    cm = metrics["confusion_matrix"]
    labels = ["not_fraud (0)", "fraud (1)"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=160)
    plt.close()


def plot_overall_metrics(metrics):
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [metrics[name] for name in metric_names]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=["#2a9d8f", "#264653", "#e9c46a", "#f4a261", "#e76f51"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Overall Performance Metrics")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(METRICS_BAR_FILE, dpi=160)
    plt.close()


def plot_class_metrics(metrics):
    report = metrics["classification_report"]
    classes = ["not_fraud (0)", "fraud (1)"]
    precision_vals = [report["0"]["precision"], report["1"]["precision"]]
    recall_vals = [report["0"]["recall"], report["1"]["recall"]]
    f1_vals = [report["0"]["f1-score"], report["1"]["f1-score"]]

    x = range(len(classes))
    width = 0.24

    plt.figure(figsize=(8, 4.8))
    plt.bar([i - width for i in x], precision_vals, width=width, label="Precision", color="#264653")
    plt.bar(x, recall_vals, width=width, label="Recall", color="#2a9d8f")
    plt.bar([i + width for i in x], f1_vals, width=width, label="F1", color="#e76f51")
    plt.xticks(list(x), classes)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Class-wise Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CLASS_METRICS_FILE, dpi=160)
    plt.close()


def main():
    metrics = load_metrics()
    plot_confusion_matrix(metrics)
    plot_overall_metrics(metrics)
    plot_class_metrics(metrics)
    print("Saved:", CONFUSION_MATRIX_FILE)
    print("Saved:", METRICS_BAR_FILE)
    print("Saved:", CLASS_METRICS_FILE)


if __name__ == "__main__":
    main()
