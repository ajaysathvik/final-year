import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(REPORTS_DIR / "mplconfig")
(REPORTS_DIR / "mplconfig").mkdir(parents=True, exist_ok=True)

baseline_path = ARTIFACTS_DIR / "xgb_balanced_train_imbalanced_test" / "metrics.json"
raw_path = ARTIFACTS_DIR / "xgb_raw_imbalanced" / "metrics.json"

with open(baseline_path, 'r', encoding='utf-8') as f:
    baseline_data = json.load(f)

with open(raw_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

def extract_metrics(data):
    cr = data["classification_report"]
    return {
        "Minority (Not Fraud)\nPrecision": cr["0"]["precision"],
        "Minority (Not Fraud)\nRecall": cr["0"]["recall"],
        "Minority (Not Fraud)\nF1-Score": cr["0"]["f1-score"],
        "Majority (Fraud)\nF1-Score": cr["1"]["f1-score"],
        "Overall Accuracy": data["accuracy"]
    }

b_metrics = extract_metrics(baseline_data)
r_metrics = extract_metrics(raw_data)

labels = list(b_metrics.keys())
b_values = [b_metrics[k] for k in labels]
r_values = [r_metrics[k] for k in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
rects1 = ax.bar(x - width/2, b_values, width, label='Baseline (1:1 Ratio)', color='#e74c3c')
rects2 = ax.bar(x + width/2, r_values, width, label='Raw Imbalanced Data', color='#3498db')

ax.set_ylabel('Metric Score', fontsize=12)
ax.set_title('Performance Comparison: XGBoost Baseline vs Raw Imbalanced', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1.1)

ax.legend(loc='upper right', framealpha=0.9)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
output_path = REPORTS_DIR / "baseline_vs_raw_comparison.png"
plt.savefig(output_path, dpi=150)
print(f"Saved plot successfully to: {output_path}")
