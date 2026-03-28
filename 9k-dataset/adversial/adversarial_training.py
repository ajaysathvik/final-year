"""
Adversarial Training — CTGAN 3k → FGSM → XGBoost
==================================================
Input : CTGAN-augmented train_3k.csv + test_3k.csv
Method: FGSM perturbation on numerical features only
Steps :
  1. Load CTGAN train/test split
  2. Preprocess + label encode
  3. Generate FGSM adversarial copies of training data (2× rows)
  4. Train Baseline XGB vs Adversarial XGB
  5. Evaluate on Clean + FGSM-attacked test sets
  6. Save summary_table.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
# train_test_split not needed — using pre-split CTGAN files
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts" / "xgb_ctgan_3k"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

TRAIN_PATH = ARTIFACT_DIR / "train_3k.csv"
TEST_PATH = ARTIFACT_DIR / "test_3k.csv"
TARGET_COL  = "annotation.is_fraud"
EPSILON     = 0.05   # FGSM perturbation budget
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("Loading CTGAN-augmented train/test sets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"  Train shape: {train_df.shape}")
print(f"  Test shape : {test_df.shape}")
print(f"  Train fraud distribution:\n{train_df[TARGET_COL].value_counts()}")

# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────
print("\nPreprocessing...")

# ── Drop leaked annotation columns (direct proxies of the target) ──
LEAKED_COLS = [
    "annotation.fraud_type",
    "annotation.fraud_labels.transaction_upi_fraud",
    "annotation.fraud_labels.transaction_card_fraud",
    "annotation.fraud_labels.transaction_bank_transfer",
    "annotation.fraud_labels.commerce_nondelivery",
    "annotation.fraud_labels.commerce_fake_seller",
    "annotation.fraud_labels.credential_phishing",
    "annotation.fraud_labels.social_authority_scam",
    "annotation.fraud_labels.social_urgency_scam",
    "annotation.fraud_labels.meta_victim_story",
    "annotation.fraud_labels.meta_fraud_question",
    "annotation.key_features.amount_mentioned",
    "annotation.key_features.currency",
    "annotation.key_features.impersonated_entity",
]

# Numerical columns to perturb (behavioral signals only)
NUMERIC_COLS = [
    "annotation.key_features.urgency_level",
    "annotation.psychological_tactics.urgency",
    "annotation.psychological_tactics.fear",
    "annotation.psychological_tactics.authority",
    "annotation.psychological_tactics.reward",
    "annotation.key_features.amount_normalized",
    "annotation.key_features.has_amount",
]
NUMERIC_COLS = [c for c in NUMERIC_COLS if c in train_df.columns]
print(f"  Numerical columns to perturb: {NUMERIC_COLS}")

# Combine train + test for consistent encoding, then re-split
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
n_train = len(train_df)

# Drop target + leaked columns
drop_cols = [TARGET_COL] + [c for c in LEAKED_COLS if c in combined_df.columns]
features_all = combined_df.drop(columns=drop_cols)
y_all = combined_df[TARGET_COL].values

print(f"  Kept {features_all.shape[1]} features (dropped {len(drop_cols)-1} leaked cols)")

# Encode all object/string columns with LabelEncoder
le_map = {}
for col in features_all.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    features_all[col] = features_all[col].astype(str)
    features_all[col] = le.fit_transform(features_all[col])
    le_map[col] = le

# Fill NaN with 0
features_all = features_all.fillna(0)

X_all = features_all.values.astype(np.float32)
feature_names = list(features_all.columns)

print(f"  Combined features shape: {X_all.shape}")

# ─────────────────────────────────────────────
# 3. USE PRE-SPLIT TRAIN / TEST (from CTGAN pipeline)
# ─────────────────────────────────────────────
X_train = X_all[:n_train]
y_train = y_all[:n_train]
X_test  = X_all[n_train:]
y_test  = y_all[n_train:]
print(f"\nUsing CTGAN split → Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# Map numeric col names to their indices in the feature array
numeric_indices = [feature_names.index(c) for c in NUMERIC_COLS if c in feature_names]
print(f"  Numeric feature indices to perturb: {numeric_indices}")

# ─────────────────────────────────────────────
# 4. PERTURBATION FUNCTIONS
# ─────────────────────────────────────────────

def fgsm_perturb(X, epsilon=EPSILON, numeric_idx=None):
    """
    FGSM-style perturbation on tabular data.
    Since tree models have no gradient, we use random sign perturbation
    (equivalent to the 'gradient' direction being unknown — worst case random).
    """
    X_adv = X.copy()
    noise = epsilon * np.sign(np.random.randn(X.shape[0], len(numeric_idx)))
    X_adv[:, numeric_idx] = X_adv[:, numeric_idx] + noise
    # Clip to [0, 1] range for normalized features
    X_adv[:, numeric_idx] = np.clip(X_adv[:, numeric_idx], 0.0, None)
    return X_adv.astype(np.float32)



# ─────────────────────────────────────────────
# 5. GENERATE ADVERSARIAL TRAINING DATA
# ─────────────────────────────────────────────
print("\nGenerating adversarial training examples (FGSM)...")
X_train_adv = fgsm_perturb(X_train, epsilon=EPSILON, numeric_idx=numeric_indices)

# Combine original + adversarial → 2× rows
X_train_augmented = np.vstack([X_train, X_train_adv])
y_train_augmented = np.hstack([y_train, y_train])

print(f"  Original train rows : {X_train.shape[0]}")
print(f"  Adversarial rows    : {X_train_adv.shape[0]}")
print(f"  Augmented total     : {X_train_augmented.shape[0]}  ← 2×✅")

# ─────────────────────────────────────────────
# 6. TRAIN MODELS
# ─────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    verbosity=0,
)

print("\nTraining Baseline XGBoost (original data only)...")
baseline_model = XGBClassifier(**XGB_PARAMS)
baseline_model.fit(X_train, y_train)
print("  ✅ Baseline trained")

print("Training Adversarial XGBoost (original + perturbed data)...")
adv_model = XGBClassifier(**XGB_PARAMS)
adv_model.fit(X_train_augmented, y_train_augmented)
print("  ✅ Adversarial model trained")

# ─────────────────────────────────────────────
# 7. GENERATE TEST ATTACK SETS
# ─────────────────────────────────────────────
print("\nGenerating FGSM test attack set...")
X_test_fgsm = fgsm_perturb(X_test, epsilon=EPSILON, numeric_idx=numeric_indices)

# ─────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, X_eval, y_eval, model_name, test_name):
    y_pred  = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    return {
        "Model"    : model_name,
        "Test Set" : test_name,
        "Accuracy" : round(accuracy_score(y_eval,  y_pred),  4),
        "Precision": round(precision_score(y_eval, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_eval,    y_pred,  zero_division=0), 4),
        "F1"       : round(f1_score(y_eval,        y_pred,  zero_division=0), 4),
        "AUC"      : round(roc_auc_score(y_eval,   y_proba), 4),
    }

print("\nEvaluating models...")
results = []

# Baseline evaluations
results.append(evaluate(baseline_model, X_test,      y_test, "Baseline XGB",   "Clean"))
results.append(evaluate(baseline_model, X_test_fgsm, y_test, "Baseline XGB",   "FGSM Attack"))

# Adversarial model evaluations
results.append(evaluate(adv_model, X_test,           y_test, "Adversarial XGB", "Clean"))
results.append(evaluate(adv_model, X_test_fgsm,      y_test, "Adversarial XGB", "FGSM Attack"))

# ─────────────────────────────────────────────
# 9. PRINT RESULTS
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("ADVERSARIAL TRAINING RESULTS COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)

# Highlight robustness gain
baseline_fgsm = results_df[(results_df["Model"] == "Baseline XGB") & (results_df["Test Set"] == "FGSM Attack")]["F1"].values[0]
adv_fgsm      = results_df[(results_df["Model"] == "Adversarial XGB") & (results_df["Test Set"] == "FGSM Attack")]["F1"].values[0]

print(f"\n📊 Robustness Gain (F1 — FGSM Attack):")
print(f"   Baseline XGB  : {baseline_fgsm}")
print(f"   Adversarial XGB: {adv_fgsm}")
print(f"   Δ = {round(adv_fgsm - baseline_fgsm, 4):+}")

# ─────────────────────────────────────────────
# 10. SAVE CSV + GENERATE VISUALIZATIONS
# ─────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

out_path = OUTPUT_DIR / "summary_table.csv"
results_df.to_csv(out_path, index=False)
print(f"\n✅ Results saved to: {out_path}")

METRICS   = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
MODELS    = ["Baseline XGB", "Adversarial XGB"]
TEST_SETS = ["Clean", "FGSM Attack"]
COLORS    = {
    ("Baseline XGB",   "Clean"):       "#4A90D9",
    ("Baseline XGB",   "FGSM Attack"): "#E05C5C",
    ("Adversarial XGB","Clean"):       "#27AE60",
    ("Adversarial XGB","FGSM Attack"): "#F39C12",
}

# ── Chart 1: Grouped bar — all metrics ────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("#1A1A2E")
ax.set_facecolor("#16213E")

n_metrics  = len(METRICS)
n_bars     = len(MODELS) * len(TEST_SETS)
bar_width  = 0.18
group_gap  = 0.05
x          = np.arange(n_metrics)

bar_idx = 0
legend_patches = []
for model in MODELS:
    for ts in TEST_SETS:
        row    = results_df[(results_df["Model"] == model) & (results_df["Test Set"] == ts)]
        vals   = [row[m].values[0] for m in METRICS]
        offset = (bar_idx - (n_bars - 1) / 2) * (bar_width + group_gap / n_bars)
        color  = COLORS[(model, ts)]
        bars   = ax.bar(x + offset, vals, bar_width, color=color, alpha=0.88,
                        edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                    color="white", fontweight="bold")
        legend_patches.append(mpatches.Patch(color=color, label=f"{model} | {ts}"))
        bar_idx += 1

ax.set_xticks(x)
ax.set_xticklabels(METRICS, color="white", fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", color="white", fontsize=11)
ax.set_title("Adversarial Training — CTGAN 3k | Baseline vs Adversarial XGB (FGSM)",
             color="white", fontsize=13, fontweight="bold", pad=14)
ax.tick_params(colors="white")
ax.spines[["top","right","left","bottom"]].set_color("#334155")
ax.yaxis.set_tick_params(labelcolor="white")
ax.legend(handles=legend_patches, loc="lower right", framealpha=0.25,
          labelcolor="white", fontsize=9)
ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.4)
plt.tight_layout()
p1 = OUTPUT_DIR / "chart1_all_metrics.png"
plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Chart 1 saved: {p1}")

# ── Chart 2: F1 Robustness highlight ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("#1A1A2E")
ax.set_facecolor("#16213E")

labels = ["Baseline\n(Clean)", "Baseline\n(FGSM)", "Adversarial\n(Clean)", "Adversarial\n(FGSM)"]
f1_vals = [
    results_df[(results_df["Model"]=="Baseline XGB")    & (results_df["Test Set"]=="Clean")      ]["F1"].values[0],
    results_df[(results_df["Model"]=="Baseline XGB")    & (results_df["Test Set"]=="FGSM Attack")]["F1"].values[0],
    results_df[(results_df["Model"]=="Adversarial XGB") & (results_df["Test Set"]=="Clean")      ]["F1"].values[0],
    results_df[(results_df["Model"]=="Adversarial XGB") & (results_df["Test Set"]=="FGSM Attack")]["F1"].values[0],
]
bar_colors = ["#4A90D9", "#E05C5C", "#27AE60", "#F39C12"]
bars = ax.bar(labels, f1_vals, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.55)
for bar, v in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{v:.4f}", ha="center", va="bottom", color="white",
            fontsize=11, fontweight="bold")

ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 Score", color="white", fontsize=11)
ax.set_title("F1 Score — Robustness Under FGSM Attack", color="white",
             fontsize=12, fontweight="bold", pad=12)
ax.tick_params(colors="white", labelsize=10)
ax.spines[["top","right","left","bottom"]].set_color("#334155")
ax.yaxis.set_tick_params(labelcolor="white")
ax.grid(axis="y", color="#334155", linestyle="--", alpha=0.4)
plt.tight_layout()
p2 = OUTPUT_DIR / "chart2_f1_robustness.png"
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Chart 2 saved: {p2}")

# ── Chart 3: Feature perturbation — before vs after ──────────
fig, axes = plt.subplots(1, min(3, len(numeric_indices)), figsize=(13, 4), sharey=False)
fig.patch.set_facecolor("#1A1A2E")
if len(numeric_indices) == 1:
    axes = [axes]

sample_cols = NUMERIC_COLS[:3]
for i, (ax, col) in enumerate(zip(axes, sample_cols)):
    ax.set_facecolor("#16213E")
    col_idx = feature_names.index(col)
    orig = X_train[:, col_idx]
    pert = X_train_adv[:, col_idx]
    short = col.split(".")[-1].replace("_", " ")
    ax.hist(orig, bins=20, color="#4A90D9", alpha=0.7, label="Original", density=True)
    ax.hist(pert, bins=20, color="#F39C12", alpha=0.7, label="Perturbed", density=True)
    ax.set_title(short, color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[["top","right","left","bottom"]].set_color("#334155")
    ax.yaxis.set_tick_params(labelcolor="white")
    ax.xaxis.set_tick_params(labelcolor="white")
    if i == 0:
        ax.legend(fontsize=8, framealpha=0.3, labelcolor="white")

fig.suptitle("Feature Distribution: Original vs FGSM Perturbed (Training Data)",
             color="white", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
p3 = OUTPUT_DIR / "chart3_feature_perturbation.png"
plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Chart 3 saved: {p3}")

# ── Chart 4: Feature Importance Stability ────────────────────
print("\nGenerating Feature Importance comparison...")
baseline_imp = baseline_model.feature_importances_
adv_imp      = adv_model.feature_importances_

# Sort by baseline importance for better visualization
indices = np.argsort(baseline_imp)[-15:]  # Top 15 features
top_features = [feature_names[i].split(".")[-1] for i in indices]
b_vals = baseline_imp[indices]
a_vals = adv_imp[indices]

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#1A1A2E")
ax.set_facecolor("#16213E")

y_pos = np.arange(len(top_features))
h = 0.35

ax.barh(y_pos + h/2, b_vals, h, color="#4A90D9", label="Baseline (Sensitive)", alpha=0.8)
ax.barh(y_pos - h/2, a_vals, h, color="#27AE60", label="Adversarial (Robust)", alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_features, color="white", fontsize=10)
ax.set_xlabel("Importance Score", color="white", fontsize=11)
ax.set_title("Feature Importance Stability: Baseline vs Adversarial", 
             color="white", fontsize=13, fontweight="bold", pad=15)
ax.tick_params(colors="white")
ax.spines[["top","right","left","bottom"]].set_color("#334155")
ax.legend(facecolor="#1A1A2E", labelcolor="white", edgecolor="#334155")
ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.3)

plt.tight_layout()
p4 = os.path.join(OUTPUT_DIR, "chart4_feature_importance.png")
plt.savefig(p4, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"✅ Chart 4 saved: {p4}")

print("\nDone! 🎉")
