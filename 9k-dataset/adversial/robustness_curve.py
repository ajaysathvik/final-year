"""
Robustness Curve Generation (Varying Epsilon)
=============================================
This script verifies the robustness of the adversarially trained model
by testing it against FGSM attacks of increasing strength (epsilon).
It generates a "Robustness Curve" plot.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
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
TRAIN_EPSILON = 0.05   # The epsilon used to train the robust model
TEST_EPSILONS = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30] 
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA & PREPROCESS (Same as before)
# ─────────────────────────────────────────────
print("Loading CTGAN-augmented train/test sets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

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

combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
n_train = len(train_df)

drop_cols = [TARGET_COL] + [c for c in LEAKED_COLS if c in combined_df.columns]
features_all = combined_df.drop(columns=drop_cols)
y_all = combined_df[TARGET_COL].values

for col in features_all.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    features_all[col] = features_all[col].astype(str)
    features_all[col] = le.fit_transform(features_all[col])

features_all = features_all.fillna(0)
X_all = features_all.values.astype(np.float32)
feature_names = list(features_all.columns)

X_train, y_train = X_all[:n_train], y_all[:n_train]
X_test,  y_test  = X_all[n_train:], y_all[n_train:]
numeric_indices = [feature_names.index(c) for c in NUMERIC_COLS if c in feature_names]

# ─────────────────────────────────────────────
# 2. FGSM PERTURBATION FUNCTION
# ─────────────────────────────────────────────
def fgsm_perturb(X, epsilon, numeric_idx=None):
    if epsilon == 0.0:
        return X.copy()
    X_adv = X.copy()
    noise = epsilon * np.sign(np.random.randn(X.shape[0], len(numeric_idx)))
    X_adv[:, numeric_idx] = X_adv[:, numeric_idx] + noise
    X_adv[:, numeric_idx] = np.clip(X_adv[:, numeric_idx], 0.0, None)
    return X_adv.astype(np.float32)

# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────
XGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_SEED, verbosity=0)

print("\nTraining Baseline XGBoost...")
baseline_model = XGBClassifier(**XGB_PARAMS)
baseline_model.fit(X_train, y_train)

print(f"Training Adversarial XGBoost (Train Epsilon = {TRAIN_EPSILON})...")
X_train_adv = fgsm_perturb(X_train, epsilon=TRAIN_EPSILON, numeric_idx=numeric_indices)
X_train_augmented = np.vstack([X_train, X_train_adv])
y_train_augmented = np.hstack([y_train, y_train])

adv_model = XGBClassifier(**XGB_PARAMS)
adv_model.fit(X_train_augmented, y_train_augmented)

# ─────────────────────────────────────────────
# 4. EVALUATE ACROSS EPSILONS
# ─────────────────────────────────────────────
print("\nEvaluating across varying attack strengths (Epsilon)...")
baseline_f1_scores = []
adv_f1_scores = []

for eps in TEST_EPSILONS:
    # Generate test attack for this epsilon
    X_test_attacked = fgsm_perturb(X_test, epsilon=eps, numeric_idx=numeric_indices)
    
    # Evaluate Baseline
    base_pred = baseline_model.predict(X_test_attacked)
    base_f1 = f1_score(y_test, base_pred, zero_division=0)
    baseline_f1_scores.append(base_f1)
    
    # Evaluate Adversarial
    adv_pred = adv_model.predict(X_test_attacked)
    adv_f1 = f1_score(y_test, adv_pred, zero_division=0)
    adv_f1_scores.append(adv_f1)
    
    print(f"  ε={eps:<4} | Baseline F1: {base_f1:.4f} | Adversarial F1: {adv_f1:.4f}")

# ─────────────────────────────────────────────
# 5. PLOT ROBUSTNESS CURVE
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("#1A1A2E")
ax.set_facecolor("#16213E")

ax.plot(TEST_EPSILONS, baseline_f1_scores, marker='o', linewidth=2, markersize=6, 
        color="#E05C5C", label="Baseline XGBoost")
ax.plot(TEST_EPSILONS, adv_f1_scores, marker='s', linewidth=2, markersize=6, 
        color="#27AE60", label=f"Adversarial XGBoost (Trained at ε={TRAIN_EPSILON})")

# Mark the training epsilon with a vertical line
ax.axvline(x=TRAIN_EPSILON, color="#F39C12", linestyle="--", alpha=0.7, 
           label=f"Training ε ({TRAIN_EPSILON})")

ax.set_xlabel("Attack Strength (FGSM Epsilon)", color="white", fontsize=12, fontweight="bold")
ax.set_ylabel("Model Performance (F1 Score)", color="white", fontsize=12, fontweight="bold")
ax.set_title("Robustness Verification Curve: Model Degradation Under Attack", 
             color="white", fontsize=14, fontweight="bold", pad=15)

ax.set_ylim(min(min(baseline_f1_scores), min(adv_f1_scores)) - 0.05, 1.02)
ax.tick_params(colors="white", labelsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#334155")
ax.grid(color="#334155", linestyle="--", alpha=0.5)

ax.legend(facecolor="#1A1A2E", edgecolor="#334155", labelcolor="white", fontsize=10)

plt.tight_layout()
out_plot = OUTPUT_DIR / "robustness_curve.png"
plt.savefig(out_plot, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n✅ Robustness Curve saved to: {out_plot}")
