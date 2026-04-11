import re

file_path = r'd:\ajay\final-year\9k-dataset\adversial\adversarial_training.py'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update header
text = text.replace('CTGAN 3k → FGSM → XGBoost', 'CTGAN 3k → CAA → XGBoost')
text = text.replace('Method: FGSM perturbation on numerical features only', 'Method: Constrained Adaptive Attack (CAA) perturbation')
# We can just let the Steps header update by itself
text = text.replace('FGSM adversarial copies', 'CAA adversarial copies')
text = text.replace('FGSM-attacked test sets', 'CAA-attacked test sets')

# 2. Imports
text = text.replace('from xgboost import XGBClassifier', 'from xgboost import XGBClassifier\nfrom tabularbench_caa import caa_attack')

# 3. FGSM to CAA general text
text = text.replace('FGSM perturbation budget', 'Perturbation budget')

# 4. Remove fgsm_perturb and replace generation + training
# Try simple regex or exact match
old_logic = '''# ─────────────────────────────────────────────
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
print("\\nGenerating adversarial training examples (FGSM)...")
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

print("\\nTraining Baseline XGBoost (original data only)...")
baseline_model = XGBClassifier(**XGB_PARAMS)
baseline_model.fit(X_train, y_train)
baseline_model.save_model(str(BASELINE_MODEL_PATH))
print("  ✅ Baseline trained")'''

new_logic = '''# ─────────────────────────────────────────────
# 4. TRAIN BASELINE MODEL (For CAA)
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

print("\\nTraining Baseline XGBoost (original data only)...")
baseline_model = XGBClassifier(**XGB_PARAMS)
baseline_model.fit(X_train, y_train)
baseline_model.save_model(str(BASELINE_MODEL_PATH))
print("  ✅ Baseline trained")

# ─────────────────────────────────────────────
# 5. GENERATE ADVERSARIAL TRAINING DATA (CAA)
# ─────────────────────────────────────────────
print("\\nGenerating adversarial training examples (CAA)...")
X_train_adv, _, _ = caa_attack(X_train, y_train, baseline_model, feature_names, epsilon=EPSILON)

# Combine original + adversarial → 2× rows
X_train_augmented = np.vstack([X_train, X_train_adv])
y_train_augmented = np.hstack([y_train, y_train])

print(f"  Original train rows : {X_train.shape[0]}")
print(f"  Adversarial rows    : {X_train_adv.shape[0]}")
print(f"  Augmented total     : {X_train_augmented.shape[0]}  ← 2×✅")

# ─────────────────────────────────────────────
# 6. TRAIN ADVERSARIAL MODEL
# ─────────────────────────────────────────────'''

text = text.replace(old_logic, new_logic)

# Replace FGSM with CAA throughout
text = text.replace('X_test_fgsm', 'X_test_caa')
text = text.replace('FGSM Attack', 'CAA Attack')
text = text.replace('FGSM test attack set', 'CAA test attack set')
text = text.replace('FGSM test attack', 'CAA test attack')
text = text.replace('fgsm_perturb(X_test, epsilon=EPSILON, numeric_idx=numeric_indices)', 'caa_attack(X_test, y_test, baseline_model, feature_names, epsilon=EPSILON)[0]')
text = text.replace('baseline_fgsm', 'baseline_caa')
text = text.replace('adv_fgsm', 'adv_caa')
text = text.replace('FGSM Perturbed', 'CAA Perturbed')
text = text.replace('(FGSM', '(CAA')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print('Update successful')
