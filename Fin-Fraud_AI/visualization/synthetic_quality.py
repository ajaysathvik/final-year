import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ORIG_PATH  = os.path.join(ROOT, "original_dataset", "final1.csv")
CTGAN_PATH = os.path.join(ROOT, "ctgan", "ctgan_balanced_data.csv")
ADV_PATH   = os.path.join(ROOT, "adversarial_training", "adv_balanced_data.csv")


def load_numeric(filepath):
    df = pd.read_csv(filepath)
    for col in ["title", "body", "is_fraud"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df


def align_cols(ref, target):
    for c in set(ref.columns) - set(target.columns):
        target[c] = 0
    return target[ref.columns]


def preprocess(X_real, X_synth):
    imputer = SimpleImputer(strategy="mean")
    scaler  = StandardScaler()
    X_r = scaler.fit_transform(imputer.fit_transform(X_real)).astype(np.float32)
    X_s = scaler.transform(imputer.transform(X_synth)).astype(np.float32)
    return X_r, X_s


def compute_fid(X_real, X_synth):
    """Tabular FID: ||mu_r - mu_s||^2 + Tr(cov_r + cov_s - 2*sqrt(cov_r @ cov_s))"""
    mu_r, mu_s = X_real.mean(axis=0), X_synth.mean(axis=0)
    cov_r = np.cov(X_real, rowvar=False)
    cov_s = np.cov(X_synth, rowvar=False)

    diff = mu_r - mu_s
    cov_sqrt, _ = sqrtm(cov_r @ cov_s, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    return float(diff @ diff + np.trace(cov_r + cov_s - 2 * cov_sqrt))


def compute_diversity(X_synth):
    """Mean std per feature — higher = more diverse synthetic data."""
    return float(np.mean(np.std(X_synth, axis=0)))


def compute_coverage(X_real, X_synth, k=5, radius_factor=0.5):
    """Fraction of real samples with at least one synthetic neighbour within radius."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_real)
    dists, _ = nbrs.kneighbors(X_real)
    radius = np.median(dists[:, -1]) * radius_factor

    synth_nbrs = NearestNeighbors(radius=radius).fit(X_synth)
    covered = synth_nbrs.radius_neighbors(X_real, return_distance=False)
    return float(np.mean([len(c) > 0 for c in covered]))


# ── Load data ──
print("=" * 55)
print("  Synthetic Data Quality Metrics")
print("=" * 55)
print("\nLoading datasets...")
df_orig  = load_numeric(ORIG_PATH)
df_ctgan = load_numeric(CTGAN_PATH)
df_adv   = load_numeric(ADV_PATH)

df_ctgan = align_cols(df_orig, df_ctgan)
df_adv   = align_cols(df_orig, df_adv)

# Extract only the synthetic rows (added beyond original length)
n_orig = len(pd.read_csv(ORIG_PATH))
synth_ctgan = df_ctgan.iloc[n_orig:].reset_index(drop=True)
synth_adv   = df_adv.iloc[n_orig:].reset_index(drop=True)
print(f"Synthetic rows — CTGAN: {len(synth_ctgan)}  |  Adv-CTGAN: {len(synth_adv)}")

X_real, X_ctgan = preprocess(df_orig, synth_ctgan)
X_real2, X_adv  = preprocess(df_orig, synth_adv)

# ── Compute metrics ──
results = []
for label, X_s, X_r in [("CTGAN", X_ctgan, X_real), ("Adv-CTGAN", X_adv, X_real2)]:
    print(f"\nComputing metrics for {label}...")
    fid = compute_fid(X_r, X_s)
    div = compute_diversity(X_s)
    cov = compute_coverage(X_r, X_s)
    print(f"  FID:       {fid:.4f}  (lower = better)")
    print(f"  Diversity: {div:.4f}  (higher = better)")
    print(f"  Coverage:  {cov:.4f}  (higher = better)")
    results.append({"Model": label, "FID": round(fid, 4),
                    "Diversity": round(div, 4), "Coverage": round(cov, 4)})

results_df = pd.DataFrame(results)
print(f"\n{'='*55}")
print(results_df.to_string(index=False))

csv_path = os.path.join(OUT_DIR, "synthetic_quality_metrics.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nSaved -> {csv_path}")

# ── Chart ──
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col, label in zip(axes,
    ["FID", "Diversity", "Coverage"],
    ["FID (lower is better)", "Diversity (higher is better)", "Coverage (higher is better)"]):
    sns.barplot(data=results_df, x="Model", y=col, palette="Set2", ax=ax)
    ax.set_title(label, fontweight="bold")
    ax.set_xlabel("")
plt.suptitle("Synthetic Data Quality: CTGAN vs Adv-CTGAN", fontsize=14, fontweight="bold")
plt.tight_layout()
out_png = os.path.join(OUT_DIR, "synthetic_quality_comparison.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_png}")
plt.close()

print("\nDone!")
