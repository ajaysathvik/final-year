import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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


# ── Load data ──
print("=" * 55)
print("  Distribution Visualization (PCA + t-SNE)")
print("=" * 55)
print("\nLoading datasets...")
df_orig  = load_numeric(ORIG_PATH)
df_ctgan = load_numeric(CTGAN_PATH)
df_adv   = load_numeric(ADV_PATH)

df_ctgan = align_cols(df_orig, df_ctgan)
df_adv   = align_cols(df_orig, df_adv)

n_orig = len(pd.read_csv(ORIG_PATH))
synth_ctgan = df_ctgan.iloc[n_orig:].reset_index(drop=True)
synth_adv   = df_adv.iloc[n_orig:].reset_index(drop=True)

# Sample to keep plots readable
N = min(500, len(df_orig), len(synth_ctgan), len(synth_adv))
real_sample  = df_orig.sample(N, random_state=42)
ctgan_sample = synth_ctgan.sample(N, random_state=42)
adv_sample   = synth_adv.sample(N, random_state=42)

combined = pd.concat([real_sample, ctgan_sample, adv_sample], ignore_index=True)
labels   = ["Real"] * N + ["CTGAN Synthetic"] * N + ["Adv-CTGAN Synthetic"] * N

# Preprocess
imputer = SimpleImputer(strategy="mean")
scaler  = StandardScaler()
X = scaler.fit_transform(imputer.fit_transform(combined)).astype(np.float32)

PALETTE = {"Real": "#2196F3", "CTGAN Synthetic": "#FF9800", "Adv-CTGAN Synthetic": "#4CAF50"}

# ── PCA ──
print("\nRunning PCA...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var = pca.explained_variance_ratio_

df_pca = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Source": labels})
plt.figure(figsize=(9, 7))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Source",
                palette=PALETTE, alpha=0.6, s=30, linewidth=0)
plt.title("PCA — Real vs Synthetic Data Distributions", fontsize=13, fontweight="bold")
plt.xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
plt.legend(title="Source")
plt.tight_layout()
out_pca = os.path.join(OUT_DIR, "data_distribution_pca.png")
plt.savefig(out_pca, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_pca}")
plt.close()

# ── t-SNE ──
print("Running t-SNE (may take 30-60s)...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)

df_tsne = pd.DataFrame({"Dim1": X_tsne[:, 0], "Dim2": X_tsne[:, 1], "Source": labels})
plt.figure(figsize=(9, 7))
sns.scatterplot(data=df_tsne, x="Dim1", y="Dim2", hue="Source",
                palette=PALETTE, alpha=0.6, s=30, linewidth=0)
plt.title("t-SNE — Real vs Synthetic Data Distributions", fontsize=13, fontweight="bold")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Source")
plt.tight_layout()
out_tsne = os.path.join(OUT_DIR, "data_distribution_tsne.png")
plt.savefig(out_tsne, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_tsne}")
plt.close()

print("\nDone!")
