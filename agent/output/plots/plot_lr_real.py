import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import sys
import os

# Ensure the pipeline's config allows imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import DATASET_PATH, NUMERIC_FEATURE_COLS, TARGET_COL

def main():
    print(f"Loading data from {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    
    # Filter features
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(-1).astype(int)
    df = df[df[TARGET_COL].isin([0, 1])].copy()
    
    X = df[NUMERIC_FEATURE_COLS].copy()
    # Replace any non numeric or inf
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
    y = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0).astype(int)
    
    # Weight
    scale_pos_weight = max(1, int((y == 0).sum() / max(1, (y == 1).sum())))

    # define learning rates
    learning_rates = {
        'very high learning rate': {'lr': 2.5, 'color': 'orange'},
        'high learning rate': {'lr': 0.8, 'color': 'green'},
        'good learning rate': {'lr': 0.1, 'color': 'red'},
        'low learning rate': {'lr': 0.001, 'color': 'blue'}
    }
    
    plt.figure(figsize=(8, 6))
    
    epochs = 100
    
    # Train and plot
    for label, params in learning_rates.items():
        print(f"Training XGBoost with {label} ({params['lr']})...")
        model = xgb.XGBClassifier(
            n_estimators=epochs,
            max_depth=6,
            learning_rate=params['lr'],
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
        
        # We pass eval_set to capture the logloss per iteration
        model.fit(X, y, eval_set=[(X, y)], verbose=False)
        
        # Extract losses
        # evals_result() returns dict like {'validation_0': {'logloss': [...]}}
        results = model.evals_result()
        logloss = results['validation_0']['logloss']
        
        # Plot
        plt.plot(range(1, epochs + 1), logloss, color=params['color'], linewidth=2, label=label)

    # Customize the plot to look like the theoretical graph
    ax = plt.gca()
    
    # Position text labels
    ax.text(epochs*0.05, 0.6, 'very high learning rate', color='orange', fontsize=14)
    ax.text(epochs*0.6, 0.45, 'low learning rate', color='blue', fontsize=14)
    ax.text(epochs*0.5, 0.35, 'high learning rate', color='green', fontsize=14)
    ax.text(epochs*0.5, 0.20, 'good learning rate', color='red', fontsize=14)
    
    ax.set_ylim(0, max(logloss)*1.5)
    ax.set_xlim(0, epochs)
    
    # Customize axes to mimic hand-drawn axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Add arrows
    ax.plot((epochs), (0), ls="", marker=">", ms=10, color="k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (ax.get_ylim()[1]), ls="", marker="^", ms=10, color="k", transform=ax.get_xaxis_transform(), clip_on=False)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlabel('epoch', fontsize=16, loc='right')
    ax.set_ylabel('loss', fontsize=16, loc='top', rotation=0, labelpad=-40)
    
    plt.tight_layout()
    output_path = '/home/norm/Projects/Reddit-dataset/agent/output/plots/learning_rate_plot_real.png'
    plt.savefig(output_path, dpi=300)
    print(f"Real data plot saved to {output_path}")

if __name__ == "__main__":
    main()
