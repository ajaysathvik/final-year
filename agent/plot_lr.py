import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_plot():
    epochs = np.linspace(0, 10, 100)
    
    # good learning rate (red)
    good_lr = 0.1 + 0.9 * np.exp(-0.8 * epochs)
    
    # low learning rate (blue)
    low_lr = 1.0 - 0.06 * epochs
    
    # high learning rate (green)
    # it drops fast but has some small wiggles and plateaus higher
    high_lr = 0.35 + 0.65 * np.exp(-1.5 * epochs) + 0.01 * np.sin(2 * epochs)
    
    # very high learning rate (orange)
    # drops slightly then explodes
    epochs_vh = np.linspace(0, 4, 100)
    very_high_lr = 1.0 - 0.3 * epochs_vh + 0.02 * np.exp(1.2 * epochs_vh)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(epochs_vh, very_high_lr, color='orange', linewidth=2, label='very high learning rate')
    ax.plot(epochs, low_lr, color='blue', linewidth=2, label='low learning rate')
    ax.plot(epochs, high_lr, color='green', linewidth=2, label='high learning rate')
    ax.plot(epochs, good_lr, color='red', linewidth=2, label='good learning rate')
    
    # Add labels text instead of legend to match the image better
    ax.text(3, 1.3, 'very high learning rate', color='orange', fontsize=14)
    ax.text(3, 0.7, 'low learning rate', color='blue', fontsize=14)
    ax.text(6, 0.45, 'high learning rate', color='green', fontsize=14)
    ax.text(0.5, 0.15, 'good learning rate', color='red', fontsize=14)
    
    ax.set_ylim(0, 1.5)
    ax.set_xlim(0, 10)
    
    # Customize axes to mimic the hand-drawn axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Add arrows to axes
    ax.plot((10), (0), ls="", marker=">", ms=10, color="k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1.5), ls="", marker="^", ms=10, color="k", transform=ax.get_xaxis_transform(), clip_on=False)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlabel('epoch', fontsize=16, loc='right')
    ax.set_ylabel('loss', fontsize=16, loc='top', rotation=0, labelpad=-40)
    
    plt.tight_layout()
    output_path = Path('agent/output/plots/learning_rate_plot.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    generate_plot()
