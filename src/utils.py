"""Shared helpers: seeding, run folders, and plots."""

import random
from datetime import datetime
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Make runs reproducible by fixing every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch is installed in M3; seed it only if it is available.
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU-only machines


def make_run_dir(experiment_name, base="runs"):
    """Create runs/<experiment_name>_<timestamp>/ and return its path."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{experiment_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def plot_loss_curves(history, path):
    """Training and validation loss per epoch (history = list of per-epoch dicts)."""
    import matplotlib
    matplotlib.use("Agg")  # draw to files, no window needed
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], label="validation")
    best = min(history, key=lambda h: h["val_loss"])
    ax.axvline(best["epoch"], color="grey", linestyle="--", linewidth=1, label=f"best epoch ({best['epoch']})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardised targets)")
    ax.set_title("Loss curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pred_vs_actual(pred, true, designs, target_names, path):
    """One scatter plot per target: predicted vs actual (real units), coloured by design."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    designs = np.asarray(designs)
    fig, axes = plt.subplots(1, len(target_names), figsize=(5 * len(target_names), 4.5))
    axes = np.atleast_1d(axes)
    for j, (ax, name) in enumerate(zip(axes, target_names)):
        for d in sorted(set(designs)):
            m = designs == d
            ax.scatter(true[m, j], pred[m, j], s=6, alpha=0.5, label=d)
        lo, hi = min(true[:, j].min(), pred[:, j].min()), max(true[:, j].max(), pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)  # perfect predictions lie on this line
        ax.set_xlabel(f"actual {name}")
        ax.set_ylabel(f"predicted {name}")
        ax.set_title(f"{name}: predicted vs actual")
    axes[0].legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
