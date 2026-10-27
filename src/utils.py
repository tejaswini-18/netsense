"""Shared helpers: seeding now; logging and plotting helpers are added in M5."""

import random

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
