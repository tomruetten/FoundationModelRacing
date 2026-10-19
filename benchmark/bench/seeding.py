"""Reproducible randomness for sampling models (reasoning tokens and diffusion noise are random)."""

import random

import numpy as np


def sample_seed(seed: int, sample_idx: int) -> int:
    """Seed for one sample: stable per (run seed, sample index), independent of what ran before.

    Seeding every sample separately makes a sample's randomness independent of warm-up, of `max_samples` and of
    other samples, so partial reruns and runs with different K see the same seed stream.
    """
    return (int(seed) * 1_000_003 + int(sample_idx) * 7919) % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:  # CPU-only environment without torch
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
