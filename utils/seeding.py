"""Shared deterministic seeding and boolean command-line argument helpers."""

import argparse

import numpy as np
import torch


def set_seed_for_step(base_seed, *args_indices):
    """Derive a deterministic seed from ``base_seed`` and loop indices.

    Structure: ``base + idx1*100000 + idx2*1000 + idx3*10 + idx4``.
    """
    # Simple hashing strategy to mix indices
    current_seed = base_seed
    multipliers = [100000, 1000, 10, 1]
    for i, idx in enumerate(args_indices):
        if i < len(multipliers):
            current_seed += int(idx) * multipliers[i]
        else:
            current_seed += int(idx)  # Fallback

    # Cap to valid 32-bit integer range if needed, though python handles large ints.
    # torch.manual_seed expects int64 (long), so it's fine.
    current_seed = current_seed % (2**32 - 1)

    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)
    return current_seed


def str2bool(v):
    """Argparse boolean parser accepting true/false/1/0/yes/no (case-insensitive)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    v = str(v).lower()
    if v in ('true', 't', '1', 'yes', 'y'):
        return True
    if v in ('false', 'f', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected (true/false or 1/0).')
