"""
_utils.py
=========
Small shared helpers used across the individual scorer modules.
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def safe_score(fn, smi, default=np.nan):
    """Call fn(smi) safely, returning `default` on any error."""
    try:
        return fn(smi)
    except Exception:
        return default