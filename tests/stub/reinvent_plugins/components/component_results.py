"""Minimal stand-in for REINVENT4's `ComponentResults` (see add_tag.py)."""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ComponentResults:
    """One float array per endpoint, as a scoring component returns them."""

    scores: List[np.ndarray]
    uncertainty: List[np.ndarray] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
