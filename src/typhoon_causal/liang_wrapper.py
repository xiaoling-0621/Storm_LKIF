from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIANG_ROOT = PROJECT_ROOT / "LKIF"
if str(LIANG_ROOT) not in sys.path:
    sys.path.insert(0, str(LIANG_ROOT))

from causality_estimation import causal_est_matrix  # noqa: E402


@dataclass
class LiangResult:
    causality: np.ndarray
    variance: np.ndarray
    normalized_causality: np.ndarray
    significance_mask: np.ndarray


def run_liang(X: np.ndarray, dt: float, n_step: int, significance_z: float) -> LiangResult:
    causality, variance, normalized = causal_est_matrix(X, n_step=n_step, dt=dt)
    variance = np.maximum(variance, 0.0)
    significance_mask = np.abs(causality) > np.sqrt(variance) * significance_z
    return LiangResult(
        causality=causality,
        variance=variance,
        normalized_causality=normalized,
        significance_mask=significance_mask,
    )
