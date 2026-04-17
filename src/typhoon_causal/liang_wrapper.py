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


def _finalize_liang(causality: np.ndarray, variance: np.ndarray, normalized: np.ndarray, significance_z: float) -> LiangResult:
    variance = np.maximum(variance, 0.0)
    significance_mask = np.abs(causality) > np.sqrt(variance) * significance_z
    return LiangResult(
        causality=causality,
        variance=variance,
        normalized_causality=normalized,
        significance_mask=significance_mask,
    )


def run_liang(X: np.ndarray, dt: float, n_step: int, significance_z: float) -> LiangResult:
    causality, variance, normalized = causal_est_matrix(X, n_step=n_step, dt=dt)
    return _finalize_liang(causality, variance, normalized, significance_z)


def run_liang_segmented(segments: list[np.ndarray], dt: float, n_step: int, significance_z: float) -> LiangResult:
    valid_segments = [segment for segment in segments if segment.shape[1] > n_step]
    if not valid_segments:
        raise ValueError("No valid segments for segmented Liang analysis.")

    x_prev = np.concatenate([segment[:, :-n_step] for segment in valid_segments], axis=1)
    d_x = np.concatenate(
        [(segment[:, n_step:] - segment[:, :-n_step]) / (n_step * dt) for segment in valid_segments],
        axis=1,
    )

    nx, sample_count = x_prev.shape
    if sample_count <= 1:
        raise ValueError("Not enough pooled samples for segmented Liang analysis.")

    C = np.cov(x_prev)
    dC = (
        (x_prev - np.mean(x_prev, axis=1, keepdims=True))
        @ (d_x - np.mean(d_x, axis=1, keepdims=True)).T
        / (sample_count - 1)
    )

    try:
        T_pre = np.linalg.solve(C, dC)
    except np.linalg.LinAlgError:
        T_pre = np.linalg.pinv(C) @ dC

    C_diag = np.diag(1.0 / np.diag(C))
    cM = (C @ C_diag) * T_pre

    ff = np.mean(d_x, axis=1) - (np.mean(x_prev, axis=1, keepdims=True).T @ T_pre).squeeze()
    RR = d_x - ff[:, None] - T_pre.T @ x_prev

    QQ = np.sum(RR**2, axis=-1)
    total_points = sum(segment.shape[1] for segment in valid_segments)
    bb = np.sqrt(QQ * dt / (total_points - len(valid_segments) * n_step))
    dH_noise = bb**2 / 2 / np.diag(C)

    ZZ = np.sum(np.abs(cM), axis=0, keepdims=True) + np.abs(dH_noise[None, :])
    cM_Z = cM / ZZ

    N = sample_count
    NNI = np.zeros((nx, nx + 2, nx + 2))
    center = x_prev @ x_prev.T
    RS1 = np.sum(RR, axis=-1)
    RS2 = np.sum(RR**2, axis=-1)

    center = dt / bb[:, None, None] ** 2 * center[None, ...]
    top_center = (dt / bb[:, None] ** 2) @ np.sum(x_prev, axis=-1, keepdims=True).T
    right_center = (2 * dt / bb[:, None] ** 3) * (RR @ x_prev.T)

    top_left_corner = N * dt / bb**2
    top_right_corner = 2 * dt / bb**3 * RS1
    bottom_right_corner = 3 * dt / bb**4 * RS2 - N / bb**2

    NNI[:, 1:-1, 1:-1] = center
    NNI[:, 0, 1:-1] = top_center
    NNI[:, 1:-1, 0] = top_center
    NNI[:, 1:-1, -1] = right_center
    NNI[:, -1, 1:-1] = right_center
    NNI[:, 0, 0] = top_left_corner
    NNI[:, 0, -1] = top_right_corner
    NNI[:, -1, 0] = top_right_corner
    NNI[:, -1, -1] = bottom_right_corner

    inv_per_slice = list(map(np.linalg.pinv, [NNI[i] for i in range(nx)]))
    diag_per_slice = [np.diag(inv_per_slice[i])[1:-1] for i in range(nx)]
    var = (C_diag @ C) ** 2 * np.array(diag_per_slice)

    return _finalize_liang(cM, var.T, cM_Z, significance_z)
