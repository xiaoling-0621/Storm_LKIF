from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class EOFResult:
    channel_name: str
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    scores: np.ndarray
    selected_scores: np.ndarray
    selected_components: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    selected_count: int
    valid_mask: np.ndarray


def fit_channel_pca(
    channel_name: str,
    channel_data: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    eof_mode: str,
    pc_k: int,
    variance_threshold: float,
) -> EOFResult:
    time_count = channel_data.shape[0]
    flattened = channel_data.reshape(time_count, -1)
    valid_mask = np.isfinite(flattened).all(axis=0)
    flattened = flattened[:, valid_mask]
    max_components = min(flattened.shape[0], flattened.shape[1])
    pca = PCA(n_components=max_components)
    scores = pca.fit_transform(flattened)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    if eof_mode == "fixed_k":
        selected_count = min(pc_k, scores.shape[1])
    elif eof_mode == "variance_threshold":
        selected_count = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    else:
        raise ValueError(f"Unsupported eof_mode: {eof_mode}")

    selected_scores = scores[:, :selected_count]
    selected_components = pca.components_[:selected_count]
    return EOFResult(
        channel_name=channel_name,
        explained_variance_ratio=explained,
        cumulative_explained_variance=cumulative,
        scores=scores,
        selected_scores=selected_scores,
        selected_components=selected_components,
        latitudes=latitudes,
        longitudes=longitudes,
        selected_count=selected_count,
        valid_mask=valid_mask,
    )


def save_eof_plots(
    eof_result: EOFResult,
    output_dir: Path,
    eof_map_components: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    max_bar = min(20, len(eof_result.explained_variance_ratio))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(1, max_bar + 1), eof_result.explained_variance_ratio[:max_bar])
    ax.set_title(f"{eof_result.channel_name} explained variance ratio")
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance ratio")
    fig.tight_layout()
    fig.savefig(output_dir / f"{eof_result.channel_name}_explained_variance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(eof_result.cumulative_explained_variance) + 1), eof_result.cumulative_explained_variance)
    ax.set_title(f"{eof_result.channel_name} cumulative explained variance")
    ax.set_xlabel("PC")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / f"{eof_result.channel_name}_cumulative_explained_variance.png", dpi=150)
    plt.close(fig)

    map_count = min(eof_map_components, eof_result.selected_components.shape[0])
    if map_count <= 0:
        return

    lat_count = len(eof_result.latitudes)
    lon_count = len(eof_result.longitudes)
    valid_map = np.full((lat_count * lon_count,), np.nan, dtype=np.float64)

    for idx in range(map_count):
        component = eof_result.selected_components[idx]
        eof_map = valid_map.copy()
        eof_map[eof_result.valid_mask] = component
        eof_map = eof_map.reshape(lat_count, lon_count)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(eof_map, aspect="auto", origin="lower")
        ax.set_title(f"{eof_result.channel_name} EOF {idx + 1}")
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(output_dir / f"{eof_result.channel_name}_eof_{idx + 1}.png", dpi=150)
        plt.close(fig)
