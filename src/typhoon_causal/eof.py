from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class EOFModel:
    channel_name: str
    pca: PCA
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    selected_count: int
    latitudes: np.ndarray
    longitudes: np.ndarray
    valid_mask: np.ndarray


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


def _select_component_count(
    explained_variance_ratio: np.ndarray,
    eof_mode: str,
    pc_k: int,
    variance_threshold: float,
) -> int:
    if len(explained_variance_ratio) == 0:
        return 0
    cumulative = np.cumsum(explained_variance_ratio)
    if eof_mode == "fixed_k":
        return min(pc_k, len(explained_variance_ratio))
    if eof_mode == "variance_threshold":
        clipped_threshold = min(max(variance_threshold, 0.0), 1.0)
        return min(int(np.searchsorted(cumulative, clipped_threshold, side="left") + 1), len(explained_variance_ratio))
    raise ValueError(f"Unsupported eof_mode: {eof_mode}")


def fit_channel_pca_model(
    channel_name: str,
    channel_data: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    eof_mode: str,
    pc_k: int,
    variance_threshold: float,
) -> EOFModel:
    time_count = channel_data.shape[0]
    flattened = channel_data.reshape(time_count, -1)
    valid_mask = np.isfinite(flattened).all(axis=0)
    flattened = flattened[:, valid_mask]
    max_components = min(flattened.shape[0], flattened.shape[1])
    pca = PCA(n_components=max_components)
    pca.fit(flattened)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    selected_count = _select_component_count(explained, eof_mode, pc_k, variance_threshold)
    return EOFModel(
        channel_name=channel_name,
        pca=pca,
        explained_variance_ratio=explained,
        cumulative_explained_variance=cumulative,
        selected_count=selected_count,
        latitudes=latitudes,
        longitudes=longitudes,
        valid_mask=valid_mask,
    )


def transform_channel_data(channel_data: np.ndarray, eof_model: EOFModel) -> EOFResult:
    time_count = channel_data.shape[0]
    flattened = channel_data.reshape(time_count, -1)
    flattened = flattened[:, eof_model.valid_mask]
    scores = eof_model.pca.transform(flattened)
    selected_count = min(eof_model.selected_count, scores.shape[1], eof_model.pca.components_.shape[0])
    selected_scores = scores[:, :selected_count]
    selected_components = eof_model.pca.components_[:selected_count]
    return EOFResult(
        channel_name=eof_model.channel_name,
        explained_variance_ratio=eof_model.explained_variance_ratio,
        cumulative_explained_variance=eof_model.cumulative_explained_variance,
        scores=scores,
        selected_scores=selected_scores,
        selected_components=selected_components,
        latitudes=eof_model.latitudes,
        longitudes=eof_model.longitudes,
        selected_count=selected_count,
        valid_mask=eof_model.valid_mask,
    )


def fit_channel_pca(
    channel_name: str,
    channel_data: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    eof_mode: str,
    pc_k: int,
    variance_threshold: float,
) -> EOFResult:
    model = fit_channel_pca_model(
        channel_name=channel_name,
        channel_data=channel_data,
        latitudes=latitudes,
        longitudes=longitudes,
        eof_mode=eof_mode,
        pc_k=pc_k,
        variance_threshold=variance_threshold,
    )
    return transform_channel_data(channel_data, model)


def save_eof_plots(
    eof_result: EOFResult,
    output_dir: Path,
    eof_map_components: int,
    file_prefix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{file_prefix}_" if file_prefix else ""

    max_bar = min(20, len(eof_result.explained_variance_ratio))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(1, max_bar + 1), eof_result.explained_variance_ratio[:max_bar])
    ax.set_title(f"{eof_result.channel_name} explained variance ratio")
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance ratio")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}{eof_result.channel_name}_explained_variance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(eof_result.cumulative_explained_variance) + 1), eof_result.cumulative_explained_variance)
    ax.set_title(f"{eof_result.channel_name} cumulative explained variance")
    ax.set_xlabel("PC")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}{eof_result.channel_name}_cumulative_explained_variance.png", dpi=150)
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
        fig.savefig(output_dir / f"{prefix}{eof_result.channel_name}_eof_{idx + 1}.png", dpi=150)
        plt.close(fig)
