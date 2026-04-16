from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .data_utils import (
    align_storm,
    discover_storm_records,
    inspect_era_sample,
    infer_intensity_columns,
    load_channel_time_series,
    select_storm_ids,
    summarize_records,
)
from .eof import fit_channel_pca, save_eof_plots
from .liang_wrapper import run_liang


@dataclass
class BaselineConfig:
    era5_dir: Path
    intensity_dir: Path
    intensity_split: str
    storm_select_mode: str
    selected_storm_id: str | None
    random_n: int
    random_seed: int
    eof_mode: str
    pc_k: int
    variance_threshold: float
    liang_dt: float
    liang_n_step: int
    liang_significance_z: float
    min_storm_length: int
    max_eof_plots: int
    eof_map_components: int
    results_dir: Path

    @classmethod
    def from_yaml(cls, path: Path) -> "BaselineConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["era5_dir"] = Path(raw["era5_dir"])
        raw["intensity_dir"] = Path(raw["intensity_dir"])
        raw["results_dir"] = Path(raw["results_dir"])
        return cls(**raw)


def build_data_summary(config: BaselineConfig) -> str:
    records = discover_storm_records(config.era5_dir, config.intensity_dir, config.intensity_split)
    summary_df = summarize_records(records)
    if summary_df.empty:
        raise RuntimeError("No matched storm records found.")

    sample_storm_id = summary_df.iloc[0]["storm_id"]
    sample_record = records[sample_storm_id]
    sample_era_path = next(iter(sample_record.era_files.values()))
    era_meta = inspect_era_sample(sample_era_path)
    intensity_meta = infer_intensity_columns(sample_record.intensity)

    top_rows = summary_df.head(20)
    header = "| storm_id | era_time_count | intensity_time_count | intensity_time_min | intensity_time_max |"
    divider = "| --- | ---: | ---: | --- | --- |"
    body = [
        "| "
        + " | ".join(
            [
                str(row["storm_id"]),
                str(row["era_time_count"]),
                str(row["intensity_time_count"]),
                str(row["intensity_time_min"]),
                str(row["intensity_time_max"]),
            ]
        )
        + " |"
        for _, row in top_rows.iterrows()
    ]
    top_lines = "\n".join([header, divider, *body])
    return "\n".join(
        [
            "# Data Summary",
            "",
            f"- Matched storms: {len(records)}",
            f"- ERA5 sample file: `{sample_era_path}`",
            f"- Sample storm_id format: `{sample_storm_id}`",
            f"- Intensity timestamp column: `{intensity_meta['timestamp_column']}`",
            f"- Inferred wind column: `{intensity_meta['wind_column']}`",
            f"- Inferred pressure column: `{intensity_meta['pressure_column']}`",
            f"- Inference note: {intensity_meta['note']}",
            "",
            "## ERA5 Structure",
            "",
            f"- Dims: `{era_meta['dims']}`",
            f"- Variables: `{list(era_meta['variables'].keys())}`",
            f"- Expanded channels: `{era_meta['channels']}`",
            "",
            "## Storm Counts",
            "",
            top_lines,
            "",
        ]
    )


def build_feature_matrix(aligned_storm, config: BaselineConfig, eof_output_dir: Path):
    channel_data = load_channel_time_series(aligned_storm)
    eof_results = {}
    variable_names = ["intensity_wind"]
    rows = [aligned_storm.intensity_values.astype(np.float64)]

    for idx, channel_name in enumerate(sorted(channel_data.keys())):
        channel_info = channel_data[channel_name]
        eof_result = fit_channel_pca(
            channel_name=channel_name,
            channel_data=channel_info["data"],
            latitudes=channel_info["latitude"],
            longitudes=channel_info["longitude"],
            eof_mode=config.eof_mode,
            pc_k=config.pc_k,
            variance_threshold=config.variance_threshold,
        )
        eof_results[channel_name] = eof_result
        if idx < config.max_eof_plots:
            save_eof_plots(eof_result, eof_output_dir, config.eof_map_components)
        for pc_index in range(eof_result.selected_count):
            variable_names.append(f"{channel_name}_pc{pc_index + 1}")
            rows.append(eof_result.selected_scores[:, pc_index])

    X = np.vstack(rows)
    feature_df = pd.DataFrame(X.T, columns=variable_names, index=aligned_storm.timestamps)
    return X, variable_names, feature_df, eof_results


def causality_to_frame(storm_id: str, variable_names: list[str], liang_result) -> pd.DataFrame:
    rows = []
    for source_idx, source_name in enumerate(variable_names):
        for target_idx, target_name in enumerate(variable_names):
            rows.append(
                {
                    "storm_id": storm_id,
                    "source": source_name,
                    "target": target_name,
                    "causality": liang_result.causality[source_idx, target_idx],
                    "variance": liang_result.variance[source_idx, target_idx],
                    "normalized_causality": liang_result.normalized_causality[source_idx, target_idx],
                    "significant": bool(liang_result.significance_mask[source_idx, target_idx]),
                }
            )
    return pd.DataFrame(rows)


def save_eof_metadata(eof_results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for channel_name, eof_result in eof_results.items():
        pd.DataFrame(
            {
                "pc": np.arange(1, len(eof_result.explained_variance_ratio) + 1),
                "explained_variance_ratio": eof_result.explained_variance_ratio,
                "cumulative_explained_variance": eof_result.cumulative_explained_variance,
            }
        ).to_csv(output_dir / f"{channel_name}_explained_variance.csv", index=False)


def plot_causality_outputs(summary_df: pd.DataFrame, matrix: np.ndarray, variable_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary_df["source"], summary_df["mean_causality_to_intensity"])
    ax.set_title("Mean causality to intensity")
    ax.set_ylabel("Causality")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_causality_to_intensity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary_df["source"], summary_df["mean_normalized_causality"])
    ax.set_title("Mean normalized causality to intensity")
    ax.set_ylabel("Normalized causality")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_normalized_causality_to_intensity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_title("Mean causality matrix")
    ax.set_xticks(range(len(variable_names)))
    ax.set_yticks(range(len(variable_names)))
    ax.set_xticklabels(variable_names, rotation=90)
    ax.set_yticklabels(variable_names)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_causality_matrix.png", dpi=150)
    plt.close(fig)


def run_baseline(config: BaselineConfig) -> dict:
    records = discover_storm_records(config.era5_dir, config.intensity_dir, config.intensity_split)
    selected_ids = select_storm_ids(
        all_storm_ids=records.keys(),
        mode=config.storm_select_mode,
        selected_storm_id=config.selected_storm_id,
        random_n=config.random_n,
        random_seed=config.random_seed,
    )

    per_storm_dir = config.results_dir / "per_storm"
    summary_dir = config.results_dir / "summary"
    eof_fig_dir = config.results_dir / "figures" / "eof"
    causality_fig_dir = config.results_dir / "figures" / "causality"
    eof_meta_dir = config.results_dir / "summary" / "eof"
    per_storm_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    storm_result_frames = []
    kept_variable_names = None
    aligned_count = 0

    for storm_id in selected_ids:
        aligned = align_storm(records[storm_id], config.min_storm_length)
        if aligned is None:
            continue
        aligned_count += 1
        X, variable_names, feature_df, eof_results = build_feature_matrix(aligned, config, eof_fig_dir)
        liang_result = run_liang(
            X=X,
            dt=config.liang_dt,
            n_step=config.liang_n_step,
            significance_z=config.liang_significance_z,
        )
        result_df = causality_to_frame(storm_id, variable_names, liang_result)
        result_df.to_csv(per_storm_dir / f"{storm_id.replace(':', '_')}_causality.csv", index=False)
        feature_df.to_csv(per_storm_dir / f"{storm_id.replace(':', '_')}_features.csv")
        save_eof_metadata(eof_results, eof_meta_dir / storm_id.replace(":", "_"))
        storm_result_frames.append(result_df)
        kept_variable_names = variable_names

    if not storm_result_frames:
        raise RuntimeError("No storms passed alignment and min_storm_length filters.")

    all_results = pd.concat(storm_result_frames, ignore_index=True)
    intensity_results = all_results[all_results["target"] == "intensity_wind"].copy()
    summary_df = (
        intensity_results.groupby("source")
        .agg(
            mean_causality_to_intensity=("causality", "mean"),
            median_causality_to_intensity=("causality", "median"),
            mean_normalized_causality=("normalized_causality", "mean"),
            significance_frequency=("significant", "mean"),
            positive_significant_frequency=(
                "significant",
                lambda s: float(
                    ((intensity_results.loc[s.index, "significant"]) & (intensity_results.loc[s.index, "causality"] > 0)).mean()
                ),
            ),
            negative_significant_frequency=(
                "significant",
                lambda s: float(
                    ((intensity_results.loc[s.index, "significant"]) & (intensity_results.loc[s.index, "causality"] < 0)).mean()
                ),
            ),
        )
        .reset_index()
    )
    summary_df.to_csv(summary_dir / "causality_summary.csv", index=False)

    variable_order = ["intensity_wind"] + sorted(
        name for name in pd.unique(pd.concat([all_results["source"], all_results["target"]])) if name != "intensity_wind"
    )
    mean_matrix_df = all_results.groupby(["source", "target"])["causality"].mean().unstack(fill_value=np.nan)
    mean_matrix_df = mean_matrix_df.reindex(index=variable_order, columns=variable_order)
    mean_matrix = mean_matrix_df.to_numpy()
    mean_matrix_df.to_csv(summary_dir / "mean_causality_matrix.csv")
    plot_causality_outputs(summary_df, mean_matrix, variable_order, causality_fig_dir)

    return {
        "selected_storm_ids": selected_ids,
        "aligned_storm_count": aligned_count,
        "variable_count": len(variable_order),
        "summary_csv": summary_dir / "causality_summary.csv",
    }
