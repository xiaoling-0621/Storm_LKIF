from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .data_utils import (
    AlignedStormData,
    align_storm,
    discover_storm_records,
    get_target_column,
    inspect_era_sample,
    infer_intensity_columns,
    load_channel_time_series,
    select_storm_ids,
    summarize_records,
)
from .eof import EOFModel, EOFResult, fit_channel_pca, fit_channel_pca_model, save_eof_plots, transform_channel_data
from .liang_wrapper import run_liang, run_liang_segmented


@dataclass
class BaselineConfig:
    era5_dir: Path
    intensity_dir: Path
    intensity_split: str
    storm_select_mode: str
    selected_storm_id: str | None
    random_n: int
    random_seed: int
    target_variable: str
    target_mode: str
    eof_fit_scope: str
    causality_scope: str
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


def build_experiment_tag(config: BaselineConfig) -> str:
    return f"{config.target_variable}_{config.target_mode}_{config.eof_fit_scope}_{config.causality_scope}"


def build_target_name(config: BaselineConfig) -> str:
    return f"{config.target_variable}_{config.target_mode}"


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
    target_column = get_target_column(config.target_variable)
    return "\n".join(
        [
            "# Data Summary",
            "",
            f"- Matched storms: {len(records)}",
            f"- ERA5 sample file: `{sample_era_path}`",
            f"- Sample storm_id format: `{sample_storm_id}`",
            f"- Intensity timestamp column: `{intensity_meta['timestamp_column']}`",
            f"- Target variable: `{config.target_variable}` -> `{target_column}`",
            f"- Target mode: `{config.target_mode}`",
            f"- EOF fit scope: `{config.eof_fit_scope}`",
            f"- Causality scope: `{config.causality_scope}`",
            f"- Inference note: {intensity_meta['note']}",
            "",
            "## ERA5 Structure",
            "",
            f"- Dims: `{era_meta['dims']}`",
            f"- Variables: `{list(era_meta['variables'].keys())}`",
            f"- Expanded channels: `{era_meta['channels']}`",
            "",
            "## Delta Target Alignment",
            "",
            "- `delta` mode uses `target[t+1] - target[t]`.",
            "- ERA5 features are truncated to the first `T-1` timestamps.",
            "- The mapping is `ERA5(t) -> target(t+1) - target(t)`.",
            "",
            "## Storm Counts",
            "",
            top_lines,
            "",
        ]
    )


def prepare_target_and_channels(
    aligned_storm: AlignedStormData,
    channel_data: dict[str, dict],
    config: BaselineConfig,
) -> tuple[list[pd.Timestamp], np.ndarray, dict[str, dict]]:
    timestamps = list(aligned_storm.timestamps)
    target_values = aligned_storm.target_values.astype(np.float64)
    adjusted_channel_data = {
        channel_name: {
            "data": info["data"].copy(),
            "latitude": info["latitude"],
            "longitude": info["longitude"],
        }
        for channel_name, info in channel_data.items()
    }

    if config.target_mode == "raw":
        return timestamps, target_values, adjusted_channel_data
    if config.target_mode == "delta":
        if len(target_values) < 2:
            raise ValueError(f"Storm {aligned_storm.storm_id} is too short for delta target.")
        delta_target = target_values[1:] - target_values[:-1]
        truncated_timestamps = timestamps[:-1]
        for channel_name in adjusted_channel_data:
            adjusted_channel_data[channel_name]["data"] = adjusted_channel_data[channel_name]["data"][:-1]
        return truncated_timestamps, delta_target, adjusted_channel_data
    raise ValueError(f"Unsupported target_mode: {config.target_mode}")


def collect_aligned_storms(config: BaselineConfig) -> tuple[list[str], list[AlignedStormData]]:
    records = discover_storm_records(config.era5_dir, config.intensity_dir, config.intensity_split)
    selected_ids = select_storm_ids(
        all_storm_ids=records.keys(),
        mode=config.storm_select_mode,
        selected_storm_id=config.selected_storm_id,
        random_n=config.random_n,
        random_seed=config.random_seed,
    )
    aligned_storms = []
    for storm_id in selected_ids:
        aligned = align_storm(records[storm_id], config.min_storm_length, config.target_variable)
        if aligned is not None:
            aligned_storms.append(aligned)
    return selected_ids, aligned_storms


def fit_global_eof_models(prepared_storm_inputs: dict[str, dict], config: BaselineConfig) -> dict[str, EOFModel]:
    channel_names = sorted(next(iter(prepared_storm_inputs.values()))["channel_data"].keys())
    models: dict[str, EOFModel] = {}
    for channel_name in channel_names:
        arrays = [prepared_storm_inputs[storm_id]["channel_data"][channel_name]["data"] for storm_id in prepared_storm_inputs]
        reference = next(iter(prepared_storm_inputs.values()))["channel_data"][channel_name]
        combined = np.concatenate(arrays, axis=0)
        models[channel_name] = fit_channel_pca_model(
            channel_name=channel_name,
            channel_data=combined,
            latitudes=reference["latitude"],
            longitudes=reference["longitude"],
            eof_mode=config.eof_mode,
            pc_k=config.pc_k,
            variance_threshold=config.variance_threshold,
        )
    return models


def build_feature_matrix_for_storm(
    storm_id: str,
    prepared_input: dict,
    config: BaselineConfig,
    eof_output_dir: Path,
    eof_models: dict[str, EOFModel] | None = None,
    allow_plots: bool = True,
):
    channel_data = prepared_input["channel_data"]
    target_values = prepared_input["target_values"]
    timestamps = prepared_input["timestamps"]
    target_name = build_target_name(config)

    eof_results: dict[str, EOFResult] = {}
    variable_names = [target_name]
    rows = [target_values.astype(np.float64)]

    plot_prefix = f"{build_experiment_tag(config)}_{storm_id.replace(':', '_')}"
    for idx, channel_name in enumerate(sorted(channel_data.keys())):
        channel_info = channel_data[channel_name]
        if eof_models is None:
            eof_result = fit_channel_pca(
                channel_name=channel_name,
                channel_data=channel_info["data"],
                latitudes=channel_info["latitude"],
                longitudes=channel_info["longitude"],
                eof_mode=config.eof_mode,
                pc_k=config.pc_k,
                variance_threshold=config.variance_threshold,
            )
        else:
            eof_result = transform_channel_data(channel_info["data"], eof_models[channel_name])
        eof_results[channel_name] = eof_result
        if allow_plots and idx < config.max_eof_plots:
            save_eof_plots(eof_result, eof_output_dir, config.eof_map_components, file_prefix=plot_prefix)
        for pc_index in range(eof_result.selected_count):
            variable_names.append(f"{channel_name}_pc{pc_index + 1}")
            rows.append(eof_result.selected_scores[:, pc_index])

    X = np.vstack(rows)
    feature_df = pd.DataFrame(X.T, columns=variable_names, index=timestamps)
    return X, variable_names, feature_df, eof_results


def causality_to_frame(scope_id: str, variable_names: list[str], liang_result) -> pd.DataFrame:
    rows = []
    for source_idx, source_name in enumerate(variable_names):
        for target_idx, target_name in enumerate(variable_names):
            rows.append(
                {
                    "scope_id": scope_id,
                    "source": source_name,
                    "target": target_name,
                    "causality": liang_result.causality[source_idx, target_idx],
                    "variance": liang_result.variance[source_idx, target_idx],
                    "normalized_causality": liang_result.normalized_causality[source_idx, target_idx],
                    "significant": bool(liang_result.significance_mask[source_idx, target_idx]),
                }
            )
    return pd.DataFrame(rows)


def save_eof_metadata(eof_results: dict[str, EOFResult], output_dir: Path, file_prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for channel_name, eof_result in eof_results.items():
        pd.DataFrame(
            {
                "pc": np.arange(1, len(eof_result.explained_variance_ratio) + 1),
                "explained_variance_ratio": eof_result.explained_variance_ratio,
                "cumulative_explained_variance": eof_result.cumulative_explained_variance,
            }
        ).to_csv(output_dir / f"{file_prefix}_{channel_name}.csv", index=False)


def plot_causality_outputs(summary_df: pd.DataFrame, matrix: np.ndarray, variable_names: list[str], output_dir: Path, file_prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary_df["source"], summary_df["mean_causality_to_target"])
    ax.set_title("Mean causality to target")
    ax.set_ylabel("Causality")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}_mean_causality_to_target.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary_df["source"], summary_df["mean_normalized_causality"])
    ax.set_title("Mean normalized causality to target")
    ax.set_ylabel("Normalized causality")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}_mean_normalized_causality_to_target.png", dpi=150)
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
    fig.savefig(output_dir / f"{file_prefix}_mean_causality_matrix.png", dpi=150)
    plt.close(fig)


def summarize_target_results(all_results: pd.DataFrame, target_name: str) -> pd.DataFrame:
    target_results = all_results[all_results["target"] == target_name].copy()
    summary_df = (
        target_results.groupby("source")
        .agg(
            mean_causality_to_target=("causality", "mean"),
            median_causality_to_target=("causality", "median"),
            mean_normalized_causality=("normalized_causality", "mean"),
            significance_frequency=("significant", "mean"),
            positive_significant_frequency=(
                "significant",
                lambda s: float(
                    ((target_results.loc[s.index, "significant"]) & (target_results.loc[s.index, "causality"] > 0)).mean()
                ),
            ),
            negative_significant_frequency=(
                "significant",
                lambda s: float(
                    ((target_results.loc[s.index, "significant"]) & (target_results.loc[s.index, "causality"] < 0)).mean()
                ),
            ),
        )
        .reset_index()
    )
    return summary_df


def run_baseline(config: BaselineConfig) -> dict:
    selected_ids, aligned_storms = collect_aligned_storms(config)
    if not aligned_storms:
        raise RuntimeError("No storms passed alignment and min_storm_length filters.")

    experiment_tag = build_experiment_tag(config)
    target_name = build_target_name(config)
    per_storm_dir = config.results_dir / "per_storm"
    summary_dir = config.results_dir / "summary"
    eof_fig_dir = config.results_dir / "figures" / "eof"
    causality_fig_dir = config.results_dir / "figures" / "causality"
    eof_meta_dir = config.results_dir / "eof_metadata"
    per_storm_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    eof_meta_dir.mkdir(parents=True, exist_ok=True)

    prepared_inputs: dict[str, dict] = {}
    for aligned in aligned_storms:
        raw_channel_data = load_channel_time_series(aligned)
        timestamps, target_values, adjusted_channel_data = prepare_target_and_channels(aligned, raw_channel_data, config)
        if len(target_values) < config.min_storm_length:
            continue
        prepared_inputs[aligned.storm_id] = {
            "storm": aligned,
            "timestamps": timestamps,
            "target_values": target_values,
            "channel_data": adjusted_channel_data,
        }
    if not prepared_inputs:
        raise RuntimeError("No storms remained after applying target alignment.")

    eof_models = None
    if config.eof_fit_scope == "global_selected":
        eof_models = fit_global_eof_models(prepared_inputs, config)

    storm_result_frames = []
    segment_matrices = []
    variable_order = None

    for storm_id, prepared_input in prepared_inputs.items():
        X, variable_names, feature_df, eof_results = build_feature_matrix_for_storm(
            storm_id=storm_id,
            prepared_input=prepared_input,
            config=config,
            eof_output_dir=eof_fig_dir,
            eof_models=eof_models,
            allow_plots=True,
        )
        segment_matrices.append(X)
        variable_order = variable_names

        if config.causality_scope == "per_storm":
            liang_result = run_liang(
                X=X,
                dt=config.liang_dt,
                n_step=config.liang_n_step,
                significance_z=config.liang_significance_z,
            )
            result_df = causality_to_frame(storm_id, variable_names, liang_result)
            storm_result_frames.append(result_df)
            result_df.to_csv(per_storm_dir / f"{storm_id.replace(':', '_')}_{experiment_tag}_causality.csv", index=False)

        feature_df.to_csv(per_storm_dir / f"{storm_id.replace(':', '_')}_{experiment_tag}_features.csv")
        save_eof_metadata(eof_results, eof_meta_dir, f"{storm_id.replace(':', '_')}_{experiment_tag}")

    if variable_order is None:
        raise RuntimeError("No valid feature matrices were produced.")

    if eof_models is not None:
        for channel_name, eof_model in eof_models.items():
            pd.DataFrame(
                {
                    "pc": np.arange(1, len(eof_model.explained_variance_ratio) + 1),
                    "explained_variance_ratio": eof_model.explained_variance_ratio,
                    "cumulative_explained_variance": eof_model.cumulative_explained_variance,
                }
            ).to_csv(eof_meta_dir / f"global_{experiment_tag}_{channel_name}.csv", index=False)

    if config.causality_scope == "global_segmented":
        liang_result = run_liang_segmented(
            segments=segment_matrices,
            dt=config.liang_dt,
            n_step=config.liang_n_step,
            significance_z=config.liang_significance_z,
        )
        global_df = causality_to_frame("global_segmented", variable_order, liang_result)
        storm_result_frames = [global_df]
        global_df.to_csv(summary_dir / f"{experiment_tag}_global_segmented_causality.csv", index=False)

    all_results = pd.concat(storm_result_frames, ignore_index=True)
    summary_df = summarize_target_results(all_results, target_name)
    summary_df.to_csv(summary_dir / f"{experiment_tag}_causality_summary.csv", index=False)

    full_variable_order = [target_name] + sorted(
        name for name in pd.unique(pd.concat([all_results["source"], all_results["target"]])) if name != target_name
    )
    mean_matrix_df = all_results.groupby(["source", "target"])["causality"].mean().unstack(fill_value=np.nan)
    mean_matrix_df = mean_matrix_df.reindex(index=full_variable_order, columns=full_variable_order)
    mean_matrix_df.to_csv(summary_dir / f"{experiment_tag}_mean_causality_matrix.csv")
    plot_causality_outputs(
        summary_df=summary_df,
        matrix=mean_matrix_df.to_numpy(),
        variable_names=full_variable_order,
        output_dir=causality_fig_dir,
        file_prefix=experiment_tag,
    )

    return {
        "selected_storm_ids": selected_ids,
        "usable_storm_ids": sorted(prepared_inputs.keys()),
        "aligned_storm_count": len(prepared_inputs),
        "variable_count": len(full_variable_order),
        "target_name": target_name,
        "experiment_tag": experiment_tag,
        "summary_csv": summary_dir / f"{experiment_tag}_causality_summary.csv",
    }
