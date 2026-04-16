from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import xarray as xr


ERA_TIMESTAMP_RE = re.compile(r"_(\d{10})_")
INTENSITY_STEM_RE = re.compile(r"WP(\d{4})BST(.+)", re.IGNORECASE)

INTENSITY_COLUMNS = [
    "row_index",
    "basin_flag",
    "track_lat_feature",
    "track_lon_feature",
    "pressure_like_feature",
    "wind_like_feature",
    "timestamp",
    "storm_name",
]


@dataclass
class StormRecord:
    storm_id: str
    year: str
    storm_name: str
    era_files: Dict[pd.Timestamp, Path]
    intensity: pd.DataFrame


@dataclass
class AlignedStormData:
    storm_id: str
    storm_name: str
    timestamps: List[pd.Timestamp]
    era_files: List[Path]
    intensity_values: np.ndarray


def normalize_storm_name(name: str) -> str:
    return name.upper().replace(" ", "_")


def make_storm_id(year: str, storm_name: str) -> str:
    return f"{year}:{normalize_storm_name(storm_name)}"


def extract_era_timestamp(path: Path) -> pd.Timestamp:
    match = ERA_TIMESTAMP_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse timestamp from {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d%H")


def load_intensity_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] != len(INTENSITY_COLUMNS):
        raise ValueError(f"Unexpected intensity column count in {path}: {df.shape[1]}")
    df.columns = INTENSITY_COLUMNS
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str), format="%Y%m%d%H")
    df["storm_name"] = df["storm_name"].astype(str).str.upper()
    return df


def discover_storm_records(era5_dir: Path, intensity_dir: Path, intensity_split: str) -> Dict[str, StormRecord]:
    era_index: Dict[str, Dict[pd.Timestamp, Path]] = {}
    for nc_path in sorted(era5_dir.rglob("*.nc")):
        year = nc_path.parts[-3]
        storm_name = nc_path.parts[-2]
        storm_id = make_storm_id(year, storm_name)
        era_index.setdefault(storm_id, {})[extract_era_timestamp(nc_path)] = nc_path

    records: Dict[str, StormRecord] = {}
    for txt_path in sorted((intensity_dir / intensity_split).glob("*.txt")):
        match = INTENSITY_STEM_RE.match(txt_path.stem)
        if not match:
            continue
        year, storm_name = match.groups()
        storm_id = make_storm_id(year, storm_name)
        if storm_id not in era_index:
            continue
        intensity_df = load_intensity_file(txt_path)
        records[storm_id] = StormRecord(
            storm_id=storm_id,
            year=year,
            storm_name=normalize_storm_name(storm_name),
            era_files=era_index[storm_id],
            intensity=intensity_df,
        )
    return records


def inspect_era_sample(sample_path: Path) -> dict:
    ds = xr.open_dataset(sample_path)
    try:
        channels = []
        variables = {}
        for var_name, da in ds.data_vars.items():
            variables[var_name] = {"dims": list(da.dims), "shape": [int(v) for v in da.shape]}
            if "pressure_level" in da.dims:
                for level in da["pressure_level"].values.tolist():
                    channels.append(f"{var_name}_{int(level)}")
            else:
                channels.append(var_name)
        return {
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "variables": variables,
            "channels": channels,
        }
    finally:
        ds.close()


def infer_intensity_columns(df: pd.DataFrame) -> dict:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return {
        "timestamp_column": "timestamp",
        "storm_name_column": "storm_name",
        "wind_column": "wind_like_feature",
        "pressure_column": "pressure_like_feature",
        "candidate_numeric_columns": numeric_cols,
        "note": (
            "Columns 4/5 are inferred normalized pressure/wind features; "
            "column 5 is used as baseline intensity because it increases when storms intensify "
            "while column 4 typically decreases, matching wind/pressure behavior."
        ),
    }


def summarize_records(records: Dict[str, StormRecord]) -> pd.DataFrame:
    rows = []
    for storm_id, record in records.items():
        rows.append(
            {
                "storm_id": storm_id,
                "era_time_count": len(record.era_files),
                "intensity_time_count": len(record.intensity),
                "intensity_time_min": record.intensity["timestamp"].min(),
                "intensity_time_max": record.intensity["timestamp"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values(["era_time_count", "storm_id"], ascending=[False, True])


def select_storm_ids(
    all_storm_ids: Iterable[str],
    mode: str,
    selected_storm_id: str | None = None,
    random_n: int | None = None,
    random_seed: int = 42,
) -> List[str]:
    storm_ids = sorted(all_storm_ids)
    if mode == "all":
        return storm_ids
    if mode == "single":
        if not selected_storm_id:
            raise ValueError("selected_storm_id is required when mode='single'")
        if selected_storm_id not in storm_ids:
            raise ValueError(f"Storm {selected_storm_id} not found")
        return [selected_storm_id]
    if mode == "random_n":
        if not random_n or random_n <= 0:
            raise ValueError("random_n must be positive when mode='random_n'")
        rng = np.random.default_rng(random_seed)
        count = min(random_n, len(storm_ids))
        return sorted(rng.choice(storm_ids, size=count, replace=False).tolist())
    raise ValueError(f"Unsupported storm selection mode: {mode}")


def align_storm(record: StormRecord, min_storm_length: int) -> AlignedStormData | None:
    era_times = set(record.era_files.keys())
    intensity_df = record.intensity.copy()
    common_times = sorted(set(intensity_df["timestamp"]) & era_times)
    if len(common_times) < min_storm_length:
        return None
    aligned_intensity = (
        intensity_df.set_index("timestamp")
        .loc[common_times, "wind_like_feature"]
        .astype(float)
        .to_numpy()
    )
    return AlignedStormData(
        storm_id=record.storm_id,
        storm_name=record.storm_name,
        timestamps=common_times,
        era_files=[record.era_files[timestamp] for timestamp in common_times],
        intensity_values=aligned_intensity,
    )


def split_channels(ds: xr.Dataset) -> Dict[str, xr.DataArray]:
    channels: Dict[str, xr.DataArray] = {}
    for var_name, da in ds.data_vars.items():
        if "time" in da.dims and da.sizes.get("time", 0) == 1:
            da = da.isel(time=0, drop=True)
        if "pressure_level" in da.dims:
            for level in da["pressure_level"].values.tolist():
                channel_name = f"{var_name}_{int(level)}"
                channels[channel_name] = da.sel(pressure_level=level, drop=True)
        else:
            channels[var_name] = da
    return channels


def load_channel_time_series(aligned_storm: AlignedStormData) -> Dict[str, dict]:
    channel_arrays: Dict[str, List[np.ndarray]] = {}
    channel_coords: Dict[str, dict] = {}
    for era_file in aligned_storm.era_files:
        ds = xr.open_dataset(era_file)
        try:
            for channel_name, da in split_channels(ds).items():
                values = da.to_numpy().astype(np.float64)
                channel_arrays.setdefault(channel_name, []).append(values)
                if channel_name not in channel_coords:
                    channel_coords[channel_name] = {
                        "latitude": da["latitude"].to_numpy(),
                        "longitude": da["longitude"].to_numpy(),
                    }
        finally:
            ds.close()
    return {
        channel_name: {
            "data": np.stack(arrays, axis=0),
            "latitude": channel_coords[channel_name]["latitude"],
            "longitude": channel_coords[channel_name]["longitude"],
        }
        for channel_name, arrays in channel_arrays.items()
    }
