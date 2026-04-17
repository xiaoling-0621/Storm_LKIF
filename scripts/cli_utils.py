from __future__ import annotations

import argparse
from pathlib import Path

from typhoon_causal.pipeline import BaselineConfig


CONFIG_FIELDS = [
    "era5_dir",
    "intensity_dir",
    "intensity_split",
    "storm_select_mode",
    "selected_storm_id",
    "random_n",
    "random_seed",
    "target_variable",
    "target_mode",
    "eof_fit_scope",
    "causality_scope",
    "eof_mode",
    "pc_k",
    "variance_threshold",
    "liang_dt",
    "liang_n_step",
    "liang_significance_z",
    "min_storm_length",
    "max_eof_plots",
    "eof_map_components",
    "results_dir",
]


def add_config_override_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--era5-dir", dest="era5_dir", help="Override era5_dir")
    parser.add_argument("--intensity-dir", dest="intensity_dir", help="Override intensity_dir")
    parser.add_argument("--intensity-split", dest="intensity_split", help="Override intensity_split, e.g. train/test/vaild")
    parser.add_argument("--storm-select-mode", dest="storm_select_mode", choices=["single", "random_n", "all"], help="Storm selection mode")
    parser.add_argument("--selected-storm-id", dest="selected_storm_id", help="Storm ID for single mode, e.g. 1950:BILLIE")
    parser.add_argument("--random-n", dest="random_n", type=int, help="Number of random storms to sample")
    parser.add_argument("--random-seed", dest="random_seed", type=int, help="Random seed for random_n mode")
    parser.add_argument("--target-variable", dest="target_variable", choices=["wind", "pressure"], help="Target variable")
    parser.add_argument("--target-mode", dest="target_mode", choices=["raw", "delta"], help="Target mode")
    parser.add_argument("--eof-fit-scope", dest="eof_fit_scope", choices=["per_storm", "global_selected"], help="EOF fit scope")
    parser.add_argument("--causality-scope", dest="causality_scope", choices=["per_storm", "global_segmented"], help="Causality scope")
    parser.add_argument("--eof-mode", dest="eof_mode", choices=["fixed_k", "variance_threshold"], help="EOF PC selection mode")
    parser.add_argument("--pc-k", dest="pc_k", type=int, help="Number of PCs when eof_mode=fixed_k")
    parser.add_argument("--variance-threshold", dest="variance_threshold", type=float, help="Cumulative explained variance threshold")
    parser.add_argument("--liang-dt", dest="liang_dt", type=float, help="Liang dt parameter")
    parser.add_argument("--liang-n-step", dest="liang_n_step", type=int, help="Liang derivative step")
    parser.add_argument("--liang-significance-z", dest="liang_significance_z", type=float, help="Z threshold for significance")
    parser.add_argument("--min-storm-length", dest="min_storm_length", type=int, help="Minimum usable length after alignment")
    parser.add_argument("--max-eof-plots", dest="max_eof_plots", type=int, help="Maximum channels to plot EOF figures for")
    parser.add_argument("--eof-map-components", dest="eof_map_components", type=int, help="Number of EOF spatial maps to save per channel")
    parser.add_argument("--results-dir", dest="results_dir", help="Override results_dir")


def build_config_from_args(config_path: Path, args: argparse.Namespace) -> BaselineConfig:
    config = BaselineConfig.from_yaml(config_path)
    for field in CONFIG_FIELDS:
        override = getattr(args, field, None)
        if override is not None:
            if field.endswith("_dir"):
                override = Path(override)
            setattr(config, field, override)
    return config


def format_config(config: BaselineConfig) -> str:
    lines = []
    for field in CONFIG_FIELDS:
        lines.append(f"{field}: {getattr(config, field)}")
    return "\n".join(lines)
