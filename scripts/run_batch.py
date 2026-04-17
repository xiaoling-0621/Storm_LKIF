from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cli_utils import add_config_override_arguments, build_config_from_args, format_config
from typhoon_causal.pipeline import run_baseline


def resolve_config_paths(args: list[str]) -> list[Path]:
    if not args:
        return [PROJECT_ROOT / "configs" / "baseline.yaml"]

    resolved: list[Path] = []
    for raw_arg in args:
        candidate = Path(raw_arg)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate.is_dir():
            resolved.extend(sorted(candidate.glob("*.yaml")))
        else:
            resolved.append(candidate)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple typhoon causal baseline experiments from YAML configs or a config directory."
    )
    parser.add_argument(
        "config_paths",
        nargs="*",
        help="Config files or directories containing YAML configs. Defaults to configs/baseline.yaml",
    )
    add_config_override_arguments(parser)
    args = parser.parse_args()

    config_paths = resolve_config_paths(args.config_paths)
    if not config_paths:
        raise RuntimeError("No config files found.")

    for config_path in config_paths:
        print(f"=== Running {config_path} ===")
        config = build_config_from_args(config_path, args)
        result = run_baseline(config)
        print(f"config_path: {config_path}")
        print("effective_config:")
        print(format_config(config))
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
