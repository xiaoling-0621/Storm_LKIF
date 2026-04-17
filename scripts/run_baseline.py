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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one typhoon causal baseline experiment from a YAML config, with optional CLI overrides."
    )
    parser.add_argument("config_path", nargs="?", default="configs/baseline.yaml", help="Path to a YAML config file")
    add_config_override_arguments(parser)
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = build_config_from_args(config_path, args)
    result = run_baseline(config)
    print(f"config_path: {config_path}")
    print("effective_config:")
    print(format_config(config))
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
