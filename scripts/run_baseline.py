from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from typhoon_causal.pipeline import BaselineConfig, run_baseline


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "baseline.yaml"
    config = BaselineConfig.from_yaml(config_path)
    result = run_baseline(config)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
