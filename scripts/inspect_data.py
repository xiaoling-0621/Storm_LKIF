from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from typhoon_causal.pipeline import BaselineConfig, build_data_summary


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "baseline.yaml"
    config = BaselineConfig.from_yaml(config_path)
    summary = build_data_summary(config)
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    output_path = docs_dir / "data_summary.md"
    output_path.write_text(summary, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
