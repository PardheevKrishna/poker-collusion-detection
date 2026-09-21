#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poker_coordination.reproduce import reproduce_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce both selected Kaggle submission files")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    print(json.dumps(reproduce_all(args.artifact_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

