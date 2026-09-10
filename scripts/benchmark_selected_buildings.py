"""Benchmark existing Roofer and City3D artifacts for selected buildings."""

import argparse
import json
from pathlib import Path

from urbanstock3d.reconstruction.batch_benchmark import benchmark_selected_buildings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("data/selected_buildings.json"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    report = benchmark_selected_buildings(arguments.selection, arguments.outputs_root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote selected-building benchmark to {arguments.output}")


if __name__ == "__main__":
    main()
