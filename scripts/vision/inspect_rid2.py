"""Verify RID2 and write a compact archive inventory without extracting it."""

import argparse
from pathlib import Path

from urbanstock3d.vision.roof_objects.datasets import (
    inspect_zip_archive,
    load_rid2_source,
    verify_rid2_archive,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        nargs="?",
        type=Path,
        default=Path("data/vision/rid2/raw/roof_information_dataset_2.zip"),
    )
    parser.add_argument(
        "--source-config", type=Path, default=Path("config/vision/datasets/rid2.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/vision/rid2/archive_inventory.json")
    )
    arguments = parser.parse_args()

    source = load_rid2_source(arguments.source_config)
    verify_rid2_archive(arguments.archive, source)
    inventory = inspect_zip_archive(arguments.archive)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(inventory.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(inventory.model_dump_json(indent=2))
    print(f"Wrote archive inventory to {arguments.output}")


if __name__ == "__main__":
    main()
