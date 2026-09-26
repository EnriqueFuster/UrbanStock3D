"""Convert verified RID2 imagery and vectors to canonical YOLO segmentation data."""

import argparse
from pathlib import Path

from urbanstock3d.vision.common import load_taxonomy
from urbanstock3d.vision.roof_objects.datasets import load_rid2_source, verify_rid2_archive
from urbanstock3d.vision.roof_objects.rid2_conversion import convert_rid2_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        nargs="?",
        type=Path,
        default=Path("data/vision/rid2/raw/roof_information_dataset_2.zip"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/vision/rid2/yolo"))
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()

    source = load_rid2_source(Path("config/vision/datasets/rid2.yaml"))
    taxonomy = load_taxonomy(Path("config/vision/taxonomy.yaml"))
    verify_rid2_archive(arguments.archive, source)
    manifest = convert_rid2_archive(
        arguments.archive, arguments.output, source, taxonomy, limit=arguments.limit
    )
    print(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
