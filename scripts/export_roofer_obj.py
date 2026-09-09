"""Export a Roofer CityJSONSeq building to OBJ."""

import argparse
from pathlib import Path

from urbanstock3d.formats.cityjson import read_cityjsonseq, write_obj


def main() -> None:
    """Read arguments and export the requested geometry."""
    parser = argparse.ArgumentParser(description="Export Roofer CityJSONSeq to OBJ.")
    parser.add_argument("cityjsonseq", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lod", default="2.2")
    args = parser.parse_args()
    metadata, feature = read_cityjsonseq(args.cityjsonseq)
    obj_path, material_path = write_obj(metadata, feature, args.output, lod=args.lod)
    print(f"Wrote {obj_path}")
    print(f"Wrote {material_path}")


if __name__ == "__main__":
    main()
