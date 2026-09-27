"""Download a georeferenced PNOA orthophoto crop for one resolved building."""

import argparse
from pathlib import Path

from urbanstock3d.providers.pnoa_orthophoto import acquire_pnoa_crop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--buffer-m", type=float, default=10.0)
    parser.add_argument("--pixel-size-m", type=float, default=0.25)
    arguments = parser.parse_args()
    if not arguments.building_geojson.exists():
        raise SystemExit(f"Missing building geometry: {arguments.building_geojson}")

    output = arguments.output or arguments.building_geojson.with_name("orthophoto_crop.tif")
    metadata = acquire_pnoa_crop(
        arguments.building_geojson,
        output,
        buffer_m=arguments.buffer_m,
        pixel_size_m=arguments.pixel_size_m,
    )
    print(f"Wrote PNOA orthophoto to {output}")
    print(f"Wrote acquisition metadata to {metadata}")


if __name__ == "__main__":
    main()
