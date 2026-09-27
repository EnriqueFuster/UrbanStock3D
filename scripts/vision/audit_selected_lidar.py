"""Audit third-coverage PNOA-LiDAR availability for selected buildings."""

import argparse
import json
from pathlib import Path

import httpx

from urbanstock3d.config import Settings
from urbanstock3d.providers.pnoa_lidar import discover_cnig_lidar_cell
from urbanstock3d.vision.roof_objects.lidar_audit import audit_selected_lidar_catalog


def parse_arguments() -> argparse.Namespace:
    """Read sample, artifact root and report destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("data/selected_buildings.json"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vision/lidar_catalog_audit.json"),
    )
    parser.add_argument("--buffer-m", type=float, default=25.0)
    return parser.parse_args()


def main() -> None:
    """Discover unique CNIG cells and persist an auditable JSON report."""
    arguments = parse_arguments()
    settings = Settings()
    timeout = httpx.Timeout(
        settings.http_read_timeout_seconds,
        connect=settings.http_connect_timeout_seconds,
    )
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "UrbanStock3D/0.1 data-audit"},
    ) as client:
        report = audit_selected_lidar_catalog(
            arguments.selection,
            arguments.outputs_root,
            lambda cell: discover_cnig_lidar_cell(client, cell),
            buffer_m=arguments.buffer_m,
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(f"Wrote LiDAR catalogue audit to {arguments.output}")
    print(
        f"Buildings verified={summary['catalog_verified_buildings']} "
        f"partial={summary['partial_coverage_buildings']} "
        f"unavailable={summary['catalog_unavailable_buildings']}"
    )


if __name__ == "__main__":
    main()
