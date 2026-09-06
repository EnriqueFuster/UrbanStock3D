"""Download one CNIG LAZ temporarily and inspect its header."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from urbanstock3d.config import Settings
from urbanstock3d.processors.lidar import inspect_lidar_header
from urbanstock3d.providers.pnoa_lidar import (
    download_cnig_asset,
    parse_cnig_asset_page,
)


def parse_arguments() -> argparse.Namespace:
    """Read the official detail URL and report destination."""
    parser = argparse.ArgumentParser(description="Inspect one official CNIG LiDAR asset.")
    parser.add_argument("detail_url")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def inspect_remote_asset(detail_url: str, output: Path) -> Path:
    """Inspect a CNIG LAZ while keeping the raw source temporary."""
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
        detail_response = client.get(detail_url)
        detail_response.raise_for_status()
        asset = parse_cnig_asset_page(detail_response.content, detail_url)

        with TemporaryDirectory(prefix="urbanstock3d-lidar-") as temporary_dir:
            raw_path = Path(temporary_dir) / asset.filename
            download = download_cnig_asset(client, asset, raw_path)
            header = inspect_lidar_header(raw_path)

    report = {
        "provider": "IGN-CNIG",
        "product": "PNOA-LiDAR third coverage (2022-2025)",
        "status": "verified",
        "asset": asdict(asset),
        "download": asdict(download),
        "header": header,
        "raw_source_retained": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    """Run the temporary LiDAR inspection."""
    args = parse_arguments()
    report_path = inspect_remote_asset(args.detail_url, args.output)
    print(f"Wrote LiDAR asset audit to {report_path}")


if __name__ == "__main__":
    main()
