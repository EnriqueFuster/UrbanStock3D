"""Audit the facade image advertised by an exported Catastro building."""

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

from urbanstock3d.config import Settings


def parse_arguments() -> argparse.Namespace:
    """Read the building artifact and output options."""
    parser = argparse.ArgumentParser(description="Audit one Catastro facade image.")
    parser.add_argument("building_geojson", type=Path)
    parser.add_argument("--save-preview", action="store_true")
    return parser.parse_args()


def facade_url(geojson: dict[str, Any]) -> str:
    """Return the facade URL advertised in a building artifact."""
    document = geojson.get("properties", {}).get("facade_document")
    if not document or not document.get("url"):
        raise ValueError("Building has no advertised facade image")
    return str(document["url"])


def inspect_image(content: bytes) -> dict[str, Any]:
    """Decode image bytes and return evidence about the payload."""
    try:
        with Image.open(BytesIO(content)) as image:
            evidence: dict[str, Any] = {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "decode_warning": None,
            }
            try:
                image.load()
                evidence["channel_extrema"] = image.getextrema()
            except OSError as error:
                if "truncated" not in str(error).lower():
                    raise ValueError("Catastro facade image could not be decoded") from error
                evidence["decode_warning"] = str(error)
                evidence["channel_extrema"] = None
            return evidence
    except UnidentifiedImageError as error:
        raise ValueError("Catastro facade response is not a valid image") from error


def audit_facade(building_geojson: Path, *, save_preview: bool = False) -> Path:
    """Download, validate and record an advertised facade image."""
    geojson: dict[str, Any] = json.loads(building_geojson.read_text(encoding="utf-8"))
    url = facade_url(geojson)
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
        response = client.get(url)
        response.raise_for_status()

    image_evidence = inspect_image(response.content)
    preview_path = building_geojson.with_name("facade_preview.jpg")
    if save_preview:
        preview_path.write_bytes(response.content)

    report = {
        "provider": "Dirección General del Catastro",
        "product": "cadastral facade photograph",
        "status": ("available_with_warning" if image_evidence["decode_warning"] else "available"),
        "request": {"advertised_url": url},
        "response": {
            "resolved_url": str(response.url),
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type"),
            "bytes": len(response.content),
            **image_evidence,
        },
        "preview_path": preview_path.name if save_preview else None,
    }
    report_path = building_geojson.with_name("facade_audit.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    """Run the facade audit from the command line."""
    args = parse_arguments()
    report_path = audit_facade(args.building_geojson, save_preview=args.save_preview)
    print(f"Wrote facade audit to {report_path}")


if __name__ == "__main__":
    main()
