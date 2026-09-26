"""Render a compact visual check of converted RID2 segmentation labels."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from urbanstock3d.vision.common import load_taxonomy

COLORS = {
    "pv_panel": "#00A6D6",
    "hvac_unit": "#F28E2B",
    "chimney": "#D62728",
    "elevator_overrun": "#9467BD",
    "skylight": "#2CA02C",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/vision/rid2/yolo"))
    parser.add_argument("--output", type=Path, default=Path("data/vision/rid2/sample_montage.png"))
    arguments = parser.parse_args()

    taxonomy = load_taxonomy(Path("config/vision/taxonomy.yaml"))
    class_names = {identifier: name for name, identifier in taxonomy.roof_objects.items()}
    candidates = sorted((arguments.dataset / "labels").glob("*/*.txt"))
    selected = [path for path in candidates if path.stat().st_size > 0][:6]
    if not selected:
        raise SystemExit("No non-empty converted labels were found")

    tiles: list[Image.Image] = []
    for label_path in selected:
        image_path = (
            arguments.dataset / "images" / label_path.parent.name / f"{label_path.stem}.png"
        )
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            class_name = class_names[int(values[0])]
            coordinates = [float(value) for value in values[1:]]
            points = [
                (coordinates[index] * image.width, coordinates[index + 1] * image.height)
                for index in range(0, len(coordinates), 2)
            ]
            draw.polygon(points, outline=COLORS[class_name], width=3)
            draw.text(
                points[0], class_name, fill=COLORS[class_name], stroke_width=2, stroke_fill="white"
            )
        tiles.append(image)

    tile_width, tile_height = tiles[0].size
    montage = Image.new("RGB", (tile_width * 3, tile_height * 2), "white")
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % 3) * tile_width, (index // 3) * tile_height))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(arguments.output)
    print(f"Wrote {len(tiles)} annotated samples to {arguments.output}")


if __name__ == "__main__":
    main()
