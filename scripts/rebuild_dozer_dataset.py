#!/usr/bin/env python3
"""Rebuild a synthetic dozer object-detection dataset.

This reconstructs the missing dozer dataset notebook's main output from the
available Blender captures:

  data/captures/{train,test}/image_XXXXX_img.png
  data/captures/{train,test}/image_XXXXX_layer.png

The layer files encode the foreground object in red. The script cuts that
foreground out, composites it onto random COCO validation backgrounds, and
writes annotations.csv files in the shape expected by dozer_object_detector.ipynb.
It can also emit YOLO detection labels from the same mask-derived boxes.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


CSV_HEADER = ["id", "image", "width", "height", "xmin", "ymin", "xmax", "ymax", "class", "mask"]
CLASS_NAME = "bulldozer"


@dataclass(frozen=True)
class CapturePair:
    image: Path
    layer: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-dir", type=Path, default=Path("data/captures"))
    parser.add_argument("--backgrounds-zip", type=Path, default=Path("data/external/val2017.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=200)
    parser.add_argument("--size", type=int, default=512, help="Square output image size.")
    parser.add_argument("--seed", type=int, default=20210523)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument(
        "--write-yolo",
        action="store_true",
        help="Also write YOLO labels and dataset.yaml for modern detector training.",
    )
    return parser.parse_args()


def find_pairs(captures_dir: Path, split: str) -> list[CapturePair]:
    split_dir = captures_dir / split
    pairs: list[CapturePair] = []
    for image in sorted(split_dir.glob("*_img.png")):
        layer = image.with_name(image.name.replace("_img.png", "_layer.png"))
        if layer.exists():
            pairs.append(CapturePair(image=image, layer=layer))
    if not pairs:
        raise FileNotFoundError(f"No *_img.png/*_layer.png pairs found in {split_dir}")
    return pairs


def list_backgrounds(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            name
            for name in zf.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png")) and not name.endswith("/")
        ]
    if not names:
        raise FileNotFoundError(f"No background images found in {zip_path}")
    return names


def load_background(zip_path: Path, name: str, size: int) -> Image.Image:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(name) as fp:
            image = Image.open(fp).convert("RGB")
            return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def red_foreground_mask(layer: Image.Image) -> np.ndarray:
    arr = np.asarray(layer.convert("RGB"), dtype=np.int16)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    return (red > 48) & (red > green * 1.2) & (red > blue * 1.2)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Foreground mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_foreground(pair: CapturePair) -> tuple[Image.Image, Image.Image]:
    image = Image.open(pair.image).convert("RGB")
    layer = Image.open(pair.layer).convert("RGB")
    mask_arr = red_foreground_mask(layer)
    bbox = mask_bbox(mask_arr)

    foreground = image.crop(bbox).convert("RGBA")
    alpha = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L").crop(bbox)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=0.7))
    foreground.putalpha(alpha)
    return foreground, alpha


def random_scale(rng: random.Random, fg: Image.Image, canvas_size: int) -> float:
    max_side = max(fg.size)
    lower = min(0.45, canvas_size / max_side)
    upper = min(1.05, canvas_size / max_side)
    return rng.uniform(lower, upper)


def resize_foreground(foreground: Image.Image, alpha: Image.Image, scale: float) -> tuple[Image.Image, Image.Image]:
    width = max(1, round(foreground.width * scale))
    height = max(1, round(foreground.height * scale))
    size = (width, height)
    return (
        foreground.resize(size, Image.Resampling.LANCZOS),
        alpha.resize(size, Image.Resampling.LANCZOS),
    )


def paste_location(rng: random.Random, canvas_size: int, fg_size: tuple[int, int]) -> tuple[int, int]:
    fg_w, fg_h = fg_size
    max_x = max(0, canvas_size - fg_w)
    max_y = max(0, canvas_size - fg_h)
    return rng.randint(0, max_x), rng.randint(0, max_y)


def save_label_map(output_dir: Path) -> None:
    (output_dir / "label_map.pbtxt").write_text("item {\n    id: 1\n    name: 'bulldozer'\n}\n", encoding="utf-8")


def yolo_row(xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int) -> str:
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"


def save_yolo_dataset_yaml(output_dir: Path) -> None:
    content = "\n".join(
        [
            f"path: {output_dir.resolve()}",
            "train: train/generated",
            "val: test/generated",
            "",
            "names:",
            f"  0: {CLASS_NAME}",
            "",
        ]
    )
    (output_dir / "dataset.yaml").write_text(content, encoding="utf-8")


def compose_split(
    *,
    split: str,
    count: int,
    pairs: list[CapturePair],
    backgrounds: list[str],
    zip_path: Path,
    output_dir: Path,
    size: int,
    rng: random.Random,
    jpeg_quality: int,
    write_yolo: bool,
) -> list[Path]:
    split_dir = output_dir / split
    generated_dir = split_dir / "generated"
    mask_dir = split_dir / "mask"
    annotation_dir = split_dir / "annotation"
    label_dir = split_dir / "labels"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    directories = [generated_dir, mask_dir, annotation_dir]
    if write_yolo:
        directories.append(label_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = []
    preview_paths: list[Path] = []

    for idx in range(count):
        pair = rng.choice(pairs)
        for _ in range(25):
            background_name = rng.choice(backgrounds)
            try:
                canvas = load_background(zip_path, background_name, size)
                break
            except (OSError, zipfile.BadZipFile):
                continue
        else:
            raise RuntimeError("Could not load a valid background image after 25 attempts")

        foreground, alpha = crop_foreground(pair)
        scale = random_scale(rng, foreground, size)
        foreground, alpha = resize_foreground(foreground, alpha, scale)
        x, y = paste_location(rng, size, foreground.size)

        canvas_rgba = canvas.convert("RGBA")
        canvas_rgba.alpha_composite(foreground, (x, y))

        mask_canvas = Image.new("L", (size, size), 0)
        mask_canvas.paste(alpha, (x, y))
        bbox_arr = np.asarray(mask_canvas) > 16
        xmin, ymin, xmax, ymax = mask_bbox(bbox_arr)

        image_name = f"image_{idx:06d}_gen.jpg"
        mask_name = f"image_{idx:06d}_mask.png"
        image_path = generated_dir / image_name
        mask_path = mask_dir / mask_name

        canvas_rgba.convert("RGB").save(image_path, quality=jpeg_quality, optimize=True)
        mask_canvas.save(mask_path)
        if write_yolo:
            (label_dir / f"image_{idx:06d}_gen.txt").write_text(
                yolo_row(xmin, ymin, xmax, ymax, size, size),
                encoding="utf-8",
            )

        rows.append([idx, image_name, size, size, xmin, ymin, xmax, ymax, CLASS_NAME, mask_name])
        if len(preview_paths) < 12:
            preview_paths.append(image_path)

    with (annotation_dir / "annotations.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    return preview_paths


def save_preview(paths: list[Path], output_dir: Path, size: int) -> None:
    if not paths:
        return
    thumb_size = 160
    columns = 4
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size, rows * thumb_size), "white")
    for idx, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        x = (idx % columns) * thumb_size
        y = (idx // columns) * thumb_size
        sheet.paste(image, (x, y))
    sheet.save(output_dir / "preview.jpg", quality=92)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = list_backgrounds(args.backgrounds_zip)
    train_pairs = find_pairs(args.captures_dir, "train")
    test_pairs = find_pairs(args.captures_dir, "test")

    preview_paths: list[Path] = []
    preview_paths.extend(
        compose_split(
            split="train",
            count=args.train_count,
            pairs=train_pairs,
            backgrounds=backgrounds,
            zip_path=args.backgrounds_zip,
            output_dir=output_dir,
            size=args.size,
            rng=rng,
            jpeg_quality=args.jpeg_quality,
            write_yolo=args.write_yolo,
        )
    )
    preview_paths.extend(
        compose_split(
            split="test",
            count=args.test_count,
            pairs=test_pairs,
            backgrounds=backgrounds,
            zip_path=args.backgrounds_zip,
            output_dir=output_dir,
            size=args.size,
            rng=rng,
            jpeg_quality=args.jpeg_quality,
            write_yolo=args.write_yolo,
        )
    )
    save_label_map(output_dir)
    if args.write_yolo:
        save_yolo_dataset_yaml(output_dir)
    save_preview(preview_paths[: args.preview_count], output_dir, args.size)

    print(f"Wrote dataset to {output_dir.resolve()}")
    print(f"Train rows: {args.train_count}")
    print(f"Test rows: {args.test_count}")
    print(f"Label map: {output_dir / 'label_map.pbtxt'}")
    if args.write_yolo:
        print(f"YOLO dataset: {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
