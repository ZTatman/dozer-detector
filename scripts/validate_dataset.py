#!/usr/bin/env python3
"""Validate a reconstructed dozer dataset.

The validator checks the generated dataset shape used by the recovered
notebooks:

  dataset/
    label_map.pbtxt
    train/generated/*.jpg
    train/mask/*.png
    train/annotation/annotations.csv
    test/generated/*.jpg
    test/mask/*.png
    test/annotation/annotations.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


REQUIRED_COLUMNS = ["id", "image", "width", "height", "xmin", "ymin", "xmax", "ymax", "class", "mask"]


@dataclass
class SplitReport:
    split: str
    rows: int
    images: int
    masks: int
    errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Dataset directory to validate.")
    parser.add_argument("--expected-train", type=int, default=None)
    parser.add_argument("--expected-test", type=int, default=None)
    parser.add_argument("--class-name", default="bulldozer")
    parser.add_argument("--check-yolo", action="store_true", help="Also validate YOLO label files and dataset.yaml.")
    return parser.parse_args()


def read_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    if not csv_path.exists():
        return [], [f"Missing CSV: {csv_path}"]

    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            errors.append(f"{csv_path}: missing columns {missing}")
        rows = list(reader)

    return rows, errors


def positive_int(value: str, field: str, row_index: int, errors: list[str]) -> int | None:
    try:
        parsed = int(value)
    except ValueError:
        errors.append(f"row {row_index}: {field} is not an integer: {value!r}")
        return None
    if parsed < 0:
        errors.append(f"row {row_index}: {field} is negative: {parsed}")
    return parsed


def validate_row(
    *,
    row: dict[str, str],
    row_index: int,
    generated_dir: Path,
    mask_dir: Path,
    class_name: str,
    errors: list[str],
) -> None:
    image_path = generated_dir / row.get("image", "")
    mask_path = mask_dir / row.get("mask", "")

    if not image_path.exists():
        errors.append(f"row {row_index}: missing image {image_path}")
        return
    if not mask_path.exists():
        errors.append(f"row {row_index}: missing mask {mask_path}")
        return

    width = positive_int(row.get("width", ""), "width", row_index, errors)
    height = positive_int(row.get("height", ""), "height", row_index, errors)
    xmin = positive_int(row.get("xmin", ""), "xmin", row_index, errors)
    ymin = positive_int(row.get("ymin", ""), "ymin", row_index, errors)
    xmax = positive_int(row.get("xmax", ""), "xmax", row_index, errors)
    ymax = positive_int(row.get("ymax", ""), "ymax", row_index, errors)

    if None in (width, height, xmin, ymin, xmax, ymax):
        return

    assert width is not None
    assert height is not None
    assert xmin is not None
    assert ymin is not None
    assert xmax is not None
    assert ymax is not None

    if row.get("class") != class_name:
        errors.append(f"row {row_index}: class is {row.get('class')!r}, expected {class_name!r}")

    if not (0 <= xmin < xmax <= width):
        errors.append(f"row {row_index}: invalid x bounds {xmin}, {xmax} for width {width}")
    if not (0 <= ymin < ymax <= height):
        errors.append(f"row {row_index}: invalid y bounds {ymin}, {ymax} for height {height}")

    with Image.open(image_path) as image:
        if image.size != (width, height):
            errors.append(f"row {row_index}: image size {image.size} does not match CSV {(width, height)}")
    with Image.open(mask_path) as mask:
        if mask.size != (width, height):
            errors.append(f"row {row_index}: mask size {mask.size} does not match CSV {(width, height)}")


def validate_yolo_label(label_path: Path, row_index: int, errors: list[str]) -> None:
    if not label_path.exists():
        errors.append(f"row {row_index}: missing YOLO label {label_path}")
        return
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        errors.append(f"row {row_index}: expected one YOLO row in {label_path}, found {len(lines)}")
        return
    parts = lines[0].split()
    if len(parts) != 5:
        errors.append(f"row {row_index}: YOLO row should have 5 fields in {label_path}")
        return
    if parts[0] != "0":
        errors.append(f"row {row_index}: YOLO class should be 0 in {label_path}, found {parts[0]!r}")
    for value in parts[1:]:
        try:
            number = float(value)
        except ValueError:
            errors.append(f"row {row_index}: YOLO value is not a float in {label_path}: {value!r}")
            continue
        if not 0 <= number <= 1:
            errors.append(f"row {row_index}: YOLO value out of range in {label_path}: {number}")


def validate_split(
    dataset_dir: Path,
    split: str,
    expected: int | None,
    class_name: str,
    check_yolo: bool,
) -> SplitReport:
    split_dir = dataset_dir / split
    generated_dir = split_dir / "generated"
    mask_dir = split_dir / "mask"
    annotation_dir = split_dir / "annotation"
    label_dir = split_dir / "labels"
    csv_path = annotation_dir / "annotations.csv"

    errors: list[str] = []
    for directory in (generated_dir, mask_dir, annotation_dir):
        if not directory.exists():
            errors.append(f"Missing directory: {directory}")
    if check_yolo and not label_dir.exists():
        errors.append(f"Missing directory: {label_dir}")

    rows, csv_errors = read_rows(csv_path)
    errors.extend(csv_errors)

    image_count = len(list(generated_dir.glob("*"))) if generated_dir.exists() else 0
    mask_count = len(list(mask_dir.glob("*"))) if mask_dir.exists() else 0
    label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0

    if expected is not None and len(rows) != expected:
        errors.append(f"{split}: expected {expected} rows, found {len(rows)}")
    if image_count != len(rows):
        errors.append(f"{split}: image count {image_count} does not match row count {len(rows)}")
    if mask_count != len(rows):
        errors.append(f"{split}: mask count {mask_count} does not match row count {len(rows)}")
    if check_yolo and label_count != len(rows):
        errors.append(f"{split}: YOLO label count {label_count} does not match row count {len(rows)}")

    for index, row in enumerate(rows, start=2):
        validate_row(
            row=row,
            row_index=index,
            generated_dir=generated_dir,
            mask_dir=mask_dir,
            class_name=class_name,
            errors=errors,
        )
        if check_yolo:
            image_name = row.get("image", "")
            validate_yolo_label(label_dir / Path(image_name).with_suffix(".txt").name, index, errors)

    return SplitReport(split=split, rows=len(rows), images=image_count, masks=mask_count, errors=errors)


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir

    errors: list[str] = []
    if not dataset_dir.exists():
        errors.append(f"Dataset directory does not exist: {dataset_dir}")
    if not (dataset_dir / "label_map.pbtxt").exists():
        errors.append(f"Missing label map: {dataset_dir / 'label_map.pbtxt'}")
    if args.check_yolo and not (dataset_dir / "dataset.yaml").exists():
        errors.append(f"Missing YOLO dataset yaml: {dataset_dir / 'dataset.yaml'}")

    reports = [
        validate_split(dataset_dir, "train", args.expected_train, args.class_name, args.check_yolo),
        validate_split(dataset_dir, "test", args.expected_test, args.class_name, args.check_yolo),
    ]

    for report in reports:
        print(f"{report.split}: rows={report.rows} images={report.images} masks={report.masks}")
        errors.extend(report.errors)

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
