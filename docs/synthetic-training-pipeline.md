# Synthetic Bulldozer Training Pipeline

## Goal

Generate a reproducible synthetic dataset from the three Blender bulldozer
models, composite those foregrounds onto COCO backgrounds, and train detectors
that recognize real bulldozers.

## Recommended Outputs

The dataset generator should emit two annotation formats from the same
mask-derived boxes:

- TensorFlow Object Detection API CSV/TFRecord layout for the recovered
  notebook and EfficientDet-D0 experiments.
- YOLO detection labels for a modern baseline with simpler training and export.

This avoids choosing a model too early. The same rendered images can be used to
compare legacy TensorFlow training against a current detector.

## Capture Stage

Run Blender in background mode with `scripts/render_blender_captures.py`.

```bash
blender --background blender/dozer.blend --python scripts/render_blender_captures.py -- \
  --output-dir data/captures/blender_generated \
  --train-count 300 \
  --test-count 60 \
  --size 768
```

The script randomly chooses one of the three top-level model roots, isolates it,
randomizes the camera and area lights, and writes paired files:

```text
data/captures/blender_generated/train/image_000000_img.png
data/captures/blender_generated/train/image_000000_layer.png
```

The `*_img.png` files preserve the original model materials. The `*_layer.png`
files render the visible model as red, which the compositor thresholds into a
foreground mask.

## Compositing Stage

Use the existing COCO validation archive as random backgrounds.

```bash
python3 scripts/rebuild_dozer_dataset.py \
  --captures-dir data/captures/blender_generated \
  --backgrounds-zip data/external/val2017.zip \
  --output-dir dataset_blender \
  --train-count 1000 \
  --test-count 200 \
  --write-yolo
```

The TensorFlow-compatible output remains:

```text
dataset_blender/label_map.pbtxt
dataset_blender/train/generated/*.jpg
dataset_blender/train/mask/*.png
dataset_blender/train/annotation/annotations.csv
dataset_blender/test/generated/*.jpg
dataset_blender/test/mask/*.png
dataset_blender/test/annotation/annotations.csv
```

With `--write-yolo`, the script also writes:

```text
dataset_blender/dataset.yaml
dataset_blender/train/labels/*.txt
dataset_blender/test/labels/*.txt
```

## Model Choices

Use EfficientDet-D0 when the objective is continuity with the TensorFlow Object
Detection API workflow. It is pretrained on COCO, uses 512x512 inputs, and is a
reasonable first EfficientDet variant for a single bulldozer class.

Use Ultralytics YOLO as the practical baseline. It is easier to train, inspect,
and export, and should be compared against EfficientDet on real bulldozer
validation photos.

## Validation

Synthetic-only metrics are not enough. Keep a small hand-labeled real-photo
validation set and measure both models on that set. The Blender generator should
be judged by whether real-photo validation improves, not only by whether
synthetic boxes look clean.
