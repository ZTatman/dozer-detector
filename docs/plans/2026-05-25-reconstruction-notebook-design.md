# Reconstruction Notebook Design

## Goal

Expand `notebooks/01_reconstruct_dataset.ipynb` into the canonical walkthrough
for recreating the TensorFlow Object Detection API dataset consumed by the
historical `dozer_object_detector` workflow.

## Scope

The notebook explains the path from recovered or freshly generated capture
pairs to the final `dataset/` directory. It focuses on the TensorFlow CSV,
image, mask, and label-map outputs. YOLO export remains available in the script
but stays out of the main notebook flow.

## Flow

1. Resolve the project root and define input/output paths.
2. Verify recovered capture pairs and the local COCO background archive.
3. Display example `*_img.png` and `*_layer.png` capture pairs.
4. Extract the red foreground layer into a binary mask and bounding box.
5. Composite one foreground onto a COCO background to show the core method.
6. Generate and validate a small `dataset_smoke/` output for fast debugging.
7. Generate and validate a larger `dataset/` output for training.
8. Inspect the final directory shape, preview image, label map, and CSV rows.

## Dataset Sizes

The smoke path uses `8` training images and `4` test images so the notebook can
be run quickly while developing. The training path uses `10,000` training
images and `2,000` test images by default.

The larger dataset improves background, placement, and scale variation, but it
does not fully replace viewpoint diversity because the recovered foreground
captures are limited. Fresh Blender captures can be generated separately before
running this notebook if more foreground diversity is needed.
