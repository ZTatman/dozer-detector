# Dozer Detector Recovery

This is a cleaned-up portfolio version of an old synthetic-data object-detection
project. The original private dataset notebook and source 3D asset were lost,
but enough artifacts survived to reconstruct the dataset pipeline and preserve
the historical TensorFlow.js demo.

## What This Repo Shows

- Recovery of a partial machine-vision project from old files.
- Reproducible synthetic dataset generation from recovered render/mask pairs.
- Dataset validation for generated images, masks, classes, and bounding boxes.
- A preserved and modernized browser demo using the historical exported TFJS
  model.
- Clear documentation of what is reproducible today and what remains historical
  context.

## Project Structure

```text
assets/
  reference/                Original/reference photos of the dozer model
  previews/                 Capture contact sheets
data/
  captures/                 Recovered render and RGB mask pairs
  external/                 Ignored local downloads, such as COCO backgrounds
blender/                    Optional ignored local Blender scenes
models/                     Optional ignored local 3D model sources
demo/
  tfjs/                     Modernized browser demo using historical TFJS model
artifacts/
  historical-tensorflow/    Preserved frozen graph from the 2021 handoff
docs/
  recovery-notes.md         Provenance and reconstruction notes
  limitations.md            Honest project limitations
  synthetic-training-pipeline.md
                             Blender capture and training plan
notebooks/
  01_reconstruct_dataset.ipynb
  02_evaluate_historical_assets.ipynb
scripts/
  rebuild_dozer_dataset.py
  render_blender_captures.py
  validate_dataset.py
```

Generated datasets and large downloaded archives are intentionally ignored by
Git.

## Rebuild the Dataset

Download or place COCO validation images at `data/external/val2017.zip`, then
run:

```bash
python scripts/rebuild_dozer_dataset.py
python scripts/validate_dataset.py dataset --expected-train 1000 --expected-test 200
```

To also write YOLO labels for modern detector training:

```bash
python scripts/rebuild_dozer_dataset.py --write-yolo
python scripts/validate_dataset.py dataset --expected-train 1000 --expected-test 200 --check-yolo
```

Fast smoke test:

```bash
python scripts/rebuild_dozer_dataset.py --output-dir dataset_smoke --train-count 8 --test-count 4 --preview-count 8
python scripts/validate_dataset.py dataset_smoke --expected-train 8 --expected-test 4
```

## Optional Blender Captures

The cleaned portfolio repo keeps the recovered capture pairs in `data/captures`.
If you also have local Blender/model assets available at `blender/dozer.blend`,
you can generate fresh capture pairs before rebuilding a dataset:

```bash
blender --background blender/dozer.blend --python scripts/render_blender_captures.py -- \
  --output-dir data/captures/blender_generated \
  --train-count 300 \
  --test-count 60 \
  --size 768

python scripts/rebuild_dozer_dataset.py \
  --captures-dir data/captures/blender_generated \
  --output-dir dataset_blender \
  --train-count 1000 \
  --test-count 200 \
  --write-yolo
```

See `docs/synthetic-training-pipeline.md` for the EfficientDet/YOLO training
workflow.

## Run the Modern Demo

The modern demo lives in `demo/tfjs` and uses the recovered model in
`demo/tfjs/zach_model`.

```bash
cd demo/tfjs
npm install
npm run dev
```

Then open the local Vite URL printed by the terminal.

## Historical Context

The exported TFJS model and demo came from the 2021
`Ely-S/EfficientDetJS` `zach-model-revision` branch. This repo treats that
branch as historical provenance and repackages the demo in a cleaner portfolio
context.

## Limitations

- The original private `dozer_dataset.ipynb` was not recovered.
- The original GLB/GLTF or physical model was not recovered.
- The likely physical reference was a Huina/Hui Na 1700 or CY1700 1:50 die-cast
  bulldozer, inferred from surviving photos and product images.
- Full retraining with a licensed or scanned replacement model remains future
  work.

## Next Phase

Later work can add a licensed or scanned replacement 3D model, generate a new
synthetic dataset from Blender, and retrain a modern detector.
