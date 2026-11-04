# Historical Asset Evaluation Results

Date: 2026-11-04
Method: `notebooks/02_evaluate_historical_assets.ipynb` executed end to end
(all 3 code cells ran without errors, outputs saved in the notebook), plus
direct inspection of the model files. Everything below was observed in that
run; no detection accuracy is claimed, because no labeled evaluation set from
the original project survives.

## TensorFlow.js model (`demo/tfjs/zach_model/`)

- `model.json` format: `graph-model`
- generatedBy: TensorFlow 2.3.0
- convertedBy: TensorFlow.js Converter v2.1.0
- weightsManifest: 1 entry, 5 shard paths
  (`group1-shard1of5.bin` … `group1-shard5of5.bin`), all present
- 468 weight tensors declared in the manifest
- Shard sizes: 4 shards of 4,194,304 bytes + 1 shard of 1,623,752 bytes
  (18,400,968 bytes total, ~17.5 MiB)
- The modern demo in `demo/tfjs` builds against these assets
  (`npm run build` passes, verified 2026-11-04)

## Historical TensorFlow artifact

- `artifacts/historical-tensorflow/frozen_inference_graph_100k.pb`:
  19,156,874 bytes (~19 MB), sha256 prefix `b4d15151337d403a`
- Kept as provenance for the original TensorFlow Object Detection API model;
  the browser demo uses the TFJS export above, not this file.

## Reference photos

- `demo/tfjs/static/img_2444.jpg`, `dozer_zach.jpg`, and `dozer2.jpg` all
  exist and render (displayed by the notebook during the run). They document
  the likely physical model; they are not a photogrammetry set.

## What this evaluation does not establish

- No mAP or accuracy figure: the original labeled dataset is lost, and the
  regenerated composite dataset (see `docs/dataset-regeneration.md`) is a
  reconstruction, not the original test set.
- No retraining was performed. The model above is the preserved 2021 export.
