# Recovery Notes

## Surviving Inputs

- `data/captures/train` and `data/captures/test` contain recovered render/mask
  pairs.
- Each `*_img.png` file has a matching `*_layer.png` file.
- The layer image encodes the foreground object in red.
- `assets/reference/` contains original/reference photos of the physical model.
- `demo/tfjs/zach_model/` contains the exported TensorFlow.js graph model.

## Reconstructed Dataset Pipeline

The dataset reconstruction script:

1. Finds render/mask pairs.
2. Extracts the red foreground mask.
3. Crops the dozer foreground.
4. Composites the object onto random COCO backgrounds.
5. Writes generated images, binary masks, bounding boxes, and `label_map.pbtxt`.

The output CSV schema is:

```text
id,image,width,height,xmin,ymin,xmax,ymax,class,mask
```

## Historical Demo

The TFJS demo is based on the 2021 `Ely-S/EfficientDetJS` branch
`zach-model-revision`. The modern version in this repo keeps the recovered
model assets but uses a Vite entry point for a more maintainable local demo.
