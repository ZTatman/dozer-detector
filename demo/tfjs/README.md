# Modern TFJS Demo

This folder contains a cleaned-up browser demo for the recovered dozer detector.
It keeps the historical exported model in `zach_model/` and uses a modern Vite
entry point in `modern/`.

## Run

```bash
npm install
npm run dev
```

Open `/modern/index.html` if Vite does not route there automatically.

Production build:

```bash
npm run build
```

The demo loads:

- `zach_model/model.json`
- `zach_model/group1-shard*.bin`
- reference images from `static/`

## Provenance

The original demo came from the `Ely-S/EfficientDetJS` branch
`zach-model-revision`. The modern Vite files are added for maintainability; the
model assets are preserved as recovered historical artifacts.
