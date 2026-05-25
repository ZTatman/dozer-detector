# Code Organization Refactor

## Goal
Split `App.tsx` (670 lines) into focused files with clear responsibilities.

## File Structure
```
src/
├── App.tsx              — Suspense shell, exports default
├── ModelApp.tsx         — orchestrator: state, refs, callbacks, effects
├── CanvasStage.tsx       — pure: two canvas <figure> panels
├── ControlPanel.tsx      — pure: sidebar (source, blur, metrics, button)
└── constants.ts          — shared constants and types
```

## Component Boundaries

### App.tsx
- Imports `ModelApp`
- Wraps it in `<Suspense fallback={<LoadingFallback />}>`
- `LoadingFallback` inlined (tiny, ~12 lines)

### ModelApp.tsx
- Keeps everything that touches TF.js, refs, state, effects, and imperative canvas logic
- Exports `Metrics` type for use by sibling components
- Renders: `<header>`, `<video>`, `<CanvasStage>`, `<ControlPanel>`

### CanvasStage.tsx
- Receives `preprocessCanvasRef`, `previewCanvasRef` as props
- Renders the two `<figure>` panels
- Zero state, zero effects, zero refs of its own

### ControlPanel.tsx
- Receives all values + change handlers as props (~12 props)
- Renders: source select, preprocess controls, metrics grid, action button
- Zero state, zero effects, zero refs of its own

## Data Flow
```
ModelApp (state owner)
  ├─ CanvasStage  ← { preprocessCanvasRef, previewCanvasRef }
  └─ ControlPanel ← { sourceValue, applyBlur, blurValue, ..., onAction }
```

All mutable state (refs, useState) stays in `ModelApp`. `CanvasStage` and `ControlPanel` are pure presentational components.
