import type { CSSProperties } from "react";
import * as React from "react";
import { memo } from "react";
import { CANVAS_WIDTH, CANVAS_HEIGHT } from "./constants";

type CanvasStageProps = {
  preprocessCanvasRef: { current: HTMLCanvasElement | null };
  previewCanvasRef: { current: HTMLCanvasElement | null };
  preprocessOverlay: string;
  detectionOverlay: string;
};

const overlayBase: CSSProperties = {
  position: "absolute",
  top: 10,
  right: 10,
  font: "600 13px Inter, sans-serif",
  pointerEvents: "none",
  zIndex: 1,
};

const CanvasStage = memo(function CanvasStage({
  preprocessCanvasRef,
  previewCanvasRef,
  preprocessOverlay,
  detectionOverlay,
}: CanvasStageProps): JSX.Element {
  return (
    <div className="stage">
      <figure className="stage-panel" style={{ position: "relative" }}>
        <figcaption>Preprocessed</figcaption>
        <canvas ref={preprocessCanvasRef} id="preprocessCanvas" width={CANVAS_WIDTH} height={CANVAS_HEIGHT} />
        <span style={{ ...overlayBase, color: "#d38b18" }}>{preprocessOverlay}</span>
      </figure>
      <figure className="stage-panel" style={{ position: "relative" }}>
        <figcaption>Detections</figcaption>
        <canvas ref={previewCanvasRef} id="previewCanvas" width={CANVAS_WIDTH} height={CANVAS_HEIGHT} />
        <span style={{ ...overlayBase, color: "#ff5555" }}>{detectionOverlay}</span>
      </figure>
    </div>
  );
});

export default CanvasStage;
