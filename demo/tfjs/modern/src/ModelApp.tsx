import * as tf from "@tensorflow/tfjs";
import * as React from "react";
import { use, useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { blur } from "../../blur";
import { drawBoxes } from "../../drawBoxes";
import getDetections from "../../postprocess";
import CanvasStage from "./CanvasStage";
import ControlPanel from "./ControlPanel";
import { useDetectionLoop } from "./useDetectionLoop";
import { useMediaFps } from "./useMediaFps";
import { usePreprocessKernel } from "./usePreprocessKernel";
import {
  MODEL_URL,
  CANVAS_WIDTH,
  CANVAS_HEIGHT,
  CAMERA_SOURCE,
  CAMERA_TIMEOUT_MS,
  OUTPUT_NODES,
  EXAMPLES,
  EMPTY_METRICS,
} from "./constants";
import type { Metrics } from "./constants";

export type { Metrics };

type Detection = {
  bbox: [number, number, number, number];
  class: string;
  score: number;
};

type UiState = {
  inputReady: boolean;
  running: boolean;
  statusOverride: string | null;
};

const initialUiState: UiState = {
  inputReady: false,
  running: false,
  statusOverride: null,
};

type UiAction =
  | { type: "SET_STATUS"; message: string | null }
  | { type: "INPUT_LOADING"; message: string }
  | { type: "IMAGE_LOADED" }
  | { type: "CAMERA_READY" }
  | { type: "RESET" }
  | { type: "RUN_START" }
  | { type: "RUN_DONE" }
  | { type: "RUN_ERROR"; message: string };

function uiReducer(state: UiState, action: UiAction): UiState {
  switch (action.type) {
    case "SET_STATUS":
      return { ...state, statusOverride: action.message };
    case "INPUT_LOADING":
      return { ...state, inputReady: false, statusOverride: action.message };
    case "IMAGE_LOADED":
      return { ...state, inputReady: true, statusOverride: null };
    case "CAMERA_READY":
      return { ...state, inputReady: true, statusOverride: null };
    case "RESET":
      return { ...initialUiState };
    case "RUN_START":
      return { ...state, running: true, statusOverride: "Running inference..." };
    case "RUN_DONE":
      return { ...state, running: false, statusOverride: null };
    case "RUN_ERROR":
      return { ...state, running: false, statusOverride: action.message };
  }
}

const modelPromise: Promise<{ model: tf.GraphModel; backend: string }> = (async () => {
  await tf.ready();
  if (tf.findBackend("webgl")) {
    await tf.setBackend("webgl");
  }
  const backend = tf.getBackend();
  const model = await tf.loadGraphModel(MODEL_URL);
  return { model, backend };
})();

function formatMs(start: number, end: number): string {
  return `${Math.round(end - start)} ms`;
}

async function loadImage(src: string): Promise<HTMLImageElement> {
  const image = new Image();
  image.crossOrigin = "anonymous";
  image.decoding = "async";
  image.src = src;
  await image.decode();
  return image;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error(message)), timeoutMs);
    promise
      .then((value) => resolve(value))
      .catch((error) => reject(error))
      .finally(() => window.clearTimeout(timeout));
  });
}

function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof DOMException && error.name) {
    return error.message ? `${error.name}: ${error.message}` : error.name;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return fallback;
}

function waitForCameraReady(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", handleReady);
      video.removeEventListener("canplay", handleReady);
      video.removeEventListener("error", handleError);
    };

    const handleReady = () => {
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
        cleanup();
        resolve();
      }
    };

    const handleError = () => {
      cleanup();
      reject(new Error("Camera video failed to start"));
    };

    video.addEventListener("loadedmetadata", handleReady);
    video.addEventListener("canplay", handleReady);
    video.addEventListener("error", handleError);
  });
}

function resetCanvas(targetCanvas: HTMLCanvasElement): void {
  const ctx = targetCanvas.getContext("2d");
  if (!ctx) {
    throw new Error("Canvas 2D context is unavailable");
  }

  targetCanvas.width = CANVAS_WIDTH;
  targetCanvas.height = CANVAS_HEIGHT;
  ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  ctx.fillStyle = "#f5f7f8";
  ctx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
}

function drawImageToCanvas(image: HTMLImageElement, targetCanvas: HTMLCanvasElement): void {
  const ctx = targetCanvas.getContext("2d");
  if (!ctx) {
    throw new Error("Canvas 2D context is unavailable");
  }

  const sourceWidth = image.naturalWidth;
  const sourceHeight = image.naturalHeight;
  if (sourceWidth === 0 || sourceHeight === 0) {
    throw new Error("Image is not ready");
  }

  ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  ctx.fillStyle = "#f5f7f8";
  ctx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);

  const scale = Math.min(targetCanvas.width / sourceWidth, targetCanvas.height / sourceHeight);
  const width = Math.round(sourceWidth * scale);
  const height = Math.round(sourceHeight * scale);
  const x = Math.round((targetCanvas.width - width) / 2);
  const y = Math.round((targetCanvas.height - height) / 2);

  ctx.drawImage(image, x, y, width, height);
}

function drawScaledCanvas(source: HTMLCanvasElement, targetCanvas: HTMLCanvasElement): void {
  const ctx = targetCanvas.getContext("2d");
  if (!ctx) {
    throw new Error("Canvas 2D context is unavailable");
  }

  ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  ctx.drawImage(source, 0, 0, targetCanvas.width, targetCanvas.height);
}

async function drawTensorToCanvas(tensor: tf.Tensor3D, targetCanvas: HTMLCanvasElement): Promise<void> {
  const tempCanvas = document.createElement("canvas");
  const displayTensor = tf.tidy(() => tensor.clipByValue(0, 255).toInt());
  await tf.browser.toPixels(displayTensor, tempCanvas);
  displayTensor.dispose();
  drawScaledCanvas(tempCanvas, targetCanvas);
}

export default function ModelApp(): JSX.Element {
  const { model, backend } = use(modelPromise);
  const [sourceValue, setSourceValue] = useState(EXAMPLES[0].value);
  const [applyBlur, setApplyBlur] = useState(true);
  const [blurValue, setBlurValue] = useState(6);
  const [metrics, setMetrics] = useState<Metrics>(EMPTY_METRICS);
  const [ui, dispatch] = useReducer(uiReducer, initialUiState);
  const { inputReady, running, statusOverride } = ui;

  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const preprocessCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const cameraVideoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const detectFrameRef = useRef<() => Promise<void>>(async () => {});

  const { blurKernelSize, kernelRef, shouldApplyBlur } = usePreprocessKernel(applyBlur, blurValue);
  const isCameraSource = sourceValue === CAMERA_SOURCE;
  const sourceFps = useMediaFps({
    enabled: inputReady && isCameraSource,
    videoRef: cameraVideoRef,
  });

  const clearMetrics = useCallback(() => {
    setMetrics(EMPTY_METRICS);
  }, []);

  const handleLiveDetectionError = useCallback((error: unknown) => {
    console.error(error);
    dispatch({ type: "SET_STATUS", message: getErrorMessage(error, "Live detection failed") });
  }, []);

  const {
    liveDetecting,
    startDetectionLoop,
    stopDetectionLoop,
  } = useDetectionLoop({
    clearMetrics,
    detectFrameRef,
    onError: handleLiveDetectionError,
  });

  const status = useMemo(() => {
    if (statusOverride) {
      return statusOverride;
    }
    if (isCameraSource) {
      if (!inputReady) return "Camera selected";
      return liveDetecting ? "Live detection" : "Camera ready";
    }
    return inputReady ? "Ready" : "Loading image...";
  }, [inputReady, liveDetecting, isCameraSource, statusOverride]);

  const actionDisabled = liveDetecting
    ? false
    : isCameraSource
      ? running
      : running || !inputReady;

  const actionLabel = !isCameraSource
    ? "Run detection"
    : isCameraSource && !inputReady
      ? "Start camera"
      : liveDetecting
        ? "Stop inference"
        : "Start inference";

  const stopLiveDetection = useCallback(() => {
    stopDetectionLoop();
  }, [stopDetectionLoop]);

  const releaseWebcam = useCallback(() => {
    const stream = cameraStreamRef.current;
    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    }
    cameraStreamRef.current = null;
    if (cameraVideoRef.current) {
      cameraVideoRef.current.pause();
      cameraVideoRef.current.srcObject = null;
    }
  }, []);

  const stopCamera = useCallback(() => {
    stopLiveDetection();
    releaseWebcam();
  }, [releaseWebcam, stopLiveDetection]);

  const drawImagePanels = useCallback(async () => {
    const image = imageRef.current;
    const previewCanvas = previewCanvasRef.current;
    const preprocessCanvas = preprocessCanvasRef.current;
    if (!image || !previewCanvas || !preprocessCanvas) return;

    drawImageToCanvas(image, previewCanvas);

    const kernel = kernelRef.current;
    if (shouldApplyBlur && kernel) {
      const temp = document.createElement("canvas");
      temp.width = CANVAS_WIDTH;
      temp.height = CANVAS_HEIGHT;
      const tCtx = temp.getContext("2d")!;
      tCtx.fillStyle = "#f5f7f8";
      tCtx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      const scale = Math.min(CANVAS_WIDTH / image.naturalWidth, CANVAS_HEIGHT / image.naturalHeight);
      const w = Math.round(image.naturalWidth * scale);
      const h = Math.round(image.naturalHeight * scale);
      const x = Math.round((CANVAS_WIDTH - w) / 2);
      const y = Math.round((CANVAS_HEIGHT - h) / 2);
      tCtx.drawImage(image, x, y, w, h);
      const t = tf.browser.fromPixels(temp);
      const b = blur(t, kernel);
      await drawTensorToCanvas(b, preprocessCanvas);
      tf.dispose([t, b]);
    } else {
      drawImageToCanvas(image, preprocessCanvas);
    }
  }, [shouldApplyBlur]);

  const startCamera = useCallback(async () => {
    const video = cameraVideoRef.current;
    const previewCanvas = previewCanvasRef.current;
    const preprocessCanvas = preprocessCanvasRef.current;
    if (!video || !previewCanvas || !preprocessCanvas) {
      throw new Error("Camera preview is not ready");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera is not available in this browser");
    }

    stopLiveDetection();
    releaseWebcam();
    dispatch({ type: "SET_STATUS", message: "Requesting camera permission..." });

    const stream = await withTimeout(
      navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      }),
      CAMERA_TIMEOUT_MS,
      "Camera permission was not granted",
    );

    cameraStreamRef.current = stream;
    video.srcObject = stream;
    await video.play();
    await withTimeout(waitForCameraReady(video), CAMERA_TIMEOUT_MS, "Camera video did not become ready");

    const frame = tf.browser.fromPixels(video);
    await drawTensorToCanvas(frame, previewCanvas);
    await drawTensorToCanvas(frame, preprocessCanvas);
    frame.dispose();

    dispatch({ type: "CAMERA_READY" });
  }, [releaseWebcam, stopLiveDetection]);

  const detectFrame = useCallback(async (): Promise<void> => {
    const previewCanvas = previewCanvasRef.current;
    const preprocessCanvas = preprocessCanvasRef.current;
    if (!previewCanvas || !preprocessCanvas) {
      throw new Error("Detector is not ready");
    }

    let imageTensor: tf.Tensor3D;
    if (isCameraSource) {
      const video = cameraVideoRef.current;
      if (!cameraStreamRef.current || !video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        throw new Error("Camera is not ready");
      }
      imageTensor = tf.browser.fromPixels(video);
    } else {
      const image = imageRef.current;
      if (!image) {
        throw new Error("Image is not ready");
      }
      drawImageToCanvas(image, previewCanvas);
      imageTensor = tf.browser.fromPixels(previewCanvas);
    }

    const start = performance.now();
    const kernel = kernelRef.current;
    const useBlur = shouldApplyBlur && kernel !== null;
    const processedTensor = useBlur ? blur(imageTensor, kernel) : imageTensor;
    const modelInput = useBlur ? processedTensor.toInt() : processedTensor;
    const batch = modelInput.expandDims();
    const preprocessEnd = performance.now();

    const result = await model.executeAsync({ image_tensor: batch }, OUTPUT_NODES);
    const modelEnd = performance.now();

    const tensors = Array.isArray(result) ? result : [result];
    const detections = (await getDetections(tensors, previewCanvas.width, previewCanvas.height)) as Detection[];

    await drawTensorToCanvas(processedTensor, preprocessCanvas);
    if (isCameraSource) {
      await drawTensorToCanvas(imageTensor, previewCanvas);
    } else if (imageRef.current) {
      drawImageToCanvas(imageRef.current, previewCanvas);
    }
    drawBoxes(detections, previewCanvas, false);
    const end = performance.now();

    tf.dispose([
      imageTensor,
      ...(processedTensor === imageTensor ? [] : [processedTensor]),
      ...(modelInput === processedTensor ? [] : [modelInput]),
      batch,
      ...tensors,
    ]);

    setMetrics({
      preprocess: formatMs(start, preprocessEnd),
      model: formatMs(preprocessEnd, modelEnd),
      postprocess: formatMs(modelEnd, end),
      total: formatMs(start, end),
      detection: detections[0] ? `${(detections[0].score * 100).toFixed(1)}%` : "none",
    });
  }, [isCameraSource, model, shouldApplyBlur]);

  detectFrameRef.current = detectFrame;

  const runDetection = useCallback(async () => {
    dispatch({ type: "RUN_START" });
    clearMetrics();

    try {
      await detectFrame();
      dispatch({ type: "RUN_DONE" });
    } catch (error) {
      console.error(error);
      dispatch({ type: "RUN_ERROR", message: getErrorMessage(error, "Inference failed") });
    }
  }, [clearMetrics, detectFrame]);

  const startLiveDetection = useCallback(() => {
    if (isCameraSource && !cameraStreamRef.current) {
      throw new Error("Camera is not ready");
    }

    stopLiveDetection();
    dispatch({ type: "SET_STATUS", message: null });
    startDetectionLoop();
  }, [isCameraSource, startDetectionLoop, stopLiveDetection]);

  const handleAction = useCallback(async () => {
    if (!isCameraSource) {
      await runDetection();
      return;
    }

    if (liveDetecting) {
      stopLiveDetection();
      return;
    }

    if (isCameraSource && !inputReady) {
      dispatch({ type: "RUN_START" });
      clearMetrics();
      try {
        await startCamera();
        startLiveDetection();
      } catch (error) {
        console.error(error);
        stopCamera();
        dispatch({
          type: "SET_STATUS",
          message: getErrorMessage(error, "Failed to start camera"),
        });
      }
      return;
    }

    startLiveDetection();
  }, [
    clearMetrics,
    inputReady,
    isCameraSource,
    liveDetecting,
    runDetection,
    startCamera,
    startLiveDetection,
    stopCamera,
    stopLiveDetection,
  ]);

  useEffect(() => {
    let cancelled = false;

    async function syncSource(): Promise<void> {
      clearMetrics();
      stopCamera();
      imageRef.current = null;
      dispatch({ type: "RESET" });

      const previewCanvas = previewCanvasRef.current;
      const preprocessCanvas = preprocessCanvasRef.current;
      if (previewCanvas) {
        resetCanvas(previewCanvas);
      }
      if (preprocessCanvas) {
        resetCanvas(preprocessCanvas);
      }

      if (sourceValue === CAMERA_SOURCE) {
        return;
      }

      if (!previewCanvas || !preprocessCanvas) {
        dispatch({ type: "SET_STATUS", message: "Canvas is not ready" });
        return;
      }

      dispatch({ type: "INPUT_LOADING", message: "Loading image..." });

      try {
        const image = await loadImage(sourceValue);
        if (cancelled) {
          return;
        }

        imageRef.current = image;
        await drawImagePanels();
        if (cancelled) {
          return;
        }
        dispatch({ type: "IMAGE_LOADED" });
      } catch (error) {
        if (cancelled) {
          return;
        }
        console.error(error);
        dispatch({ type: "SET_STATUS", message: getErrorMessage(error, "Failed to load input") });
      }
    }

    void syncSource();

    return () => {
      cancelled = true;
    };
  }, [clearMetrics, sourceValue, stopCamera]);

  useEffect(() => {
    if (isCameraSource || !imageRef.current) {
      return;
    }

    void drawImagePanels();
    clearMetrics();
  }, [applyBlur, blurValue, clearMetrics, drawImagePanels, isCameraSource]);

  useEffect(() => {
    if (isCameraSource || !inputReady) {
      return;
    }
    runDetection();
  }, [inputReady, isCameraSource, runDetection]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">TensorFlow.js model</p>
          <h1>Dozer Detector</h1>
          <p className="subtitle">Real-time object detection with TensorFlow.js</p>
        </div>
        <div className="status" id="status">
          {status}
        </div>
      </header>

      <section className="workspace">
        <CanvasStage
          preprocessCanvasRef={preprocessCanvasRef}
          previewCanvasRef={previewCanvasRef}
          preprocessOverlay={
            shouldApplyBlur && blurKernelSize > 0
              ? `Blur: σ=${blurValue} (${blurKernelSize}×${blurKernelSize})`
              : "No blur"
          }
          detectionOverlay={
            sourceFps > 0
              ? `${sourceFps >= 10 ? Math.round(sourceFps) : sourceFps.toFixed(1)} FPS`
              : ""
          }
        />

        <video ref={cameraVideoRef} id="cameraVideo" className="camera-video" autoPlay muted playsInline />

        <ControlPanel
          sourceValue={sourceValue}
          onSourceChange={setSourceValue}
          applyBlur={applyBlur}
          onApplyBlurChange={setApplyBlur}
          blurValue={blurValue}
          onBlurValueChange={setBlurValue}
          blurKernelSize={blurKernelSize}
          backend={backend}
          metrics={metrics}
          actionDisabled={actionDisabled}
          actionLabel={actionLabel}
          onAction={handleAction}
        />
      </section>
    </main>
  );
}
