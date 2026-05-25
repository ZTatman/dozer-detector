import { useCallback, useRef, useState } from "react";

type UseDetectionLoopOptions = {
  clearMetrics: () => void;
  detectFrameRef: { current: () => Promise<void> };
  onError: (error: unknown) => void;
};

export function useDetectionLoop({
  clearMetrics,
  detectFrameRef,
  onError,
}: UseDetectionLoopOptions): {
  liveDetecting: boolean;
  startDetectionLoop: () => void;
  stopDetectionLoop: () => void;
} {
  const [liveDetecting, setLiveDetecting] = useState(false);
  const activeRef = useRef(false);
  const animationFrameRef = useRef<number | null>(null);

  const stopDetectionLoop = useCallback(() => {
    activeRef.current = false;
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    setLiveDetecting(false);
  }, []);

  const startDetectionLoop = useCallback(() => {
    stopDetectionLoop();
    activeRef.current = true;
    setLiveDetecting(true);
    clearMetrics();

    const loop = async () => {
      if (!activeRef.current) {
        return;
      }

      try {
        await detectFrameRef.current();
      } catch (error) {
        stopDetectionLoop();
        onError(error);
        return;
      }

      if (activeRef.current) {
        animationFrameRef.current = window.requestAnimationFrame(loop);
      }
    };

    void loop();
  }, [clearMetrics, detectFrameRef, onError, stopDetectionLoop]);

  return {
    liveDetecting,
    startDetectionLoop,
    stopDetectionLoop,
  };
}
