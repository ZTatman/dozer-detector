import { useEffect, useRef, useState } from "react";

type VideoFrameCallbackMetadata = {
  presentedFrames: number;
};

type VideoFrameCallback = (
  now: DOMHighResTimeStamp,
  metadata: VideoFrameCallbackMetadata,
) => void;

type VideoElementWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (callback: VideoFrameCallback) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

type UseMediaFpsOptions = {
  enabled: boolean;
  videoRef: { current: HTMLVideoElement | null };
};

export function useMediaFps({ enabled, videoRef }: UseMediaFpsOptions): number {
  const [fps, setFps] = useState(0);
  const smoothedFpsRef = useRef(0);

  useEffect(() => {
    const video = videoRef.current as VideoElementWithFrameCallback | null;
    smoothedFpsRef.current = 0;
    setFps(0);

    if (!enabled || !video || !video.requestVideoFrameCallback) {
      return;
    }

    let cancelled = false;
    let callbackHandle: number | null = null;
    let lastPresentedFrames = 0;
    let lastSampleTime = performance.now();

    const onFrame: VideoFrameCallback = (now, metadata) => {
      if (cancelled) {
        return;
      }

      const presentedFrames = metadata.presentedFrames;
      const frameDelta = presentedFrames - lastPresentedFrames;
      const elapsed = now - lastSampleTime;

      if (lastPresentedFrames > 0 && frameDelta > 0 && elapsed >= 250) {
        const instantFps = (frameDelta * 1000) / elapsed;
        smoothedFpsRef.current = smoothedFpsRef.current === 0
          ? instantFps
          : smoothedFpsRef.current * 0.75 + instantFps * 0.25;
        setFps(Number(smoothedFpsRef.current.toFixed(1)));
        lastSampleTime = now;
        lastPresentedFrames = presentedFrames;
      } else if (lastPresentedFrames === 0) {
        lastPresentedFrames = presentedFrames;
        lastSampleTime = now;
      }

      callbackHandle = video.requestVideoFrameCallback?.(onFrame) ?? null;
    };

    callbackHandle = video.requestVideoFrameCallback(onFrame);

    return () => {
      cancelled = true;
      if (callbackHandle !== null) {
        video.cancelVideoFrameCallback?.(callbackHandle);
      }
    };
  }, [enabled, videoRef]);

  return fps;
}
