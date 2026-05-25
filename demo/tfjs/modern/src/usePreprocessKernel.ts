import * as tf from "@tensorflow/tfjs";
import { useEffect, useMemo, useRef } from "react";
import { getGaussianKernel } from "../../blur";

function getBlurKernelSize(value: number): number {
  return value <= 0 ? 0 : value * 2 + 1;
}

export function usePreprocessKernel(
  applyBlur: boolean,
  blurValue: number,
): {
  blurKernelSize: number;
  kernelRef: { current: tf.Tensor4D | null };
  shouldApplyBlur: boolean;
} {
  const kernelRef = useRef<tf.Tensor4D | null>(null);
  const blurKernelSize = useMemo(() => getBlurKernelSize(blurValue), [blurValue]);
  const shouldApplyBlur = applyBlur && blurKernelSize > 0;

  useEffect(() => {
    kernelRef.current?.dispose();
    kernelRef.current = blurKernelSize > 0 ? getGaussianKernel(blurKernelSize) : null;

    return () => {
      kernelRef.current?.dispose();
      kernelRef.current = null;
    };
  }, [blurKernelSize]);

  return {
    blurKernelSize,
    kernelRef,
    shouldApplyBlur,
  };
}
