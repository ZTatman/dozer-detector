import * as React from "react";
import { memo } from "react";
import { EXAMPLES } from "./constants";
import type { Metrics } from "./constants";

type ControlPanelProps = {
  sourceValue: string;
  onSourceChange: (value: string) => void;
  applyBlur: boolean;
  onApplyBlurChange: (value: boolean) => void;
  blurValue: number;
  onBlurValueChange: (value: number) => void;
  blurKernelSize: number;
  backend: string;
  metrics: Metrics;
  actionDisabled: boolean;
  actionLabel: string;
  onAction: () => void;
};

const ControlPanel = memo(function ControlPanel({
  sourceValue,
  onSourceChange,
  applyBlur,
  onApplyBlurChange,
  blurValue,
  onBlurValueChange,
  blurKernelSize,
  backend,
  metrics,
  actionDisabled,
  actionLabel,
  onAction,
}: ControlPanelProps): JSX.Element {
  return (
    <aside className="panel">
      <div className="control-group">
        <label htmlFor="imageSelect">Input source</label>
        <select
          id="imageSelect"
          autoComplete="off"
          value={sourceValue}
          onChange={(event) => onSourceChange(event.target.value)}
        >
          {EXAMPLES.map((example) => (
            <option key={example.value} value={example.value}>
              {example.label}
            </option>
          ))}
        </select>
      </div>

      <section className="preprocess-controls" aria-labelledby="preprocessHeading">
        <label className="preprocess-heading" htmlFor="applyBlurInput">
          <input
            id="applyBlurInput"
            type="checkbox"
            checked={applyBlur}
            onChange={(event) => onApplyBlurChange(event.target.checked)}
          />
          <span id="preprocessHeading">Preprocess</span>
        </label>
        <div className="control-group">
          <label htmlFor="blurInput">
            Gaussian blur
            <span className="tooltip-trigger" aria-label="More info">?
              <span className="tooltip">Blurs the input image to reduce noise and smooth edges before detection</span>
            </span>
          </label>
          <input
            id="blurInput"
            type="range"
            min="0"
            max="15"
            step="1"
            value={blurValue}
            disabled={!applyBlur}
            onChange={(event) => onBlurValueChange(Number(event.target.value))}
          />
          <output id="blurValue">
            {!applyBlur || blurKernelSize === 0 ? "Off" : `σ = ${blurValue}`}
          </output>
        </div>
      </section>

      <section className="detection-controls" aria-labelledby="detectionHeading">
        <label className="detection-heading" id="detectionHeading">Detection</label>
        <p className="detection-note">Single-class model (bulldozers only)</p>
      </section>

      <div className="metrics">
        <div>
          <span>Backend</span>
          <strong id="backendMetric">{backend}</strong>
        </div>
        <div>
          <span>Preprocess</span>
          <strong id="preprocessMetric">{metrics.preprocess}</strong>
        </div>
        <div>
          <span>Model</span>
          <strong id="modelMetric">{metrics.model}</strong>
        </div>
        <div>
          <span>Postprocess / draw</span>
          <strong id="postprocessMetric">{metrics.postprocess}</strong>
        </div>
        <div>
          <span>Total</span>
          <strong id="timeMetric">{metrics.total}</strong>
        </div>
        <div>
          <span>Detection</span>
          <strong id="scoreMetric">{metrics.detection}</strong>
        </div>
      </div>

      <button id="runButton" type="button" tabIndex={-1} disabled={actionDisabled} onClick={() => void onAction()}>
        {actionLabel}
      </button>
    </aside>
  );
});

export default ControlPanel;
