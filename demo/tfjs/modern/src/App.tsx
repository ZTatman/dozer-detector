import * as React from "react";
import { Suspense } from "react";
import ModelApp from "./ModelApp";

function LoadingFallback(): JSX.Element {
  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">TensorFlow.js model</p>
          <h1>Dozer Detector</h1>
          <p className="subtitle">Real-time object detection with TensorFlow.js</p>
        </div>
        <div className="status" id="status">Loading model...</div>
      </header>

      <section className="workspace">
        <div className="stage">
          <figure className="stage-panel">
            <figcaption>Preprocessed</figcaption>
            <div className="canvas-skeleton" />
          </figure>
          <figure className="stage-panel">
            <figcaption>Detections</figcaption>
            <div className="canvas-skeleton" />
          </figure>
        </div>

        <aside className="panel skeleton-panel">
          <div className="skeleton-block" style={{ width: "40%", height: 14 }} />
          <div className="skeleton-block" style={{ width: "100%", height: 40 }} />
          <div className="skeleton-block" style={{ width: "30%", height: 14, marginTop: 8 }} />
          <div className="skeleton-block" style={{ width: "100%", height: 14 }} />
          <div className="skeleton-block" style={{ width: "100%", height: 14 }} />
          <div className="skeleton-block" style={{ width: "40%", height: 14 }} />
          <div className="skeleton-block" style={{ width: "50%", height: 12 }} />
          <div className="skeleton-block" style={{ width: "100%", height: 14 }} />
          <div className="skeleton-metrics">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="skeleton-metrics-row">
                <div className="skeleton-block" style={{ width: "40%", height: 14 }} />
                <div className="skeleton-block" style={{ width: "20%", height: 14 }} />
              </div>
            ))}
          </div>
          <div className="skeleton-block skeleton-button" />
        </aside>
      </section>
    </main>
  );
}

export default function App(): JSX.Element {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <ModelApp />
    </Suspense>
  );
}
