export const MODEL_URL = "/zach_model/model.json";
export const CANVAS_WIDTH = 640;
export const CANVAS_HEIGHT = 480;
export const CAMERA_SOURCE = "camera";
export const CAMERA_TIMEOUT_MS = 15000;
export const OUTPUT_NODES = [
  "detection_scores",
  "detection_boxes",
  "detection_classes",
  "num_detections",
];

export const EXAMPLES = [
  { label: "Original desk photo", value: "/static/img_2444.jpg" },
  { label: "Square desk photo", value: "/static/dozer_zach.jpg" },
  { label: "Demo crop", value: "/static/dozer2.jpg" },
  { label: "Camera", value: CAMERA_SOURCE },
];

export const EMPTY_METRICS = {
  preprocess: "-",
  model: "-",
  postprocess: "-",
  total: "-",
  detection: "-",
};

export type Metrics = typeof EMPTY_METRICS;
