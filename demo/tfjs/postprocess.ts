import type { Tensor } from "@tensorflow/tfjs";

const MAX_PREDICTIONS = 10;


interface DetectedObject {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

export default async function getDetections(
  result: Tensor[],
  width: number,
  height: number,
): Promise<DetectedObject[]> {
  const [scoresTensor, boxesTensor, classesTensor, countTensor] = result;

  if (!scoresTensor || !boxesTensor) {
    throw new Error("Model did not return detection scores and boxes");
  }

  const [scores, boxes, classes, counts] = await Promise.all([
    scoresTensor.data(),
    boxesTensor.data(),
    classesTensor?.data(),
    countTensor?.data(),
  ]);

  const availableCount = counts?.[0] ?? scoresTensor.shape.at(-1) ?? scores.length;
  const count = Math.min(Math.round(Number(availableCount)), MAX_PREDICTIONS);
  const objects: DetectedObject[] = [];

  for (let i = 0; i < count; i += 1) {
    const score = Number(scores[i]);

    const boxOffset = i * 4;
    const minY = Number(boxes[boxOffset]) * height;
    const minX = Number(boxes[boxOffset + 1]) * width;
    const maxY = Number(boxes[boxOffset + 2]) * height;
    const maxX = Number(boxes[boxOffset + 3]) * width;
    const classId = classes ? Number(classes[i]) : 1;

    objects.push({
      bbox: [minX, minY, maxX - minX, maxY - minY],
      class: classId === 1 ? "dozer" : `class ${classId}`,
      score,
    });
  }

  return objects;
}
