"""ONNX Runtime adapter for a *custom-trained* YOLO action detector.

The stock COCO YOLO weights do not know Epic Seven UI controls.  This adapter
therefore accepts only the six classes declared in ``labels.txt`` and fails
closed if the local model or labels are not present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .contracts import ActionCandidate, ActionClass


class YoloOnnxBackend:
    """Run a fine-tuned YOLOv8/YOLO11 ONNX detector locally."""

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        min_confidence: float = 0.92,
        input_size: int = 640,
    ) -> None:
        model = Path(model_path)
        labels = Path(labels_path)
        if not model.is_file():
            raise FileNotFoundError(f"Modelo ONNX não encontrado: {model}")
        if not labels.is_file():
            raise FileNotFoundError(f"Ficheiro de classes não encontrado: {labels}")

        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - depends on optional local package
            raise RuntimeError(
                "O backend YOLO ONNX requer onnxruntime. Instale-o apenas depois de copiar o modelo."
            ) from error

        self.labels = [line.strip() for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]
        unknown = set(self.labels) - {action.value for action in ActionClass}
        if unknown:
            raise ValueError(f"Classes YOLO não suportadas: {sorted(unknown)}")
        self.min_confidence = float(min_confidence)
        self.input_size = int(input_size)
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: np.ndarray) -> Sequence[ActionCandidate]:
        """Run one frame through the detector and map boxes to the original frame."""
        original_h, original_w = image.shape[:2]
        prepared, scale, pad_x, pad_y = self._letterbox(image)
        tensor = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor[np.newaxis, ...], dtype=np.float32) / 255.0
        raw = self.session.run(None, {self.input_name: tensor})[0]
        predictions = self._normalise_predictions(raw)

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for prediction in predictions:
            if len(prediction) < 5:
                continue
            class_id, confidence, box = self._decode_prediction(prediction)
            if confidence < self.min_confidence or class_id >= len(self.labels):
                continue
            x, y, width, height = box
            if width <= 1 or height <= 1:
                continue
            boxes.append([x, y, width, height])
            scores.append(confidence)
            class_ids.append(class_id)

        kept = cv2.dnn.NMSBoxes(boxes, scores, self.min_confidence, 0.45) if boxes else []
        candidates: list[ActionCandidate] = []
        for index in np.asarray(kept).reshape(-1):
            x, y, width, height = boxes[int(index)]
            x0 = max(0, int(round((x - pad_x) / scale)))
            y0 = max(0, int(round((y - pad_y) / scale)))
            x1 = min(original_w, int(round((x + width - pad_x) / scale)))
            y1 = min(original_h, int(round((y + height - pad_y) / scale)))
            if x1 <= x0 or y1 <= y0:
                continue
            candidates.append(ActionCandidate(
                action=ActionClass(self.labels[class_ids[int(index)]]),
                confidence=float(scores[int(index)]),
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                source="yolo_onnx",
            ))
        return candidates

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.input_size - resized.shape[1]) // 2
        pad_y = (self.input_size - resized.shape[0]) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
        return canvas, scale, pad_x, pad_y

    @staticmethod
    def _normalise_predictions(raw: np.ndarray) -> np.ndarray:
        predictions = np.asarray(raw)
        if predictions.ndim == 3:
            predictions = predictions[0]
        # Ultralytics exports normally use (features, anchors); transpose to rows.
        if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        return predictions

    @staticmethod
    def _decode_prediction(prediction: np.ndarray) -> tuple[int, float, tuple[int, int, int, int]]:
        # Standard YOLOv8/11 export: cx, cy, w, h, class0 ... classN.
        if len(prediction) > 6:
            class_id = int(np.argmax(prediction[4:]))
            confidence = float(prediction[4 + class_id])
            cx, cy, width, height = map(float, prediction[:4])
            return class_id, confidence, (int(cx - width / 2), int(cy - height / 2), int(width), int(height))

        # Some exporters yield x1, y1, x2, y2, confidence, class_id.
        x0, y0, x1, y1, confidence, class_id = prediction[:6]
        return int(class_id), float(confidence), (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
