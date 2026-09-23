"""Resolve a requested game action through visual identifiers.

The rule-based detector remains the safe default.  A locally supplied,
fine-tuned ONNX model can take precedence for the two different confirmation
dialogs, but no backend is allowed to return a screen-coordinate fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .vision import DetectedButton, VisionDetector
from .vision_backends import ActionCandidate, ActionClass, YoloOnnxBackend


class ActionIdentifier:
    """Combines model proposals with conservative visual verification."""

    def __init__(self, config: dict, vision: VisionDetector) -> None:
        self.vision = vision
        ai_config = config.get("VISUAL_AI", {})
        self.backend: Optional[YoloOnnxBackend] = None
        self.backend_error: Optional[str] = None

        if not ai_config.get("ENABLED", False):
            return
        if ai_config.get("BACKEND") != "yolo_onnx":
            self.backend_error = "Backend VISUAL_AI não suportado; a deteção visual conservadora continua ativa."
            return
        try:
            self.backend = YoloOnnxBackend(
                model_path=Path(ai_config.get("MODEL_PATH", "")),
                labels_path=Path(ai_config.get("LABELS_PATH", "")),
                min_confidence=ai_config.get("MIN_CONFIDENCE", 0.92),
                input_size=ai_config.get("INPUT_SIZE", 640),
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            self.backend_error = str(error)

    def find(self, image: np.ndarray, action: ActionClass) -> Optional[DetectedButton]:
        """Return a verified button for the requested action, or ``None``."""
        if self.backend is not None:
            model_hit = self._best_model_hit(image, action)
            if model_hit is not None:
                return model_hit

        if action is ActionClass.REFRESH_BUTTON:
            return self.vision.find_refresh_button(image)
        if action is ActionClass.PURCHASE_CONFIRM_BUTTON:
            return self.vision.find_purchase_confirm_button(image)
        if action is ActionClass.REFRESH_CONFIRM_BUTTON:
            return self.vision.find_refresh_confirm_button(image)
        return None

    def _best_model_hit(self, image: np.ndarray, action: ActionClass) -> Optional[DetectedButton]:
        assert self.backend is not None
        hits = [candidate for candidate in self.backend.detect(image) if candidate.action is action]
        if not hits:
            return None
        best = max(hits, key=lambda candidate: candidate.confidence)
        return self._to_button(best)

    @staticmethod
    def _to_button(candidate: ActionCandidate) -> DetectedButton:
        return DetectedButton(
            center_x=candidate.center_x,
            center_y=candidate.center_y,
            x0=candidate.x0,
            y0=candidate.y0,
            x1=candidate.x1,
            y1=candidate.y1,
            area=(candidate.x1 - candidate.x0) * (candidate.y1 - candidate.y0),
            color=candidate.source,
        )
