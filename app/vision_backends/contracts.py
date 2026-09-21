"""Contracts shared by local visual-model adapters.

Adapters return window-relative boxes.  They never control the mouse; the
buyer remains responsible for an action only after it has a verified box.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence

import numpy as np


class ActionClass(StrEnum):
    """Classes used in the Epic Seven action-detection training dataset."""

    REFRESH_BUTTON = "refresh_button"
    BUY_BUTTON = "buy_button"
    PURCHASE_CONFIRM_BUTTON = "purchase_confirm_button"
    REFRESH_CONFIRM_BUTTON = "refresh_confirm_button"
    PURCHASE_MODAL = "purchase_modal"
    REFRESH_MODAL = "refresh_modal"


@dataclass(frozen=True)
class ActionCandidate:
    """A model prediction expressed in screenshot coordinates."""

    action: ActionClass
    confidence: float
    x0: int
    y0: int
    x1: int
    y1: int
    source: str

    @property
    def center_x(self) -> int:
        return (self.x0 + self.x1) // 2

    @property
    def center_y(self) -> int:
        return (self.y0 + self.y1) // 2


class VisionBackend(Protocol):
    """Protocol implemented by a local model such as a fine-tuned YOLO ONNX."""

    def detect(self, image: np.ndarray) -> Sequence[ActionCandidate]:
        """Return labelled action candidates for one BGR screenshot."""

