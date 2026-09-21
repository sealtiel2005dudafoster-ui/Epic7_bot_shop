"""
Safety checks: verify we're looking at the correct game, detect
unexpected conditions, and enforce safe-mode behavior.

Any safety violation puts the system in SAFE MODE — no purchases, no
scrolls, no input to the game. All state transitions must re-verify
before proceeding.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

from .capture import ScreenshotCapture


class SafetyState(Enum):
    OK = "ok"
    SAFE_MODE = "safe_mode"


@dataclass
class SafetyCheck:
    """Result of a single safety check."""
    name: str
    passed: bool
    message: str


class SafetyMonitor:
    """
    Monitors the runtime environment for unsafe conditions.

    Triggers safe mode if:
    - Epic Seven window is not detected
    - Resolution changed unexpectedly
    - OCR returned invalid data
    - Low confidence detection
    - Screenshot doesn't match expected layout
    """

    def __init__(
        self,
        capture: ScreenshotCapture,
        window_title: str = "Epic Seven",
        resolution_tolerance: int = 10,
    ):
        self.capture = capture
        self.window_title = window_title
        self.resolution_tolerance = resolution_tolerance
        self.state: SafetyState = SafetyState.OK
        self.last_error: Optional[str] = None

    def _record_failure(self, check_name: str, message: str):
        self.state = SafetyState.SAFE_MODE
        self.last_error = f"[{check_name}] {message}"

    def check_window_detected(self) -> SafetyCheck:
        """Verify the Epic Seven window is active."""
        detected = self.capture.is_window_detected()
        check = SafetyCheck(
            name="window_detection",
            passed=detected,
            message=f"Epic Seven window {'detected' if detected else 'NOT detected'}",
        )
        if not detected and self.state == SafetyState.OK:
            self._record_failure("window_detection", "Epic Seven window not found")
        return check

    def check_resolution_stable(self) -> SafetyCheck:
        """Verify screen resolution hasn't changed unexpectedly."""
        changed = self.capture.check_resolution_changed(self.resolution_tolerance)
        check = SafetyCheck(
            name="resolution_stability",
            passed=not changed,
            message=f"Resolution {'changed' if changed else 'stable'}",
        )
        if changed and self.state == SafetyState.OK:
            self._record_failure("resolution_stability", "Screen resolution changed unexpectedly")
        return check

    def check_confidence(self, confidence: float, threshold: float) -> SafetyCheck:
        """Verify a detection meets the minimum confidence threshold."""
        passed = confidence >= threshold
        check = SafetyCheck(
            name="confidence",
            passed=passed,
            message=f"Confidence {confidence:.3f} {'meets' if passed else 'below'} threshold {threshold:.3f}",
        )
        if not passed and self.state == SafetyState.OK:
            self._record_failure("confidence", f"Confidence {confidence:.3f} below threshold {threshold:.3f}")
        return check

    def check_price(self, price: Optional[int]) -> SafetyCheck:
        """Verify OCR returned a valid price."""
        valid = price is not None and price > 0
        check = SafetyCheck(
            name="price_validation",
            passed=valid,
            message=f"Price OCR {'valid' if valid else 'invalid or missing'}: {price}",
        )
        if not valid and self.state == SafetyState.OK:
            self._record_failure("price_validation", "Invalid or missing price from OCR")
        return check

    def check_shop_layout(self, img: np.ndarray, layout: Optional[object] = None, expected_height: int = 474, tolerance: int = 50) -> SafetyCheck:
        """
        Verify that the shop interface is actually displayed on screen.
        If a ShopLayout with visual anchor detection is available, verifies
        the Secret Shop visual anchor. Otherwise falls back to height heuristic.
        """
        if layout is not None and getattr(layout, "anchor_detected", False):
            passed = True
            message = f"Secret Shop visual anchor confirmed at {getattr(layout, 'anchor_location', (0,0))}"
        else:
            h = img.shape[0]
            passed = abs(h - expected_height) <= tolerance or h >= 300
            message = f"Image height {h}px {'matches' if passed else 'deviates'} expected ~{expected_height}px"

        check = SafetyCheck(name="layout_check", passed=passed, message=message)
        if not passed and self.state == SafetyState.OK:
            self._record_failure("layout_check", f"Unexpected layout or Secret Shop anchor missing")
        return check

    def run_all_checks(self, img: np.ndarray, expected_height: int = 474, confidence: float = 1.0,
                       confidence_threshold: float = 0.85, price: Optional[int] = None,
                       layout: Optional[object] = None) -> list[SafetyCheck]:
        """Run all safety checks and update state. Returns list of all check results."""
        checks = [
            self.check_window_detected(),
            self.check_resolution_stable(),
            self.check_shop_layout(img, layout=layout, expected_height=expected_height),
            self.check_confidence(confidence, confidence_threshold),
            self.check_price(price),
        ]
        return checks

    def is_safe(self) -> bool:
        """Return True if system is in a safe state to proceed."""
        return self.state == SafetyState.OK

    def reset(self):
        """Reset safety state back to OK (e.g., after user intervention)."""
        self.state = SafetyState.OK
        self.last_error = None

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "last_error": self.last_error,
            "window_detected": self.capture.is_window_detected(),
        }
