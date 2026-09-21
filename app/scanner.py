"""
Scanner orchestrator: coordinates capture, detection, OCR, scrolling,
and safety checks to produce a complete picture of all 6 shop slots.

This module does NOT make purchase decisions. It only observes and reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyautogui

from .capture import ScreenshotCapture
from .detector import CoinDetector, SlotDetection
from .safety import SafetyMonitor, SafetyState
from .slots import ShopLayout


@dataclass
class ScanResult:
    """Complete result of scanning the Secret Shop."""
    slots: list[SlotDetection] = field(default_factory=list)
    scroll_success: bool = False
    scroll_attempts: int = 0
    errors: list[str] = field(default_factory=list)
    safety_state: str = "ok"
    resolution: tuple[int, int] = (0, 0)
    raw_screenshots: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "slots": [s.to_dict() for s in self.slots],
            "scroll_success": self.scroll_success,
            "scroll_attempts": self.scroll_attempts,
            "errors": self.errors,
            "safety_state": self.safety_state,
            "resolution": self.resolution,
        }


class ShopScanner:
    """
    Orchestrates the full shop scan:
    1. Capture screenshot
    2. Detect 5 visible slots
    3. Scroll down
    4. Verify 6th slot appeared
    5. Detect 6th slot
    6. Return combined results
    """

    def __init__(
        self,
        capture: ScreenshotCapture,
        detector: CoinDetector,
        safety: SafetyMonitor,
        layout: Optional[ShopLayout] = None,
        scroll_amount: int = 300,
        scroll_verify_wait_ms: int = 500,
        scroll_attempts_max: int = 5,
    ):
        self.capture = capture
        self.detector = detector
        self.safety = safety
        self.scroll_amount = scroll_amount
        self.scroll_verify_wait_ms = scroll_verify_wait_ms
        self.scroll_attempts_max = scroll_attempts_max

        if layout is None:
            self.layout = ShopLayout()
        else:
            self.layout = layout

    def scan(self, save_debug: bool = False) -> ScanResult:
        """
        Perform a full shop scan including the 6th slot.

        Returns a ScanResult with all slot detections.
        Never clicks or purchases — read-only observation.
        """
        result = ScanResult()

        # --- Step 1: Initial safety checks ---
        if not self.safety.is_safe():
            result.safety_state = self.safety.state.value
            result.errors.append(f"Safety monitor in {self.safety.state.value} state")
            return result

        # --- Step 2: Capture initial screenshot ---
        img = self.capture.screenshot()
        self.layout.adjust_for_resolution(img)
        result.resolution = (img.shape[1], img.shape[0])

        # Verify resolution stability
        res_check = self.safety.check_resolution_stable()
        if not res_check.passed:
            result.safety_state = self.safety.state.value
            result.errors.append(res_check.message)

        # Window detection check
        win_check = self.safety.check_window_detected()
        if not win_check.passed:
            result.errors.append(win_check.message)

        if not self.safety.is_safe():
            return result

        if save_debug:
            result.raw_screenshots["initial"] = img.copy()

        # --- Step 3: Detect 5 visible slots ---
        slot_count = 0
        for i in range(5):
            slot_region = self.layout.slots[i]
            detection = self.detector.detect_slot(img, slot_region, i + 1)
            result.slots.append(detection)
            slot_count += 1

        # --- Step 4: Scroll and detect 6th slot ---
        result = self._scan_slot_6(result, img, save_debug=save_debug)

        return result

    def scan_dry_run(self, save_debug: bool = False) -> ScanResult:
        """DRY_RUN mode: scan without any side effects (no clicks, no scroll).

        Only detects the 5 visible slots. Use scan_from_image_pair for
        offline testing that includes slot 6.
        """
        result = ScanResult()

        if not self.safety.is_safe():
            result.safety_state = self.safety.state.value
            result.errors.append(f"Safety monitor in {self.safety.state.value} state")
            return result

        img = self.capture.screenshot()
        self.layout.adjust_for_resolution(img)
        result.resolution = (img.shape[1], img.shape[0])

        if save_debug:
            result.raw_screenshots["initial"] = img.copy()

        for i in range(5):
            slot_region = self.layout.slots[i]
            detection = self.detector.detect_slot(img, slot_region, i + 1)
            result.slots.append(detection)

        # Slot 6 not scanned in dry run (would require scrolling)
        result.scroll_success = False
        return result

    def _scan_slot_6(self, result: ScanResult, initial_img: np.ndarray, save_debug: bool = False) -> ScanResult:
        """
        Scroll the shop to reveal slot 6, verify it appeared,
        then detect its contents.

        Scroll strategy:
        - Scroll down by `scroll_amount` pixels
        - Wait `scroll_verify_wait_ms` for animation
        - Capture a new screenshot
        - Check if slot 6 region has changed
        - If not changed, try a correction scroll
        - Limit attempts to `scroll_attempts_max`
        """
        slot_6_region = self.layout.slots[5]

        # Save the initial slot 6 region to compare later
        initial_slot_6_roi = slot_6_region.get_roi(initial_img)

        for attempt in range(1, self.scroll_attempts_max + 1):
            result.scroll_attempts = attempt

            # Perform scroll (down = negative y in pyautogui scroll)
            # Scroll direction: negative = down, positive = up
            pyautogui.scroll(-self.scroll_amount)
            time.sleep(self.scroll_verify_wait_ms / 1000.0)

            # Capture new screenshot
            new_img = self.capture.screenshot()
            if save_debug:
                result.raw_screenshots[f"after_scroll_{attempt}"] = new_img.copy()

            # Check if slot 6 region changed
            new_slot_6_roi = slot_6_region.get_roi(new_img)

            # Use SSIM-like comparison: check if the two ROIs are different enough
            changed = self._regions_differ(initial_slot_6_roi, new_slot_6_roi)

            if changed:
                # Slot 6 area changed — likely scrolled into view
                # Do a small correction scroll if needed
                detection = self.detector.detect_slot(new_img, slot_6_region, 6)

                # If slot 6 is now showing a coin or even something different from the initial empty state,
                # consider the scroll successful
                if detection.tipo != "desconhecido" or detection.confianca > 0:
                    result.slots.append(detection)
                    result.scroll_success = True
                    # Replace the last "placeholder" if we added one earlier, or just append
                    # We append because slots 1-5 are already there
                    return result

            # If we did scroll but slot 6 still looks the same, try a correction
            if attempt < self.scroll_attempts_max:
                # Small correction scroll
                pyautogui.scroll(-int(self.scroll_amount * 0.3))
                time.sleep(self.scroll_verify_wait_ms / 1000.0)
                new_img = self.capture.screenshot()
                detection = self.detector.detect_slot(new_img, slot_6_region, 6)
                if detection.tipo != "desconhecido" or detection.confianca > 0:
                    result.slots.append(detection)
                    result.scroll_success = True
                    return result

        # Max attempts reached without detecting slot 6
        result.errors.append(f"Slot 6 not detected after {self.scroll_attempts_max} scroll attempts")
        # Add a placeholder detection for slot 6
        result.slots.append(SlotDetection(slot=6, tipo="desconhecido", confianca=0.0))
        result.scroll_success = False
        return result

    def _regions_differ(self, roi_a: np.ndarray, roi_b: np.ndarray, threshold: float = 0.1) -> bool:
        """
        Return True if two regions are meaningfully different.
        Uses pixel-level comparison with a similarity threshold.
        """
        if roi_a.shape != roi_b.shape:
            return True

        # Convert to grayscale for comparison
        if len(roi_a.shape) == 3:
            roi_a = cv2.cvtColor(roi_a, cv2.COLOR_BGR2GRAY)
            roi_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2GRAY)

        # Compute mean absolute difference
        diff = cv2.absdiff(roi_a, roi_b)
        mean_diff = np.mean(diff)

        # If the mean pixel difference exceeds threshold (out of 255), regions changed
        return mean_diff > (threshold * 255)

    def scan_from_image(self, img: np.ndarray) -> ScanResult:
        """
        Scan from a pre-captured image (for testing without a live game).
        Detects all 6 slots from a single image — assumes slot 6 is already
        visible (e.g. a stitched or scrolled screenshot).
        """
        result = ScanResult()
        self.layout.adjust_for_resolution(img)
        result.resolution = (img.shape[1], img.shape[0])

        for i in range(6):
            slot_region = self.layout.slots[i]
            detection = self.detector.detect_slot(img, slot_region, i + 1)
            result.slots.append(detection)

        return result

    def scan_from_image_pair(
        self,
        img_before_scroll: np.ndarray,
        img_after_scroll: np.ndarray,
    ) -> ScanResult:
        """
        Scan from a pair of screenshots: one before and one after scrolling.
        Fully testable without a live game capture.

        Returns detections for all 6 slots.
        """
        result = ScanResult()

        # Adjust layout to match image resolution
        self.layout.adjust_for_resolution(img_before_scroll)
        result.resolution = (img_before_scroll.shape[1], img_before_scroll.shape[0])

        # Detect first 5 slots from pre-scroll image
        for i in range(5):
            slot_region = self.layout.slots[i]
            detection = self.detector.detect_slot(img_before_scroll, slot_region, i + 1)
            result.slots.append(detection)

        # Detect slot 6 from post-scroll image
        slot_6_region = self.layout.slots[5]
        slot_6_detection = self.detector.detect_slot(img_after_scroll, slot_6_region, 6)
        result.slots.append(slot_6_detection)
        result.scroll_success = True  # Assuming the test image already has slot 6 scrolled into view

        return result
