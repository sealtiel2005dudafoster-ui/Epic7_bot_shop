"""
Screen capture utilities for Epic Seven Secret Shop bot.

Works with window-relative coordinates when possible. Falls back to
full-screen capture when the Epic Seven window cannot be located.

Color format contract:
  - pyautogui.screenshot() returns PIL Images in RGB order.
  - All public methods of this class return images in BGR order (OpenCV convention).
  - Internal helper _capture_from_window() returns RGB (converted in screenshot()).
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyautogui

# Try to import pygetwindow / pywinctl (optional, for window detection).
try:
    import pygetwindow as gw
    _HAS_PYGETWINDOW = True
except ImportError:
    _HAS_PYGETWINDOW = False


class ScreenshotCapture:
    """Captures screenshots, optionally clipped to the Epic Seven window."""

    def __init__(self, window_title_keyword: str = "Epic Seven"):
        self.window_keyword = window_title_keyword
        self._cached_window = None
        self._last_resolution: Optional[tuple[int, int]] = None

    def _find_epic_window(self) -> Optional[object]:
        """Locate the Epic Seven window by partial title match.

        Returns a pygetwindow Window or None.
        """
        if not _HAS_PYGETWINDOW:
            return None
        for title in gw.getAllTitles():
            if self.window_keyword in title:
                matches = gw.getWindowsWithTitle(title)
                if matches:
                    return matches[0]
        return None

    def _ensure_window(self):
        """Cache the window handle if not already found, or re-find if stale."""
        if self._cached_window is not None:
            # Validate the cached handle is still alive
            try:
                _ = self._cached_window.title
            except Exception:
                self._cached_window = None
        if self._cached_window is None:
            self._cached_window = self._find_epic_window()

    def invalidate_window_cache(self):
        """Force re-detection on next capture (call after game restart, etc.)."""
        self._cached_window = None

    def is_window_detected(self) -> bool:
        """Check if the Epic Seven window currently exists and is visible."""
        self._ensure_window()
        if self._cached_window:
            try:
                return self._cached_window.visible
            except Exception:
                self._cached_window = None
                return False
        return False

    def activate_window(self) -> bool:
        """Bring Epic Seven window to foreground and give it focus."""
        self._ensure_window()
        if self._cached_window:
            try:
                self._cached_window.activate()
                time.sleep(0.2)
                return True
            except Exception:
                return False
        return False

    def get_window_bbox(self) -> Optional[tuple[int, int, int, int]]:
        """Return (left, top, width, height) of the detected window, or None."""
        self._ensure_window()
        if self._cached_window:
            try:
                return (
                    self._cached_window.left,
                    self._cached_window.top,
                    self._cached_window.width,
                    self._cached_window.height,
                )
            except Exception:
                self._cached_window = None
                return None
        return None

    def get_window_offset(self) -> tuple[int, int]:
        """Return (left, top) offset of the window on the desktop, or (0, 0)."""
        bbox = self.get_window_bbox()
        if bbox:
            return bbox[0], bbox[1]
        return 0, 0

    def _capture_from_window(self) -> Optional[np.ndarray]:
        """Capture screenshot clipped to the window bounds.

        Returns an RGB numpy array, or None on failure.
        """
        self._ensure_window()
        if not self._cached_window:
            return None
        try:
            bbox = (
                self._cached_window.left,
                self._cached_window.top,
                self._cached_window.width,
                self._cached_window.height,
            )
            # pyautogui screenshot returns PIL Image in RGB order
            img = pyautogui.screenshot(region=bbox)
            return np.array(img)
        except Exception:
            self._cached_window = None
            return None

    def screenshot(self) -> np.ndarray:
        """
        Full capture: returns BGR numpy array (OpenCV format).
        Prefers window-relative; falls back to full screen.
        """
        img = self._capture_from_window()
        if img is None:
            img = pyautogui.screenshot()
            img = np.array(img)
        # Convert RGB (pyautogui) -> BGR (OpenCV)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Update cached resolution from this capture
        h, w = bgr.shape[:2]
        self._last_resolution = (w, h)
        return bgr

    def save_screenshot(self, path: str | Path) -> np.ndarray:
        """Capture and save to disk. Returns the BGR image array."""
        img = self.screenshot()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return img

    def get_resolution(self) -> tuple[int, int]:
        """Return (width, height) from the last screenshot, or capture one if needed."""
        if self._last_resolution is not None:
            return self._last_resolution
        img = self.screenshot()
        h, w = img.shape[:2]
        return w, h

    def check_resolution_changed(self, tolerance: int = 10) -> bool:
        """Return True if current resolution differs from the last capture by more than `tolerance` pixels."""
        prev = self._last_resolution
        # Take a fresh screenshot (updates _last_resolution)
        self.screenshot()
        w, h = self._last_resolution
        if prev is None:
            return False
        last_w, last_h = prev
        return abs(w - last_w) > tolerance or abs(h - last_h) > tolerance

    def wait_for_stable_frame(self, wait_ms: int = 300) -> np.ndarray:
        """Small delay then capture — helps avoid motion-blurred frames during animation."""
        time.sleep(wait_ms / 1000.0)
        return self.screenshot()

