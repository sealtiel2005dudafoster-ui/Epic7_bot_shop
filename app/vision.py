"""
Visual Button and Popup Detector for Epic Seven Secret Shop Bot.

Uses HSV color analysis to locate UI elements in real-time screenshots:
  - Green "Comprar" buttons (active buy buttons)
  - Green "Confirmar" / "OK" buttons in popup modals
  - Popup detection (dark overlay = popup is open)
  - Post-action verification (did the click actually work?)

This replaces hardcoded coordinate percentages with visual detection,
solving the root cause of missed clicks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectedButton:
    """A visually detected button with its screen coordinates."""
    center_x: int
    center_y: int
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    color: str  # "green", "red", "blue"

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


class VisionDetector:
    """
    Real-time visual detector for Epic Seven UI elements.

    Uses HSV color space to find buttons regardless of exact position,
    making the bot robust to resolution changes and window position shifts.
    """

    # Epic Seven uses a dark green gradient, not a flat bright green.  The
    # previous lower V=80 threshold saw only a few highlights of each button,
    # so no complete contour could be formed.  These limits cover the full
    # gradient while the geometry checks below reject unrelated green pixels.
    GREEN_HSV_LOWER = np.array([35, 70, 25])
    GREEN_HSV_UPPER = np.array([90, 255, 230])

    # "Cancelar" is brown/orange.  A confirmation is accepted only when a
    # green or blue action button is paired with this sibling at the same
    # height.  That relationship identifies a modal without relying on its
    # screen position.  The shop-refresh modal's "Confirmar" is blue (hue
    # ~106); the purchase modal's is green.
    BROWN_HSV_LOWER = np.array([5, 60, 25])
    BROWN_HSV_UPPER = np.array([35, 255, 230])

    # Refresh confirm button ("Confirmar") on the shop-refresh modal.
    BLUE_HSV_LOWER = np.array([100, 40, 25])
    BLUE_HSV_UPPER = np.array([130, 255, 230])

    # The refresh button contains the Skystone icon.  It is an additional
    # visual signature used if the refresh template is temporarily unavailable.
    SKYSTONE_HSV_LOWER = np.array([85, 80, 70])
    SKYSTONE_HSV_UPPER = np.array([140, 255, 255])

    # Minimum button dimensions (in pixels) to filter out noise
    MIN_BUTTON_WIDTH = 30
    MIN_BUTTON_HEIGHT = 15
    MIN_BUTTON_AREA = 600

    # Maximum button dimensions to filter out background regions
    MAX_BUTTON_WIDTH = 300
    MAX_BUTTON_HEIGHT = 100

    def find_green_buttons(
        self,
        img: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[DetectedButton]:
        """Find all green buttons in the image (or a sub-region)."""
        return self._find_action_buttons(img, "green", self.GREEN_HSV_LOWER, self.GREEN_HSV_UPPER, region)

    def find_blue_buttons(
        self,
        img: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[DetectedButton]:
        """Find all blue buttons in the image (or a sub-region)."""
        return self._find_action_buttons(img, "blue", self.BLUE_HSV_LOWER, self.BLUE_HSV_UPPER, region)

    def _find_action_buttons(
        self,
        img: np.ndarray,
        color: str,
        hsv_lower: np.ndarray,
        hsv_upper: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[DetectedButton]:
        """
        Find all buttons of a given HSV color in the image (or a sub-region).

        Args:
            img: BGR image (OpenCV format)
            color: label stored on each DetectedButton
            region: Optional (x0, y0, x1, y1) to restrict search area

        Returns:
            List of DetectedButton sorted by y-coordinate (top to bottom)
        """
        if region is not None:
            rx0, ry0, rx1, ry1 = region
            h, w = img.shape[:2]
            rx0, rx1 = max(0, rx0), min(w, rx1)
            ry0, ry1 = max(0, ry0), min(h, ry1)
            if rx1 <= rx0 or ry1 <= ry0:
                return []
            crop = img[ry0:ry1, rx0:rx1]
            offset_x, offset_y = rx0, ry0
        else:
            crop = img
            offset_x, offset_y = 0, 0

        if crop is None or crop.size == 0:
            return []
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        # Morphological operations join the dark gradient, bright rim, and
        # color-matching text highlights that belong to one rendered button.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        buttons: List[DetectedButton] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Filter by size
            if w < self.MIN_BUTTON_WIDTH or h < self.MIN_BUTTON_HEIGHT:
                continue
            if w > self.MAX_BUTTON_WIDTH or h > self.MAX_BUTTON_HEIGHT:
                continue
            if area < self.MIN_BUTTON_AREA:
                continue

            # Aspect ratio check: buttons are wider than tall
            aspect = w / max(h, 1)
            if aspect < 1.2 or aspect > 8.0:
                continue

            cx = x + w // 2 + offset_x
            cy = y + h // 2 + offset_y

            buttons.append(DetectedButton(
                center_x=cx,
                center_y=cy,
                x0=x + offset_x,
                y0=y + offset_y,
                x1=x + w + offset_x,
                y1=y + h + offset_y,
                area=int(area),
                color=color,
            ))

        # Sort by vertical position (top to bottom)
        buttons.sort(key=lambda b: b.center_y)
        return buttons

    def find_brown_buttons(
        self,
        img: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[DetectedButton]:
        """Find brown modal sibling buttons such as ``Cancelar``."""
        if region is not None:
            rx0, ry0, rx1, ry1 = region
            h, w = img.shape[:2]
            rx0, rx1 = max(0, rx0), min(w, rx1)
            ry0, ry1 = max(0, ry0), min(h, ry1)
            if rx1 <= rx0 or ry1 <= ry0:
                return []
            crop = img[ry0:ry1, rx0:rx1]
            offset_x, offset_y = rx0, ry0
        else:
            crop = img
            offset_x, offset_y = 0, 0

        if crop is None or crop.size == 0:
            return []
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.BROWN_HSV_LOWER, self.BROWN_HSV_UPPER)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
            iterations=2,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        buttons: List[DetectedButton] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if not (
                self.MIN_BUTTON_WIDTH <= w <= self.MAX_BUTTON_WIDTH
                and self.MIN_BUTTON_HEIGHT <= h <= self.MAX_BUTTON_HEIGHT
                and area >= self.MIN_BUTTON_AREA
            ):
                continue
            aspect = w / max(h, 1)
            if not 1.2 <= aspect <= 8.0:
                continue
            buttons.append(DetectedButton(
                center_x=x + w // 2 + offset_x,
                center_y=y + h // 2 + offset_y,
                x0=x + offset_x,
                y0=y + offset_y,
                x1=x + w + offset_x,
                y1=y + h + offset_y,
                area=int(area),
                color="brown",
            ))
        return buttons

    def find_buy_button_for_slot(
        self,
        img: np.ndarray,
        slot_y0: int,
        slot_y1: int,
        slot_x0: int,
        slot_x1: int,
    ) -> Optional[DetectedButton]:
        """
        Find the green "Comprar" button specifically within a slot's row region.

        Searches the right portion of the slot where the buy button lives.
        """
        # Search the right 40% of the slot (where buy button is)
        search_x0 = slot_x0 + int((slot_x1 - slot_x0) * 0.55)
        search_region = (search_x0, slot_y0, slot_x1, slot_y1)

        buttons = self.find_green_buttons(img, region=search_region)

        if not buttons:
            return None

        # Return the largest button found (most likely the actual buy button)
        return max(buttons, key=lambda b: b.area)

    def is_popup_visible(self, img: np.ndarray) -> bool:
        """
        Detect if a popup/modal is currently displayed.

        Epic Seven popups have a dark semi-transparent overlay behind them.
        We check the corners of the image — if they are significantly darker
        than normal, a popup overlay is present.
        """
        h, w = img.shape[:2]

        # Sample small patches from the 4 corners (behind the popup overlay)
        corner_size = max(20, min(w, h) // 20)
        corners = [
            img[5:corner_size, 5:corner_size],                    # top-left
            img[5:corner_size, w-corner_size:w-5],                # top-right
            img[h-corner_size:h-5, 5:corner_size],                # bottom-left
            img[h-corner_size:h-5, w-corner_size:w-5],            # bottom-right
        ]

        dark_corners = 0
        for corner in corners:
            if corner.size == 0:
                continue
            mean_brightness = np.mean(cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY))
            if mean_brightness < 60:  # Dark overlay threshold
                dark_corners += 1

        return dark_corners >= 3  # At least 3 of 4 corners are dark

    def find_refresh_confirm_button(self, img: np.ndarray) -> Optional[DetectedButton]:
        """Locate the blue 'Confirmar' button in the shop-refresh modal.

        Uses the real template 'templates/popup_refresh_confirm_button.png' first.
        If unavailable or confidence is low, falls back to finding the brown 'Cancelar'
        sibling and measuring the blue button width to its right.
        """
        template_path = Path("templates/popup_refresh_confirm_button.png")
        if template_path.exists():
            tpl = cv2.imread(str(template_path))
            if tpl is not None:
                matched = self._match_template(img, tpl, min_confidence=0.75)
                if matched is not None:
                    return matched

        # Color/sibling fallback:
        brown_buttons = self.find_brown_buttons(img)
        blue_buttons = self.find_blue_buttons(img)
        for confirm in blue_buttons:
            for brown in brown_buttons:
                vertical_delta = abs(confirm.center_y - brown.center_y)
                max_height = max(confirm.height, brown.height)
                horizontal_gap = confirm.x0 - brown.x1
                comparable_height = 0.55 <= brown.height / max(confirm.height, 1) <= 1.6
                if (
                    brown.x1 <= confirm.x0
                    and vertical_delta <= max_height * 0.50
                    and horizontal_gap <= max(confirm.width, brown.width) * 1.5
                    and comparable_height
                ):
                    return confirm

        # Band fallback right of Cancelar with proper button width limit:
        if brown_buttons:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, self.BLUE_HSV_LOWER, self.BLUE_HSV_UPPER)
            img_h, img_w = img.shape[:2]
            for brown in brown_buttons:
                rx0 = brown.x1
                ry0 = max(0, brown.y0 - brown.height // 2)
                ry1 = min(img_h, brown.y1 + brown.height // 2)
                if rx0 >= img_w or ry1 <= ry0:
                    continue
                # Expected button width is approximately equal to Cancelar width (~110px)
                expected_w = int(brown.width * 1.3)
                rx1 = min(img_w, rx0 + expected_w)
                band = blue_mask[ry0:ry1, rx0:rx1]
                density = cv2.countNonZero(band) / max(band.size, 1)
                if density >= 0.20:
                    col_density = np.count_nonzero(band, axis=0) / max(band.shape[0], 1)
                    blue_cols = np.where(col_density > 0.20)[0]
                    if len(blue_cols) >= 15:
                        bx0 = rx0 + int(blue_cols[0])
                        bx1 = rx0 + int(blue_cols[-1]) + 1
                        by0 = brown.y0
                        by1 = brown.y1
                        return DetectedButton(
                            center_x=(bx0 + bx1) // 2,
                            center_y=(by0 + by1) // 2,
                            x0=bx0, y0=by0, x1=bx1, y1=by1,
                            area=(bx1 - bx0) * (by1 - by0),
                            color="blue",
                        )
        return None

    def find_purchase_confirm_button(self, img: np.ndarray) -> Optional[DetectedButton]:
        """Locate the green 'Comprar' button in the item-purchase modal.

        Uses the real template 'templates/popup_buy_button.png' first.
        If unavailable, falls back to Cancelar sibling pair detection.
        """
        template_path = Path("templates/popup_buy_button.png")
        if template_path.exists():
            tpl = cv2.imread(str(template_path))
            if tpl is not None:
                matched = self._match_template(img, tpl, min_confidence=0.75)
                if matched is not None:
                    return matched

        green_buttons = self.find_green_buttons(img)
        brown_buttons = self.find_brown_buttons(img)
        for confirm in green_buttons:
            for brown in brown_buttons:
                vertical_delta = abs(confirm.center_y - brown.center_y)
                max_height = max(confirm.height, brown.height)
                horizontal_gap = confirm.x0 - brown.x1
                comparable_height = 0.55 <= brown.height / max(confirm.height, 1) <= 1.6
                if (
                    brown.x1 <= confirm.x0
                    and vertical_delta <= max_height * 0.50
                    and horizontal_gap <= max(confirm.width, brown.width) * 1.5
                    and comparable_height
                ):
                    return confirm
        return None

    def find_popup_confirm_button(self, img: np.ndarray) -> Optional[DetectedButton]:
        """Find a modal confirmation button (either purchase or refresh)."""
        refresh_btn = self.find_refresh_confirm_button(img)
        if refresh_btn is not None:
            return refresh_btn
        return self.find_purchase_confirm_button(img)

    def find_refresh_button(self, img: np.ndarray) -> Optional[DetectedButton]:
        """Locate ``Renovar`` through its own visual identifier.

        The full button template includes the Skystone icon and the label, so
        it remains valid when the shop has scrolled or moved.  A colour/icon
        signature is retained only as a visual fallback if that template is
        unavailable; no layout or percentage coordinate is used.
        """
        from pathlib import Path

        template_path = Path("templates/anchor_refresh.png")
        if template_path.exists():
            template = cv2.imread(str(template_path))
            if template is not None:
                matched = self._match_template(img, template, min_confidence=0.80)
                if matched is not None:
                    return matched

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        candidates: List[Tuple[int, DetectedButton]] = []
        for button in self.find_green_buttons(img):
            left_half = hsv[button.y0:button.y1, button.x0:button.center_x]
            if left_half.size == 0:
                continue
            skystone = cv2.inRange(left_half, self.SKYSTONE_HSV_LOWER, self.SKYSTONE_HSV_UPPER)
            skystone_pixels = int(np.count_nonzero(skystone))
            if skystone_pixels >= 20:
                candidates.append((skystone_pixels, button))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def _match_template(
        img: np.ndarray,
        template: np.ndarray,
        min_confidence: float,
    ) -> Optional[DetectedButton]:
        """Return a multi-scale visual match or ``None``; never infer a point."""
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        image_h, image_w = image_gray.shape[:2]
        template_h, template_w = template_gray.shape[:2]
        best: Optional[Tuple[float, Tuple[int, int], int, int]] = None

        for scale in (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25):
            width, height = int(template_w * scale), int(template_h * scale)
            if width < 10 or height < 10 or width >= image_w or height >= image_h:
                continue
            resized = cv2.resize(
                template_gray,
                (width, height),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )
            result = cv2.matchTemplate(image_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, location = cv2.minMaxLoc(result)
            if best is None or confidence > best[0]:
                best = (float(confidence), location, width, height)

        if best is None or best[0] < min_confidence:
            return None
        confidence, (x, y), width, height = best
        return DetectedButton(
            center_x=x + width // 2,
            center_y=y + height // 2,
            x0=x,
            y0=y,
            x1=x + width,
            y1=y + height,
            area=width * height,
            color="template",
        )

    def verify_purchase_popup_appeared(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray,
    ) -> bool:
        """
        Verify that clicking a buy button actually opened the purchase popup.

        Compares before/after screenshots to detect:
        1. Dark overlay appeared (popup is showing)
        2. OR significant brightness change in center area
        """
        # Method 1: A verified Cancel + Confirm pair appeared.  This is more
        # reliable than corner brightness because the title bar is outside the
        # dark overlay in some emulator/window combinations.
        if self.find_popup_confirm_button(img_after) is not None:
            return True

        # Method 2: Check if popup overlay appeared
        popup_before = self.is_popup_visible(img_before)
        popup_after = self.is_popup_visible(img_after)

        if not popup_before and popup_after:
            return True  # Popup appeared!

        # Method 2: Check center brightness change (popup content is lighter than game)
        h, w = img_before.shape[:2]
        center_before = img_before[int(h*0.3):int(h*0.7), int(w*0.25):int(w*0.75)]
        center_after = img_after[int(h*0.3):int(h*0.7), int(w*0.25):int(w*0.75)]

        mean_before = np.mean(cv2.cvtColor(center_before, cv2.COLOR_BGR2GRAY))
        mean_after = np.mean(cv2.cvtColor(center_after, cv2.COLOR_BGR2GRAY))

        # Popup content is typically brighter than the darkened background
        brightness_change = abs(mean_after - mean_before)
        return brightness_change > 25

    def verify_popup_dismissed(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray,
    ) -> bool:
        """
        Verify that a popup was dismissed (confirmation button was clicked successfully).

        Checks that the dark overlay is no longer present.
        """
        popup_before = self.is_popup_visible(img_before)
        popup_after = self.is_popup_visible(img_after)

        # A verified action pair disappeared → success.
        if self.find_popup_confirm_button(img_before) is not None and self.find_popup_confirm_button(img_after) is None:
            return True

        # Popup was visible before and is gone now → success
        if popup_before and not popup_after:
            return True

        # Also check if the overall scene changed significantly
        gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_before, gray_after)
        change_ratio = np.sum(diff > 30) / diff.size

        # More than 10% of pixels changed → something happened
        return change_ratio > 0.10

    def verify_item_purchased(
        self,
        img: np.ndarray,
        slot_y0: int,
        slot_y1: int,
        slot_x0: int,
        slot_x1: int,
    ) -> bool:
        """
        Verify that an item was actually purchased by checking if the
        buy button in this slot is now grayed out / disabled (0/1).

        After a successful purchase, the green button turns dark/gray.
        """
        btn = self.find_buy_button_for_slot(img, slot_y0, slot_y1, slot_x0, slot_x1)
        # If no green button is found in the slot → it was purchased (button is gray now)
        return btn is None

    def debug_draw_buttons(
        self,
        img: np.ndarray,
        buttons: List[DetectedButton],
    ) -> np.ndarray:
        """Draw detected buttons on an image for visual debugging."""
        vis = img.copy()
        for i, btn in enumerate(buttons):
            color = (0, 255, 0) if btn.color == "green" else (0, 0, 255)
            cv2.rectangle(vis, (btn.x0, btn.y0), (btn.x1, btn.y1), color, 2)
            label = f"Btn{i+1} ({btn.center_x},{btn.center_y}) A={btn.area}"
            cv2.putText(vis, label, (btn.x0, btn.y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return vis
