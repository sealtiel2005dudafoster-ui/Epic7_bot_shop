"""
Slot layout definitions for the Epic Seven Secret Shop.

The shop UI displays items on the right side as a vertical list of rows.
Each row (slot) contains:
  - Character portrait (left)
  - Coin/Item icon + quantity (sub-region)
  - Item name / type (sub-region)
  - Gold price (sub-region)
  - Buy button "1/1 Comprar" / "0/1 Comprar" (sub-region)

The layout is dynamic and mobile:
  - It searches for a visual anchor (e.g., 'anchor_header.png' or 'anchor_refresh.png')
    to locate the exact origin and scale of the shop UI within the window,
    completely eliminating issues with Windows title bars or resolution changes.
  - If no anchor is detected, it falls back to resolution-relative proportions
    of the right-hand side of the screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class SlotRegion:
    """A region of interest for a single shop slot row, with its sub-regions."""
    name: str
    slot_index: int
    # Absolute pixel bounding box of this slot row: (x0, y0, x1, y1)
    x0: int
    y0: int
    x1: int
    y1: int
    # Sub-regions (absolute pixel coordinates)
    icon_box: Optional[tuple[int, int, int, int]] = None   # (x0, y0, x1, y1)
    price_box: Optional[tuple[int, int, int, int]] = None  # (x0, y0, x1, y1)
    button_box: Optional[tuple[int, int, int, int]] = None # (x0, y0, x1, y1)

    # Legacy relative fields for backwards compatibility
    x_start: float = 0.0
    y_start: float = 0.0
    x_end: float = 0.0
    y_end: float = 0.0
    price_region: Optional[tuple[float, float, float, float]] = None

    def to_absolute(self, screen_w: int, screen_h: int) -> tuple[int, int, int, int]:
        """Convert to absolute pixel coordinates."""
        if self.x1 > self.x0 and self.y1 > self.y0:
            return self.x0, self.y0, self.x1, self.y1
        return int(self.x_start * screen_w), int(self.y_start * screen_h), int(self.x_end * screen_w), int(self.y_end * screen_h)

    def get_roi(self, img: np.ndarray) -> np.ndarray:
        """Extract the full slot row from a screenshot."""
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self.to_absolute(w, h)
        return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]

    def get_icon_roi(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract the coin/item icon sub-region."""
        if self.icon_box is not None:
            x0, y0, x1, y1 = self.icon_box
            h, w = img.shape[:2]
            return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        slot_img = self.get_roi(img)
        sh, sw = slot_img.shape[:2]
        return slot_img[int(sh * 0.05):int(sh * 0.95), int(sw * 0.10):int(sw * 0.35)]

    def get_price_roi(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract the price sub-region."""
        if self.price_box is not None:
            x0, y0, x1, y1 = self.price_box
            h, w = img.shape[:2]
            return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        if self.price_region is not None:
            h, w = img.shape[:2]
            x0 = int(self.price_region[0] * w)
            y0 = int(self.price_region[1] * h)
            x1 = int(self.price_region[2] * w)
            y1 = int(self.price_region[3] * h)
            return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        slot_img = self.get_roi(img)
        sh, sw = slot_img.shape[:2]
        return slot_img[0:int(sh * 0.40), int(sw * 0.70):sw]

    def get_button_roi(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract the buy button sub-region."""
        if self.button_box is not None:
            x0, y0, x1, y1 = self.button_box
            h, w = img.shape[:2]
            return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        slot_img = self.get_roi(img)
        sh, sw = slot_img.shape[:2]
        return slot_img[int(sh * 0.40):sh, int(sw * 0.70):sw]

    def is_available_to_buy(self, img: np.ndarray) -> bool:
        """
        Check if item is available (active green button '1/1')
        vs already purchased / disabled (dark button '0/1').
        """
        btn = self.get_button_roi(img)
        if btn is None or btn.size == 0:
            return False
        # Active button in Epic Seven has bright green pixels (high G, lower R and B)
        green_pixels = (btn[:, :, 1] > 110) & (btn[:, :, 0] < 90) & (btn[:, :, 2] < 90)
        return int(np.sum(green_pixels)) > 20


@dataclass
class ShopLayout:
    """
    Dynamic layout for the Epic Seven Secret Shop.

    Features:
      - Mobile/adaptive frame based on visual anchor template matching.
      - Automatically snaps to the shop list regardless of window position,
        title bars, or resolution scaling.
    """
    screen_width: int = 742
    screen_height: int = 474
    templates_dir: Path = field(default_factory=lambda: Path("templates"))
    anchor_detected: bool = False
    anchor_location: tuple[int, int] = (0, 0)
    scale: float = 1.0
    slots: list[SlotRegion] = field(default_factory=list)

    # Reference metrics at 742x474 (calibrated from game screenshots):
    REF_ANCHOR_REL_X = 225  # Shop frame x relative to header anchor
    REF_ANCHOR_REL_Y = 54   # Shop frame y relative to header anchor
    REF_SLOT_WIDTH = 465
    REF_SLOT_HEIGHT = 84

    def __post_init__(self):
        if not self.slots:
            self._build_fallback_slots(self.screen_width, self.screen_height)

    def _build_fallback_slots(self, w: int, h: int):
        """Construct fallback resolution-relative vertical slots."""
        self.slots = []
        # Fallback: items occupy roughly right 60% of screen, from y=22% to 95%
        x0 = int(w * 0.35)
        x1 = int(w * 0.98)
        y_start = int(h * 0.22)
        slot_h = int(h * 0.177)

        for i in range(6):
            sy0 = y_start + i * slot_h
            sy1 = sy0 + slot_h
            self.slots.append(
                SlotRegion(
                    name=f"slot_{i+1}",
                    slot_index=i + 1,
                    x0=x0, y0=sy0, x1=x1, y1=sy1,
                    icon_box=(x0 + int((x1 - x0) * 0.13), sy0, x0 + int((x1 - x0) * 0.30), sy0 + int(slot_h * 0.75)),
                    price_box=(x0 + int((x1 - x0) * 0.75), sy0, x1, sy0 + int(slot_h * 0.38)),
                    button_box=(x0 + int((x1 - x0) * 0.75), sy0 + int(slot_h * 0.40), x1, sy1 - 5),
                    x_start=x0 / w, y_start=sy0 / h, x_end=x1 / w, y_end=sy1 / h,
                )
            )

    def adjust_for_resolution(self, img: np.ndarray) -> "ShopLayout":
        """
        Dynamically find visual anchor in screenshot and snap the shop frame.
        Falls back to proportional grid if anchor is not found.
        """
        h, w = img.shape[:2]
        self.screen_width = w
        self.screen_height = h

        # Try to match visual anchor
        anchor_path = self.templates_dir / "anchor_header.png"
        if anchor_path.exists():
            tpl = cv2.imread(str(anchor_path))
            if tpl is not None and h >= tpl.shape[0] and w >= tpl.shape[1]:
                # Template matching in grayscale
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if len(tpl.shape) == 3 else tpl

                res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val >= 0.70:
                    self.anchor_detected = True
                    self.anchor_location = max_loc
                    hx, hy = max_loc

                    # Calculate dynamic frame
                    frame_x = hx + int(self.REF_ANCHOR_REL_X * self.scale)
                    frame_y = hy + int(self.REF_ANCHOR_REL_Y * self.scale)
                    frame_w = int(self.REF_SLOT_WIDTH * self.scale)
                    slot_h = int(self.REF_SLOT_HEIGHT * self.scale)

                    self.slots = []
                    for i in range(6):
                        sy0 = frame_y + i * slot_h
                        sy1 = sy0 + slot_h
                        sx0 = frame_x
                        sx1 = frame_x + frame_w

                        # Expanded search box margins (ensures icon is never cut off):
                        icon_y0 = max(0, sy0 - int(12 * self.scale))
                        icon_y1 = min(h, sy1 + int(8 * self.scale))
                        icon_x0 = sx0 + int(35 * self.scale)
                        icon_x1 = sx0 + int(165 * self.scale)
                        icon_box = (icon_x0, icon_y0, icon_x1, icon_y1)

                        price_box = (sx0 + int(340 * self.scale), sy0, sx1, sy0 + int(35 * self.scale))
                        button_box = (sx0 + int(340 * self.scale), sy0 + int(35 * self.scale), sx1, sy1)

                        self.slots.append(
                            SlotRegion(
                                name=f"slot_{i+1}",
                                slot_index=i + 1,
                                x0=sx0, y0=sy0, x1=sx1, y1=sy1,
                                icon_box=icon_box,
                                price_box=price_box,
                                button_box=button_box,
                                x_start=sx0 / w, y_start=sy0 / h, x_end=sx1 / w, y_end=sy1 / h,
                            )
                        )
                    return self

        # Fallback if anchor is not detected
        self.anchor_detected = False
        self._build_fallback_slots(w, h)
        return self

    def get_slot_by_index(self, index: int) -> SlotRegion:
        return self.slots[index]

    def get_visible_slots(self) -> list[SlotRegion]:
        return self.slots[:5]

    def get_scrolled_slots(self) -> list[SlotRegion]:
        return self.slots[5:]

    @classmethod
    def for_resolution(cls, screen_w: int, screen_h: int) -> "ShopLayout":
        instance = cls(screen_width=screen_w, screen_height=screen_h)
        instance._build_fallback_slots(screen_w, screen_h)
        return instance

