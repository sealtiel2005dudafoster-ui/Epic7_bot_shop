"""
Template management for coin icon detection.

Loads template images from the templates/ directory and provides
matching against screenshots using OpenCV template matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class MatchResult:
    """Result of a template match on a region of interest."""
    found: bool
    confidence: float
    location: Optional[tuple[int, int]] = None  # (x, y) top-left of best match
    template_size: Optional[tuple[int, int]] = None  # (w, h) of the template


@dataclass
class CoinTemplate:
    """A loaded template for one coin type."""
    name: str            # "amizade", "marca_paginas", "medalhas_misticas"
    display_name: str    # "Pontos de Amizade", etc.
    image: np.ndarray
    width: int
    height: int

    def match(self, roi: np.ndarray) -> MatchResult:
        """
        Run multi-scale template matching on the given ROI.
        Returns a MatchResult with the best confidence found.

        We use normalized cross-correlation (TM_CCOEFF_NORMED) which is
        robust to lighting variations common on mobile game emulations.
        """
        if roi is None or roi.size == 0:
            return MatchResult(found=False, confidence=0.0)

        # Ensure we have a grayscale version of the ROI for template matching
        if len(roi.shape) == 2:
            roi_c = roi
        else:
            roi_c = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        best_confidence = 0.0
        best_location: Optional[tuple[int, int]] = None

        # Try the original template first
        template_gray = self.image if len(self.image.shape) == 2 else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        if roi_c.shape[0] >= template_gray.shape[0] and roi_c.shape[1] >= template_gray.shape[1]:
            result = cv2.matchTemplate(roi_c, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_confidence:
                best_confidence = max_val
                best_location = max_loc

        # Try multi-scale matching for robustness across different window resolutions
        for scale in [0.70, 0.80, 0.88, 0.95, 1.0, 1.05, 1.12, 1.20, 1.30, 1.45, 1.60, 1.80, 2.0, 2.25]:
            scaled_w = int(self.width * scale)
            scaled_h = int(self.height * scale)
            if scaled_w < 10 or scaled_h < 10:
                continue
            if scaled_w > roi_c.shape[1] or scaled_h > roi_c.shape[0]:
                continue
            resized = cv2.resize(template_gray, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(roi_c, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_confidence:
                best_confidence = max_val
                best_location = max_loc

        return MatchResult(
            found=best_confidence > 0.0,
            confidence=float(best_confidence),
            location=best_location,
            template_size=(self.width, self.height),
        )


class TemplateManager:
    """Loads and manages all coin templates from a directory."""

    # Maps template filename → (coin_type, display_name)
    TEMPLATE_SPECS = {
        "friendship.png": ("amizade", "Pontos de Amizade"),
        "bookmarks.png": ("marca_paginas", "Marca-Páginas da Aliança"),
        "mystic_medals.png": ("medalhas_misticas", "Medalhas Místicas"),
    }

    def __init__(self, templates_dir: str | Path):
        self.templates_dir = Path(templates_dir)
        self._templates: dict[str, CoinTemplate] = {}
        self._load_all()

    def _load_all(self):
        """Load all template images from the templates directory."""
        if not self.templates_dir.exists():
            raise FileNotFoundError(
                f"Templates directory not found: {self.templates_dir}. "
                f"Expected files: {list(self.TEMPLATE_SPECS.keys())}"
            )

        for filename, (coin_type, display_name) in self.TEMPLATE_SPECS.items():
            filepath = self.templates_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Template file not found: {filepath}. "
                    f"Please provide a screenshot of the {display_name} icon."
                )
            img = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load image: {filepath}")
            h, w = img.shape[:2]
            self._templates[coin_type] = CoinTemplate(
                name=coin_type,
                display_name=display_name,
                image=img,
                width=w,
                height=h,
            )

    def get_template(self, coin_type: str) -> Optional[CoinTemplate]:
        return self._templates.get(coin_type)

    @property
    def all_templates(self) -> dict[str, CoinTemplate]:
        return dict(self._templates)

    def list_coin_types(self) -> list[str]:
        return list(self._templates.keys())
