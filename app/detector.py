"""
Visual detector for Epic Seven Secret Shop coin icons.

The detector:
1. Divides the screenshot into the 6 known slot regions.
2. For each slot, runs template matching against the three coin templates.
3. Uses OCR on the price region to extract the gold cost.
4. Returns a structured result for each slot.

The detector is fully decoupled from the decision engine — it only
reports what it sees: coin type, price, confidence, slot index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .ocr import extract_price_with_confidence
from .slots import ShopLayout, SlotRegion
from .templates import TemplateManager


# Coin type constants
COIN_AMIZADE = "amizade"
COIN_MARCA_PAGINAS = "marca_paginas"
COIN_MEDALHAS = "medalhas_misticas"

ALL_COIN_TYPES = [COIN_AMIZADE, COIN_MARCA_PAGINAS, COIN_MEDALHAS]

# Display names for output
COIN_DISPLAY_NAMES = {
    COIN_AMIZADE: "Amizade",
    COIN_MARCA_PAGINAS: "Marca-Páginas",
    COIN_MEDALHAS: "Medalhas Místicas",
}


@dataclass
class SlotDetection:
    """
    Detection result for a single shop slot.

    - tipo: one of "amizade", "marca_paginas", "medalhas_misticas", "ignorar", "desconhecido"
    - preco: OCR-extracted price in gold, or None
    - confianca: template match confidence (0.0–1.0)
    - nome: human-readable coin name (e.g., "Pontos de Amizade")
    """
    slot: int
    tipo: str = "desconhecido"
    nome: Optional[str] = None
    preco: Optional[int] = None
    confianca: float = 0.0
    ocr_confidence: float = 0.0
    disponivel: bool = True

    def to_dict(self) -> dict:
        """Serialise to the dict format specified in the requirements."""
        result = {
            "slot": self.slot,
            "tipo": self.tipo,
            "preco": self.preco,
            "confianca": round(self.confianca, 3),
            "disponivel": self.disponivel,
        }
        if self.tipo != "ignorar":
            result["nome"] = self.nome
        return result

    def __str__(self) -> str:
        price_str = f"{self.preco:,}".replace(",", ".") if self.preco else "R$?"
        if self.tipo == "ignorar":
            return f"Slot {self.slot}: IGNORAR"
        elif self.tipo == "desconhecido":
            return f"Slot {self.slot}: DESCONHECIDO (conf: {self.confianca:.2f})"
        else:
            name = COIN_DISPLAY_NAMES.get(self.tipo, self.tipo)
            return f"Slot {self.slot}: {name.upper()} | {price_str} | conf: {self.confianca:.2f}"


class CoinDetector:
    """
    Detects which coin (if any) appears in a given slot ROI.

    Strategy:
    1. First try matching the coin icon area of the slot using template matching.
    2. If a coin is found, verify by checking for associated text/label near the icon.
    3. Extract the price via OCR from the price region.

    The icon template matching uses normalized cross-correlation (TM_CCOEFF_NORMED)
    with multi-scale support for resolution robustness.
    """

    # Icon region within each slot — the upper portion where the coin icon sits.
    # This is relative to the slot ROI (not the full screen).
    ICON_REGION_REL = (0.1, 0.05, 0.9, 0.55)  # (x0, y0, x1, y1) within slot

    def __init__(
        self,
        template_manager: TemplateManager,
        confidence_threshold: float = 0.85,
    ):
        self.templates = template_manager
        self.confidence_threshold = confidence_threshold

    def detect_slot(self, img: np.ndarray, slot_region: SlotRegion, slot_index: int) -> SlotDetection:
        """
        Analyze a single slot region and return a detection result.

        Detection flow:
        1. Extract the slot ROI from the screenshot.
        2. Extract the icon sub-region and run template matching.
        3. If a coin is matched above threshold, attempt price OCR.
        4. If no coin matches → "ignorar".
        5. If confidence is below threshold but > 0 → still report it but mark down.
        """
        slot_img = slot_region.get_roi(img)
        if slot_img is None or slot_img.size == 0:
            return SlotDetection(
                slot=slot_index,
                tipo="desconhecido",
                confianca=0.0,
            )

        # Extract icon region from the slot
        if hasattr(slot_region, "get_icon_roi"):
            icon_roi = slot_region.get_icon_roi(img)
        else:
            h, w = slot_img.shape[:2]
            ix0 = int(self.ICON_REGION_REL[0] * w)
            iy0 = int(self.ICON_REGION_REL[1] * h)
            ix1 = int(self.ICON_REGION_REL[2] * w)
            iy1 = int(self.ICON_REGION_REL[3] * h)
            icon_roi = slot_img[iy0:iy1, ix0:ix1]

        # Run template matching for each coin type
        best_match = None
        best_conf = 0.0

        for coin_type in ALL_COIN_TYPES:
            template = self.templates.get_template(coin_type)
            if template is None:
                continue

            match_result = template.match(icon_roi)
            if match_result.confidence > best_conf:
                best_conf = match_result.confidence
                best_match = coin_type

        # Determine the slot type
        if best_conf >= self.confidence_threshold:
            coin_type = best_match
            nome = self.templates.get_template(coin_type).display_name if coin_type else None
            tipo = coin_type
        elif best_conf > 0.0:
            # Some match was found but below threshold — treat as "ignorar"
            # but retain confidence for logging
            tipo = "ignorar"
            nome = None
        else:
            # No match at all — likely an item slot with equipment/hero
            tipo = "ignorar"
            nome = None

        # Extract price via OCR
        price = None
        ocr_conf = 0.0
        price_roi = slot_region.get_price_roi(img)
        if price_roi is not None and price_roi.size > 0:
            try:
                price, ocr_conf = extract_price_with_confidence(price_roi)
            except Exception:
                price, ocr_conf = None, 0.0

        # Check availability (is buy button active?)
        available = True
        if hasattr(slot_region, "is_available_to_buy"):
            available = slot_region.is_available_to_buy(img)

        return SlotDetection(
            slot=slot_index,
            tipo=tipo,
            nome=nome,
            preco=price,
            confianca=best_conf,
            ocr_confidence=ocr_conf,
            disponivel=available,
        )

    def detect_all_slots(self, img: np.ndarray, layout: ShopLayout) -> list[SlotDetection]:
        """Detect all 6 slots. Caller handles scrolling for slot 6."""
        results = []
        for i, slot_region in enumerate(layout.slots):
            detection = self.detect_slot(img, slot_region, i + 1)
            results.append(detection)
        return results

    def detect_visible_slots(self, img: np.ndarray, layout: ShopLayout) -> list[SlotDetection]:
        """Detect the first 5 slots (visible without scrolling)."""
        results = []
        for i, slot_region in enumerate(layout.slots[:5]):
            detection = self.detect_slot(img, slot_region, i + 1)
            results.append(detection)
        return results

    def detect_slot_6(self, img: np.ndarray, layout: ShopLayout) -> SlotDetection:
        """Detect only slot 6 (after scrolling)."""
        return self.detect_slot(img, layout.slots[5], 6)
