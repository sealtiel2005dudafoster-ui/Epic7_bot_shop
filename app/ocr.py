"""
OCR utilities for price extraction in the Epic Seven Secret Shop.

Tesseract is used to read gold prices, which may appear as:
  - "18,000"  (comma as thousands separator)
  - "184,000"
  - "280,000"

The parser tolerates OCR noise: extra characters, missing commas,
common digit confusions (O→0, S→5, etc.).
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
    _HAS_PYTESSERACT = True
except ImportError:
    _HAS_PYTESSERACT = False


# Common OCR confusions for digits in price text
_OCR_CHAR_FIXES = {
    "O": "0", "o": "0",
    "I": "1", "l": "1", "i": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2",
}


def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR on price text."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Upscale 2x for better OCR accuracy
    h, w = gray.shape
    if w < 100:
        gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    else:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Noise removal
    gray = cv2.fastNlMeansDenoising(gray)

    # Binarize — try both normal and inverse to handle dark-on-light or light-on-dark
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def _clean_price_text(text: str) -> str:
    """Apply OCR character fixes to digit-like strings."""
    return "".join(_OCR_CHAR_FIXES.get(ch, ch) for ch in text)


def extract_price_with_confidence(
    img: np.ndarray,
    tesseract_config: str = "--psm 7 --oem 3",
) -> tuple[int | None, float]:
    """
    Extract a gold price from an image ROI and return a confidence score.

    Returns (price, confidence) where:
      - price is the parsed integer, or None if parsing fails.
      - confidence (0.0–1.0) measures how much of the OCR text looks numeric.

    Accepts formats like "18,000", "184000", "2 800", etc.
    """
    if not _HAS_PYTESSERACT:
        raise RuntimeError("pytesseract not installed. Install with: pip install pytesseract")

    processed = _preprocess_for_ocr(img)
    raw_text = pytesseract.image_to_string(processed, config=tesseract_config)
    cleaned = _clean_price_text(raw_text.strip())

    # Confidence = how much of the OCR text looks like a number
    digits = re.sub(r"[^\d]", "", cleaned)
    total_chars = len(cleaned.replace(" ", ""))

    if total_chars == 0:
        return None, 0.0

    confidence = len(digits) / total_chars

    try:
        price = int(digits) if digits else None
    except ValueError:
        price = None

    return price, round(confidence, 3)


def extract_price(img: np.ndarray, tesseract_config: str = "--psm 7 --oem 3") -> int | None:
    """
    Extract a gold price from an image ROI.

    Returns the price as an integer, or None if parsing fails.
    Thin wrapper around ``extract_price_with_confidence``.
    """
    price, _ = extract_price_with_confidence(img, tesseract_config)
    return price


def save_ocr_debug(img: np.ndarray, output_path: str | Path):
    """Save a preprocessed image for debugging OCR issues."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    processed = _preprocess_for_ocr(img)
    cv2.imwrite(str(output_path), processed)

