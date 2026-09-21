"""Regression coverage for position-independent action identifiers."""

from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image

from app.vision import VisionDetector


ROOT = Path(__file__).resolve().parents[1]


def load_bgr(path: Path) -> np.ndarray:
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


class TestActionIdentifiers(unittest.TestCase):
    def setUp(self) -> None:
        self.vision = VisionDetector()
        self.popup = cv2.imread(str(ROOT / "debug_popup_buttons.png"))
        self.shop = load_bgr(ROOT / "Screenshot" / "Captura de ecrã 2026-09-20 165924.png")

    def test_modal_confirm_requires_cancel_confirm_pair(self) -> None:
        button = self.vision.find_popup_confirm_button(self.popup)
        self.assertIsNotNone(button)
        self.assertGreater(button.center_x, 350)
        self.assertLess(button.center_x, 550)
        self.assertGreater(button.center_y, 300)
        self.assertLess(button.center_y, 370)

    def test_shop_does_not_fake_a_modal_confirmation(self) -> None:
        self.assertIsNone(self.vision.find_popup_confirm_button(self.shop))

    def test_refresh_uses_its_visual_template(self) -> None:
        button = self.vision.find_refresh_button(self.shop)
        self.assertIsNotNone(button)
        self.assertGreater(button.center_x, 50)
        self.assertLess(button.center_x, 230)
        self.assertGreater(button.center_y, 380)
        self.assertLess(button.center_y, 465)


if __name__ == "__main__":
    unittest.main()
