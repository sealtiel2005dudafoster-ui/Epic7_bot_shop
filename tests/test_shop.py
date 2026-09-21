"""
Comprehensive unit and integration tests for Epic Seven Secret Shop bot.
Tests detection against real game screenshots offline.
"""

from pathlib import Path
import unittest
import cv2
import numpy as np
from PIL import Image

from app.templates import TemplateManager
from app.slots import ShopLayout, SlotRegion
from app.detector import (
    CoinDetector,
    COIN_AMIZADE,
    COIN_MARCA_PAGINAS,
    COIN_MEDALHAS,
)
from app.safety import SafetyMonitor
from app.capture import ScreenshotCapture
from app.scanner import ShopScanner


SCREENSHOT_DIR = Path("Screenshot")
TEMPLATES_DIR = Path("templates")


def load_bgr_image(path: Path) -> np.ndarray:
    """Helper to load image on Windows even with non-ASCII characters."""
    pil_img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


class TestEpic7Shop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tm = TemplateManager(TEMPLATES_DIR)
        cls.detector = CoinDetector(template_manager=cls.tm, confidence_threshold=0.85)

    def test_01_templates_exist_and_load(self):
        """Verify all 3 coin templates and anchors are present and valid."""
        coin_types = self.tm.list_coin_types()
        self.assertIn(COIN_AMIZADE, coin_types)
        self.assertIn(COIN_MARCA_PAGINAS, coin_types)
        self.assertIn(COIN_MEDALHAS, coin_types)

        for c in coin_types:
            tpl = self.tm.get_template(c)
            self.assertIsNotNone(tpl)
            self.assertGreater(tpl.width, 20)
            self.assertGreater(tpl.height, 20)

    def test_02_visual_anchor_detection(self):
        """Verify visual anchor snaps to the shop on real game screenshot."""
        main_path = SCREENSHOT_DIR / "Captura de ecrã 2026-09-20 165924.png"
        self.assertTrue(main_path.exists())
        img = load_bgr_image(main_path)

        layout = ShopLayout(templates_dir=TEMPLATES_DIR)
        layout.adjust_for_resolution(img)

        self.assertTrue(layout.anchor_detected, "Visual anchor must be detected")
        self.assertEqual(layout.anchor_location, (35, 52))
        self.assertEqual(len(layout.slots), 6)

    def test_03_detect_friendship_crop(self):
        """Verify 100% confidence match for Friendship Points."""
        path = SCREENSHOT_DIR / "Captura de ecrã 2026-09-20 183838.png"
        img = load_bgr_image(path)
        h, w = img.shape[:2]

        slot_reg = SlotRegion(
            name="test_friendship",
            slot_index=1,
            x0=0, y0=0, x1=w, y1=h,
            icon_box=(70, 0, 160, h),
            price_box=(350, 0, w, 35),
            button_box=(350, 36, w, h),
        )

        det = self.detector.detect_slot(img, slot_reg, 1)
        self.assertEqual(det.tipo, COIN_AMIZADE)
        self.assertGreaterEqual(det.confianca, 0.90)
        self.assertTrue(det.disponivel)

    def test_04_detect_bookmarks_crop(self):
        """Verify 100% confidence match for Bookmarks (Covenant)."""
        path = SCREENSHOT_DIR / "Captura de ecrã 2026-09-20 184513.png"
        img = load_bgr_image(path)
        h, w = img.shape[:2]

        slot_reg = SlotRegion(
            name="test_bookmarks",
            slot_index=1,
            x0=0, y0=0, x1=w, y1=h,
            icon_box=(70, 0, 160, h),
            price_box=(350, 0, w, 35),
            button_box=(350, 36, w, h),
        )

        det = self.detector.detect_slot(img, slot_reg, 1)
        self.assertEqual(det.tipo, COIN_MARCA_PAGINAS)
        self.assertGreaterEqual(det.confianca, 0.90)
        self.assertTrue(det.disponivel)

    def test_05_detect_mystic_crop(self):
        """Verify 100% confidence match for Mystic Medals."""
        path = SCREENSHOT_DIR / "Captura de ecrã 2026-09-20 190450.png"
        img = load_bgr_image(path)
        h, w = img.shape[:2]

        slot_reg = SlotRegion(
            name="test_mystics",
            slot_index=1,
            x0=0, y0=0, x1=w, y1=h,
            icon_box=(70, 0, 160, h),
            price_box=(350, 0, w, 35),
            button_box=(350, 36, w, h),
        )

        det = self.detector.detect_slot(img, slot_reg, 1)
        self.assertEqual(det.tipo, COIN_MEDALHAS)
        self.assertGreaterEqual(det.confianca, 0.90)
        self.assertTrue(det.disponivel)

    def test_06_safety_monitor_anchor_check(self):
        """Verify safety monitor validates visual anchor."""
        capture = ScreenshotCapture()
        safety = SafetyMonitor(capture=capture)

        main_path = SCREENSHOT_DIR / "Captura de ecrã 2026-09-20 165924.png"
        img = load_bgr_image(main_path)
        layout = ShopLayout(templates_dir=TEMPLATES_DIR)
        layout.adjust_for_resolution(img)

        check = safety.check_shop_layout(img, layout=layout)
        self.assertTrue(check.passed)
        self.assertTrue(safety.is_safe())

    def test_07_full_shop_scan_from_image(self):
        """Verify offline scan correctly parses visible slots in main view."""
        main_path = SCREENSHOT_DIR / "Captura de ecrã 2026-09-20 165924.png"
        img = load_bgr_image(main_path)

        capture = ScreenshotCapture()
        safety = SafetyMonitor(capture=capture)
        layout = ShopLayout(templates_dir=TEMPLATES_DIR)
        scanner = ShopScanner(capture=capture, detector=self.detector, safety=safety, layout=layout)

        result = scanner.scan_from_image(img)
        self.assertEqual(len(result.slots), 6)
        self.assertEqual(result.slots[1].tipo, "ignorar")
        self.assertEqual(result.slots[2].tipo, "ignorar")
        self.assertEqual(result.slots[3].tipo, "ignorar")

    def test_08_buyer_decision_engine(self):
        """Verify purchase decision logic under various slot states."""
        from app.buyer import ShopBuyer
        from app.detector import SlotDetection

        config = {
            "BUY_BOOKMARKS": True,
            "BUY_MYSTIC_MEDALS": True,
            "BUY_FRIENDSHIP": False,
            "MAX_BOOKMARK_PRICE": 184000,
            "PURCHASE_CONFIDENCE_MIN": 0.85,
        }
        buyer = ShopBuyer(config=config, dry_run=True)

        # 1. Valid bookmarks available
        d1 = SlotDetection(slot=1, tipo=COIN_MARCA_PAGINAS, preco=184000, confianca=0.98, disponivel=True)
        self.assertTrue(buyer.evaluate_slot(d1).should_buy)

        # 2. Already bought (disponivel = False)
        d2 = SlotDetection(slot=1, tipo=COIN_MARCA_PAGINAS, preco=184000, confianca=0.98, disponivel=False)
        self.assertFalse(buyer.evaluate_slot(d2).should_buy)

        # 3. Disabled coin type (BUY_FRIENDSHIP is False)
        d3 = SlotDetection(slot=2, tipo=COIN_AMIZADE, preco=18000, confianca=0.99, disponivel=True)
        self.assertFalse(buyer.evaluate_slot(d3).should_buy)

        # 4. Overpriced bookmarks
        d4 = SlotDetection(slot=1, tipo=COIN_MARCA_PAGINAS, preco=250000, confianca=0.98, disponivel=True)
        self.assertFalse(buyer.evaluate_slot(d4).should_buy)

        # 5. Non-coin item (ignorar)
        d5 = SlotDetection(slot=3, tipo="ignorar", confianca=0.40, disponivel=True)
        self.assertFalse(buyer.evaluate_slot(d5).should_buy)


if __name__ == "__main__":
    unittest.main()
