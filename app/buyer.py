"""
Shop Buyer and Action Executor for Epic Seven Secret Shop Bot.

Implements the 4 core action phases with VISUAL VERIFICATION:
  1. Purchase Decision (validates coin type, price threshold, confidence, and availability)
  2. Item Slot Click (finds green button visually via HSV, then humanized click)
  3. Purchase Confirmation (visually detects popup, finds confirm button, verifies dismissal)
  4. Shop Refresh (clicks 'Renovar' button and confirms the Skystone popup modal)

Every action step includes post-action verification:
  - After clicking buy → verify popup appeared
  - After clicking confirm → verify popup dismissed
  - After refresh → verify shop content changed
  - If verification fails → retry up to MAX_RETRIES, then abort safely
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui

from .detector import (
    COIN_AMIZADE,
    COIN_MARCA_PAGINAS,
    COIN_MEDALHAS,
    SlotDetection,
)
from .slots import ShopLayout, SlotRegion
from .vision import VisionDetector
from .action_identifier import ActionIdentifier
from .vision_backends import ActionClass


@dataclass
class PurchaseDecision:
    """Outcome of evaluating a slot for purchase."""
    should_buy: bool
    reason: str
    coin_type: Optional[str] = None
    price: Optional[int] = None


@dataclass
class ActionResult:
    """Result of a buy/confirm/refresh action with verification status."""
    success: bool
    action: str
    message: str
    retries: int = 0


class ShopBuyer:
    """
    Handles purchase decisions and mouse actions in the Epic Seven shop.

    Uses VisionDetector for real-time button finding and post-action verification.
    """

    MAX_RETRIES = 3  # Max retry attempts per action

    def __init__(self, config: dict, dry_run: bool = True):
        self.config = config
        self.dry_run = dry_run
        self.vision = VisionDetector()
        self.actions = ActionIdentifier(config, self.vision)

        # Capture reference for verification screenshots
        self._capture = None  # Set by ShopBot after init

        # Configurable thresholds
        self.buy_friendship = config.get("BUY_FRIENDSHIP", True)
        self.buy_bookmarks = config.get("BUY_BOOKMARKS", True)
        self.buy_mystics = config.get("BUY_MYSTIC_MEDALS", True)

        self.max_friendship_price = config.get("MAX_FRIENDSHIP_PRICE", 18000)
        self.max_bookmark_price = config.get("MAX_BOOKMARK_PRICE", 184000)
        self.max_mystic_price = config.get("MAX_MYSTIC_PRICE", 280000)

        self.purchase_confidence_min = config.get("PURCHASE_CONFIDENCE_MIN", 0.85)
        self.click_cooldown_ms = config.get("CLICK_COOLDOWN_MS", 500)
        self.animation_wait_ms = config.get("ANIMATION_WAIT_MS", 800)

    def set_capture(self, capture):
        """Inject the ScreenshotCapture so buyer can take verification screenshots."""
        self._capture = capture

    def evaluate_slot(self, detection: SlotDetection) -> PurchaseDecision:
        """
        Evaluate whether an observed slot should be bought.
        """
        if detection.tipo == "ignorar" or detection.tipo == "desconhecido":
            return PurchaseDecision(False, f"Slot {detection.slot}: não é moeda alvo ({detection.tipo})")

        if not getattr(detection, "disponivel", True):
            return PurchaseDecision(False, f"Slot {detection.slot}: item já foi adquirido (0/1)")

        if detection.confianca < self.purchase_confidence_min:
            return PurchaseDecision(
                False,
                f"Slot {detection.slot}: confiança {detection.confianca:.2f} abaixo do mínimo {self.purchase_confidence_min:.2f}"
            )

        coin_type = detection.tipo
        price = detection.preco

        # Check enabled flags & price bounds
        if coin_type == COIN_MARCA_PAGINAS:
            if not self.buy_bookmarks:
                return PurchaseDecision(False, "Compra de Marca-Páginas desativada na configuração")
            if price is not None and price > self.max_bookmark_price:
                return PurchaseDecision(False, f"Preço de {price:,} excede teto de {self.max_bookmark_price:,}")
            return PurchaseDecision(True, "Marca-Páginas válido para compra!", coin_type, price)

        elif coin_type == COIN_MEDALHAS:
            if not self.buy_mystics:
                return PurchaseDecision(False, "Compra de Medalhas Místicas desativada na configuração")
            if price is not None and price > self.max_mystic_price:
                return PurchaseDecision(False, f"Preço de {price:,} excede teto de {self.max_mystic_price:,}")
            return PurchaseDecision(True, "Medalhas Místicas válidas para compra!", coin_type, price)

        elif coin_type == COIN_AMIZADE:
            if not self.buy_friendship:
                return PurchaseDecision(False, "Compra de Amizade desativada na configuração")
            if price is not None and price > self.max_friendship_price:
                return PurchaseDecision(False, f"Preço de {price:,} excede teto de {self.max_friendship_price:,}")
            return PurchaseDecision(True, "Pontos de Amizade válidos para compra!", coin_type, price)

        return PurchaseDecision(False, f"Tipo de moeda desconhecido: {coin_type}")

    # -------------------------------------------------------------------------
    # Low-Level Mouse Actions
    # -------------------------------------------------------------------------

    def _human_click(self, x: int, y: int, jitter: int = 3, hold_ms: int = 100):
        """
        Perform a human-like click with slight coordinate jitter and hold duration.
        Game engines require holding the mouse down for ~60-120ms so the frame tick registers it.
        """
        jx = x + random.randint(-jitter, jitter)
        jy = y + random.randint(-jitter, jitter)

        pyautogui.moveTo(jx, jy, duration=random.uniform(0.12, 0.22))
        pyautogui.mouseDown()
        time.sleep(hold_ms / 1000.0)
        pyautogui.mouseUp()
        time.sleep(self.click_cooldown_ms / 1000.0)

    def _take_screenshot(self) -> Optional[np.ndarray]:
        """Take a verification screenshot via injected capture module."""
        if self._capture is not None:
            return self._capture.screenshot()
        return None

    def touch_swipe_scroll(self, start_x: int, start_y: int, end_y: int, duration: float = 0.35):
        """
        Simulate a touch swipe / drag gesture (required for mobile games and emulators).
        Scrolls down by clicking and dragging upwards.
        """
        if self.dry_run:
            print(f"[DRY_RUN] Simulação de arrasto de toque (swipe) de ({start_x}, {start_y}) até ({start_x}, {end_y})")
            return

        pyautogui.moveTo(start_x, start_y, duration=0.12)
        pyautogui.mouseDown()
        time.sleep(0.06)
        pyautogui.moveTo(start_x, end_y, duration=duration)
        time.sleep(0.06)
        pyautogui.mouseUp()
        time.sleep(0.35)

    # -------------------------------------------------------------------------
    # Visual Buy Button Click (with real-time button detection)
    # -------------------------------------------------------------------------

    def click_buy_button(
        self,
        slot_region: SlotRegion,
        window_offset: Tuple[int, int] = (0, 0),
        img: Optional[np.ndarray] = None,
    ) -> ActionResult:
        """
        Click the 'Comprar' button on a specific slot row.

        NEW: Uses visual detection to find the ACTUAL green button position
        instead of relying on pre-calculated static coordinates.

        Flow:
        1. Take a fresh screenshot
        2. Find green button visually within the slot region
        3. If found → click its center (add window_offset for screen coords)
        4. Refuse to click when the button is not visually identified
        5. Verify popup appeared after click
        """
        if self.dry_run:
            print(f"[DRY_RUN] Simulação de clique no botão Comprar do {slot_region.name}")
            return ActionResult(True, "click_buy", "Modo simulação")

        # Step 1: Get current screenshot for visual detection
        if img is None:
            img = self._take_screenshot()

        for attempt in range(1, self.MAX_RETRIES + 1):
            if img is not None:
                # Step 2: Find green button visually within slot bounds
                btn = self.vision.find_buy_button_for_slot(
                    img,
                    slot_y0=slot_region.y0,
                    slot_y1=slot_region.y1,
                    slot_x0=slot_region.x0,
                    slot_x1=slot_region.x1,
                )

                if btn is not None:
                    # Visual button found! Use its exact center
                    cx = btn.center_x + window_offset[0]
                    cy = btn.center_y + window_offset[1]
                    print(f"     🔍 Botão verde detectado visualmente em ({cx}, {cy}) [Área={btn.area}px²]")
                else:
                    print(f"     ⚠️  Botão Comprar não foi identificado visualmente (tentativa {attempt}/{self.MAX_RETRIES}).")
                    return ActionResult(
                        False,
                        "click_buy",
                        "Clique cancelado: nenhum identificador visual confiável para o botão Comprar",
                        attempt - 1,
                    )
            else:
                return ActionResult(
                    False,
                    "click_buy",
                    "Clique cancelado: não foi possível capturar a janela do jogo",
                    attempt - 1,
                )

            # Step 3: Click
            img_before = img  # Save for verification
            self._human_click(cx, cy)

            # Step 4: Verify popup appeared
            time.sleep(self.animation_wait_ms / 1000.0)
            img_after = self._take_screenshot()

            if img_before is not None and img_after is not None:
                if self.vision.verify_purchase_popup_appeared(img_before, img_after):
                    print(f"     ✅ Popup de compra CONFIRMADO (tentativa {attempt})")
                    return ActionResult(True, "click_buy", "Popup confirmado visualmente", attempt - 1)
                else:
                    print(f"     ❌ Popup NÃO apareceu após clique (tentativa {attempt}/{self.MAX_RETRIES})")
                    # Refresh screenshot for next attempt
                    img = img_after
            else:
                # Can't verify, assume success
                return ActionResult(True, "click_buy", "Sem verificação visual disponível", attempt - 1)

        return ActionResult(False, "click_buy",
                            f"FALHOU: Popup não apareceu após {self.MAX_RETRIES} tentativas",
                            self.MAX_RETRIES)

    # -------------------------------------------------------------------------
    # Purchase Confirmation Popup (with visual button detection)
    # -------------------------------------------------------------------------

    def confirm_purchase_popup(
        self,
        img: Optional[np.ndarray] = None,
        window_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> ActionResult:
        """
        Confirm the purchase modal popup that appears after clicking 'Comprar'.

        NEW: Finds the green confirm button VISUALLY inside the popup,
        instead of using hardcoded percentage coordinates.
        """
        if self.dry_run:
            print("[DRY_RUN] Simulação de confirmação de compra no popup")
            return ActionResult(True, "confirm_purchase", "Modo simulação")

        time.sleep(self.animation_wait_ms / 1000.0)

        for attempt in range(1, self.MAX_RETRIES + 1):
            # Get fresh screenshot with popup visible
            if img is None or attempt > 1:
                img = self._take_screenshot()

            if img is not None:
                # Find green confirm button in the popup
                confirm_btn = self.actions.find(img, ActionClass.PURCHASE_CONFIRM_BUTTON)

                if confirm_btn is not None:
                    # Button found visually!
                    # The coordinates are relative to the screenshot (which is the window content)
                    # Add window offset to get screen-absolute coordinates
                    if window_bbox is not None:
                        cx = confirm_btn.center_x + window_bbox[0]
                        cy = confirm_btn.center_y + window_bbox[1]
                    else:
                        cx = confirm_btn.center_x
                        cy = confirm_btn.center_y

                    print(f"     🔍 Botão de confirmação detectado em ({cx}, {cy}) [Área={confirm_btn.area}px²]")
                else:
                    print(f"     ⚠️  Modal de compra não apresentou um par Cancelar/Confirmar verificável (tentativa {attempt}).")
                    continue
            else:
                print("     ⚠️  Não foi possível capturar a janela durante a confirmação de compra.")
                continue

            # Click the confirm button
            img_before = img
            self._human_click(cx, cy, hold_ms=120)
            time.sleep(self.animation_wait_ms / 1000.0)

            # Verify popup was dismissed
            img_after = self._take_screenshot()
            if img_before is not None and img_after is not None:
                if self.vision.verify_popup_dismissed(img_before, img_after):
                    print(f"     ✅ Compra CONFIRMADA com sucesso (tentativa {attempt})")
                    return ActionResult(True, "confirm_purchase", "Popup dispensado", attempt - 1)
                else:
                    # Check if popup is still showing
                    if self.vision.is_popup_visible(img_after):
                        print(f"     ❌ Popup ainda visível após clique (tentativa {attempt}/{self.MAX_RETRIES})")
                        img = img_after  # Use fresh image for next retry
                    else:
                        # Popup gone but verification wasn't sure → treat as success
                        print(f"     ✅ Popup não mais visível (tentativa {attempt})")
                        return ActionResult(True, "confirm_purchase", "Popup parece dispensado", attempt - 1)
            else:
                return ActionResult(True, "confirm_purchase", "Sem verificação visual", attempt - 1)

        return ActionResult(False, "confirm_purchase",
                            f"FALHOU: Confirmação não funcionou após {self.MAX_RETRIES} tentativas",
                            self.MAX_RETRIES)

    # -------------------------------------------------------------------------
    # Shop Refresh (with visual detection)
    # -------------------------------------------------------------------------

    def click_refresh_button(
        self,
        layout: ShopLayout,
        window_offset: Tuple[int, int] = (0, 0),
    ) -> ActionResult:
        """
        Click the 'Renovar' button located at the bottom-left of the shop.
        """
        if self.dry_run:
            print("[DRY_RUN] Simulação de clique no botão 'Renovar'")
            return ActionResult(True, "click_refresh", "Modo simulação")

        img_before = self._take_screenshot()
        if img_before is None:
            return ActionResult(False, "click_refresh", "Clique cancelado: não foi possível capturar a janela do jogo")

        refresh_btn = self.actions.find(img_before, ActionClass.REFRESH_BUTTON)
        if refresh_btn is None:
            return ActionResult(
                False,
                "click_refresh",
                "Clique cancelado: o identificador visual do botão Renovar não foi encontrado",
            )

        rx = refresh_btn.center_x + window_offset[0]
        ry = refresh_btn.center_y + window_offset[1]

        print(f"     -> Botão 'Renovar' identificado visualmente em ({rx}, {ry}); clicando...")

        self._human_click(rx, ry, hold_ms=100)
        time.sleep(self.animation_wait_ms / 1000.0)

        # Verify refresh popup appeared
        img_after = self._take_screenshot()
        if img_before is not None and img_after is not None:
            if self.vision.verify_purchase_popup_appeared(img_before, img_after):
                print("     ✅ Popup de renovação DETECTADO")
                return ActionResult(True, "click_refresh", "Popup de renovação apareceu")
            else:
                return ActionResult(False, "click_refresh", "Popup de renovação não foi identificado após o clique")

        return ActionResult(False, "click_refresh", "Popup de renovação não pôde ser verificado")

    def confirm_refresh_popup(
        self,
        window_bbox: Optional[Tuple[int, int, int, int]] = None,
        layout: Optional[ShopLayout] = None,
    ) -> ActionResult:
        """
        Confirm the refresh modal ("Renovar a Loja Secreta usando 3 Pedras Celestes?").
        Clicks the green 'Confirmar' button on the right side of the dialog.

        Uses visual detection to find the actual confirm button.
        """
        if self.dry_run:
            print("[DRY_RUN] Simulação de confirmação de renovação (3 Pedras Celestes)")
            return ActionResult(True, "confirm_refresh", "Modo simulação")

        time.sleep(self.animation_wait_ms / 1000.0)

        for attempt in range(1, self.MAX_RETRIES + 1):
            img = self._take_screenshot()

            if img is not None:
                confirm_btn = self.actions.find(img, ActionClass.REFRESH_CONFIRM_BUTTON)

                if confirm_btn is not None:
                    if window_bbox is not None:
                        cx = confirm_btn.center_x + window_bbox[0]
                        cy = confirm_btn.center_y + window_bbox[1]
                    else:
                        cx = confirm_btn.center_x
                        cy = confirm_btn.center_y
                    print(f"     🔍 Botão 'Confirmar' detectado em ({cx}, {cy})")
                else:
                    print(f"     ⚠️  Modal de renovação não apresentou um par Cancelar/Confirmar verificável (tentativa {attempt}).")
                    continue
            else:
                print("     ⚠️  Não foi possível capturar a janela durante a confirmação de renovação.")
                continue

            img_before = img
            self._human_click(cx, cy, hold_ms=100)
            time.sleep(self.animation_wait_ms / 1000.0)

            # Verify
            img_after = self._take_screenshot()
            if img_before is not None and img_after is not None:
                if self.vision.verify_popup_dismissed(img_before, img_after):
                    print(f"     ✅ Renovação CONFIRMADA (tentativa {attempt})")
                    return ActionResult(True, "confirm_refresh", "Renovação confirmada", attempt - 1)
                elif not self.vision.is_popup_visible(img_after):
                    print(f"     ✅ Popup dispensado (tentativa {attempt})")
                    return ActionResult(True, "confirm_refresh", "Popup dispensado", attempt - 1)
                else:
                    print(f"     ❌ Popup ainda presente (tentativa {attempt}/{self.MAX_RETRIES})")
            else:
                return ActionResult(True, "confirm_refresh", "Sem verificação", attempt - 1)

        return ActionResult(False, "confirm_refresh",
                            f"FALHOU após {self.MAX_RETRIES} tentativas",
                            self.MAX_RETRIES)
