#!/usr/bin/env python3
"""
Epic Seven Secret Shop Bot — Main Execution Loop.

Orchestrates:
  1. Visual Detection via Dynamic Mobile Shop Frame (slots.py)
  2. Multi-scale Template Coin Matching (detector.py)
  3. Price Extraction & Bounds Verification (ocr.py & buyer.py)
  4. Purchase Execution & Confirmation Modal Handling (buyer.py)
  5. Scroll & Slot 6 Detection (scanner.py)
  6. Shop Refresh & Skystone Confirmation (buyer.py)
  7. Fail-safe Safety Monitoring & Graceful Shutdown (safety.py)

Usage:
  python main.py --calibrate              # Takes snapshot, tests anchor, saves debug_layout.png
  python main.py --dry-run                # Runs full loop in observation mode (no clicks)
  python main.py --live --max-refreshes 50 # Runs live auto-buy for up to 50 refreshes
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml
import pyautogui

from app.buyer import ShopBuyer, ActionResult
from app.capture import ScreenshotCapture
from app.detector import (
    CoinDetector,
    COIN_AMIZADE,
    COIN_MARCA_PAGINAS,
    COIN_MEDALHAS,
)
from app.safety import SafetyMonitor
from app.scanner import ShopScanner
from app.slots import ShopLayout
from app.templates import TemplateManager


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"[!] Arquivo de configuração '{config_path}' não encontrado. Usando padrões.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ShopBot:
    """Orchestrator for the automated Epic Seven Secret Shop loop."""

    def __init__(self, config: dict, mode: str = "dry-run", max_refreshes: int = 50):
        self.config = config
        self.mode = mode
        self.is_dry_run = (mode == "dry-run")
        self.max_refreshes = max_refreshes

        # Core components
        self.capture = ScreenshotCapture(
            window_title_keyword=config.get("WINDOW_TITLE_KEYWORD", "Epic Seven")
        )
        self.templates = TemplateManager(config.get("TEMPLATES_DIR", "templates"))
        self.detector = CoinDetector(
            template_manager=self.templates,
            confidence_threshold=config.get("CONFIDENCE_THRESHOLD", 0.85),
        )
        self.layout = ShopLayout(templates_dir=Path(config.get("TEMPLATES_DIR", "templates")))
        self.safety = SafetyMonitor(
            capture=self.capture,
            window_title=config.get("WINDOW_TITLE_KEYWORD", "Epic Seven"),
            resolution_tolerance=config.get("RESOLUTION_TOLERANCE_PX", 10),
        )
        self.buyer = ShopBuyer(config=self.config, dry_run=self.is_dry_run)
        self.buyer.set_capture(self.capture)  # Inject capture for verification screenshots
        if config.get("VISUAL_AI", {}).get("ENABLED", False):
            if self.buyer.actions.backend_error:
                print(f"[!] VISUAL_AI avisou: {self.buyer.actions.backend_error}")
                print(f"    O modelo deve estar em '{config.get('VISUAL_AI', {}).get('MODEL_PATH')}'.")
                print(f"    Copie o modelo e labels para essa pasta, ou ponha VISUAL_AI.ENABLED: false.")
                print(f"    A deteção visual conservadora continua ativa.")
            else:
                print(f"[✓] VISUAL_AI ativo: backend {config.get('VISUAL_AI', {}).get('BACKEND')} carregado.")
        self.scanner = ShopScanner(
            capture=self.capture,
            detector=self.detector,
            safety=self.safety,
            layout=self.layout,
            scroll_amount=config.get("SCROLL_AMOUNT", 300),
            scroll_verify_wait_ms=config.get("SCROLL_VERIFY_WAIT_MS", 500),
        )

        # Statistics
        self.stats = {
            "refreshes": 0,
            "bookmarks_bought": 0,
            "mystics_bought": 0,
            "friendship_bought": 0,
            "gold_spent": 0,
            "skystones_spent": 0,
        }

    def calibrate(self):
        """Perform a test capture and save a visual layout alignment image."""
        print("=" * 65)
        print(" [CALIBRAÇÃO] Verificando Janela e Encaixe da Moldura Dinâmica")
        print("=" * 65)

        img = self.capture.screenshot()
        h, w = img.shape[:2]
        print(f"[*] Resolução capturada: {w}x{h} px")

        # Verify window detection
        win_detected = self.capture.is_window_detected()
        bbox = self.capture.get_window_bbox()
        print(f"[*] Janela Epic Seven detectada: {win_detected} (BBox: {bbox})")

        # Adjust layout
        self.layout.adjust_for_resolution(img)
        print(f"[*] Âncora visual encontrada: {self.layout.anchor_detected} em {self.layout.anchor_location}")

        if not self.layout.anchor_detected:
            print("[!] AVISO: Âncora visual não detectada. Certifique-se de que a Loja Secreta está aberta na tela.")

        # Detect visible slots
        results = self.detector.detect_visible_slots(img, self.layout)
        print(f"\n[*] {len(results)} slots analisados na tela:")
        for r in results:
            status = "DISPONÍVEL" if getattr(r, "disponivel", True) else "JÁ COMPRADO"
            print(f"    - Slot {r.slot}: {r.tipo.upper():<16} | Conf: {r.confianca:.2f} | Status: {status}")

        # Draw visual debug overlay
        vis = img.copy()
        if self.layout.anchor_detected:
            ax, ay = self.layout.anchor_location
            cv2.rectangle(vis, (ax, ay), (ax + 110, ay + 30), (0, 255, 255), 2)
            cv2.putText(vis, "Ancora Loja", (ax, ay - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for i, s in enumerate(self.layout.slots[:5]):
            cv2.rectangle(vis, (s.x0, s.y0), (s.x1, s.y1), (255, 255, 0), 1)
            if s.icon_box:
                cv2.rectangle(vis, (s.icon_box[0], s.icon_box[1]), (s.icon_box[2], s.icon_box[3]), (0, 255, 0), 2)
            if s.price_box:
                cv2.rectangle(vis, (s.price_box[0], s.price_box[1]), (s.price_box[2], s.price_box[3]), (0, 165, 255), 2)
            if s.button_box:
                cv2.rectangle(vis, (s.button_box[0], s.button_box[1]), (s.button_box[2], s.button_box[3]), (255, 255, 0), 2)

        output_file = "debug_layout.png"
        cv2.imwrite(output_file, vis)
        print(f"\n[✓] Imagem com a moldura salva em: '{output_file}'")
        print("=" * 65)

    def print_dashboard(self):
        """Display real-time statistics dashboard in console."""
        print("-" * 65)
        print(f" LOJA SECRETA BOT | Modo: {self.mode.upper()}")
        print(f" Renovações: {self.stats['refreshes']} / {self.max_refreshes} | Pedras Gastas: {self.stats['skystones_spent']}")
        print(f" Marca-Páginas: {self.stats['bookmarks_bought']} (x5) | Medalhas Místicas: {self.stats['mystics_bought']} (x50) | Amizade: {self.stats['friendship_bought']}")
        print(f" Ouro Estimado Gasto: {self.stats['gold_spent']:,} G")
        print("-" * 65)

    def _process_slot_buy(self, detection, slot_region, window_offset, window_bbox, img=None):
        """Execute buy flow for a target slot with full verification."""
        decision = self.buyer.evaluate_slot(detection)
        if not decision.should_buy:
            return False

        print(f"\n[🎯] OPORTUNIDADE: {decision.reason}")
        print(f"     -> Clicando no botão Comprar do {slot_region.name}...")

        # 1. Click slot buy button (with visual detection + popup verification)
        buy_result = self.buyer.click_buy_button(
            slot_region, window_offset=window_offset, img=img
        )
        if not buy_result.success:
            print(f"[❌] FALHA AO CLICAR: {buy_result.message}")
            print(f"     -> Pulando slot {detection.slot} e continuando...")
            return False

        # 2. Confirm modal popup (with visual button detection + dismissal verification)
        print("     -> Confirmando compra no popup...")
        confirm_result = self.buyer.confirm_purchase_popup(window_bbox=window_bbox)
        if not confirm_result.success:
            print(f"[❌] FALHA NA CONFIRMAÇÃO: {confirm_result.message}")
            print(f"     -> Pulando slot {detection.slot}. A compra NÃO foi concluída.")
            # Try to dismiss any lingering popup by pressing Escape
            pyautogui.press('escape')
            time.sleep(0.5)
            return False

        # 3. Post-purchase verification: check if the item is now marked as bought
        time.sleep(0.3)
        verify_img = self.buyer._take_screenshot()
        if verify_img is not None:
            item_purchased = self.buyer.vision.verify_item_purchased(
                verify_img,
                slot_region.y0, slot_region.y1,
                slot_region.x0, slot_region.x1,
            )
            if item_purchased:
                print(f"[✓✓] VERIFICADO: Item no {slot_region.name} foi comprado com sucesso!")
            else:
                print(f"[⚠️] AVISO: Não foi possível confirmar que o item foi comprado.")
                print(f"     O botão verde ainda parece ativo. A compra pode ter falhado.")

        # 4. Update statistics
        if decision.coin_type == COIN_MARCA_PAGINAS:
            self.stats["bookmarks_bought"] += 1
            self.stats["gold_spent"] += (decision.price or 184000)
        elif decision.coin_type == COIN_MEDALHAS:
            self.stats["mystics_bought"] += 1
            self.stats["gold_spent"] += (decision.price or 280000)
        elif decision.coin_type == COIN_AMIZADE:
            self.stats["friendship_bought"] += 1
            self.stats["gold_spent"] += (decision.price or 18000)

        print(f"[✓] Compra concluída: {decision.coin_type.upper()}")
        return True

    def run_loop(self):
        """Execute the automated shop loop."""
        print("=" * 65)
        print(f" INICIANDO BOT DA LOJA SECRETA ({self.mode.upper()})")
        print(" Pressione Ctrl+C a qualquer momento para parar.")
        print(" Fail-safe PyAutoGUI: arraste o mouse até o canto da tela para emergência.")
        print("=" * 65)

        try:
            while self.stats["refreshes"] < self.max_refreshes:
                # Ensure Epic Seven has active window focus
                self.capture.activate_window()

                # --- Step 1: Capture & Safety Verification ---
                img = self.capture.screenshot()
                self.layout.adjust_for_resolution(img)

                # Safety check
                safety_check = self.safety.check_shop_layout(img, layout=self.layout)
                if not safety_check.passed:
                    print(f"[!] SEGURANÇA: {safety_check.message}. Pausando...")
                    time.sleep(2.0)
                    continue

                window_offset = self.capture.get_window_offset()
                window_bbox = self.capture.get_window_bbox()

                # --- Step 2: Scan & Buy Slots 1 to 5 ---
                visible_detections = self.detector.detect_visible_slots(img, self.layout)
                for i, detection in enumerate(visible_detections):
                    if getattr(detection, "disponivel", True) and detection.tipo in [COIN_MARCA_PAGINAS, COIN_MEDALHAS, COIN_AMIZADE]:
                        self._process_slot_buy(
                            detection=detection,
                            slot_region=self.layout.slots[i],
                            window_offset=window_offset,
                            window_bbox=window_bbox,
                            img=img,
                        )

                # --- Step 3: Scroll & Check Slot 6 ---
                if not self.is_dry_run:
                    # In mobile games and emulators, touch drag (swipe) is required to scroll.
                    # Drag from bottom of shop list upwards:
                    frame = self.layout.slots[0]
                    swipe_x = window_offset[0] + frame.x0 + (frame.x1 - frame.x0) // 2
                    swipe_y_start = window_offset[1] + self.layout.slots[3].y1 - 10
                    swipe_y_end = window_offset[1] + self.layout.slots[0].y0 + 20

                    print("     -> Rolando a loja (Touch Swipe)...")
                    self.buyer.touch_swipe_scroll(swipe_x, swipe_y_start, swipe_y_end)
                    time.sleep(self.config.get("SCROLL_VERIFY_WAIT_MS", 500) / 1000.0)

                    # Capture scrolled frame and detect slot 6 (visible in row 4 position after scroll)
                    scrolled_img = self.capture.screenshot()
                    slot_6_detection = self.detector.detect_slot(scrolled_img, self.layout.slots[3], 6)
                    if getattr(slot_6_detection, "disponivel", True) and slot_6_detection.tipo in [COIN_MARCA_PAGINAS, COIN_MEDALHAS, COIN_AMIZADE]:
                        self._process_slot_buy(
                            detection=slot_6_detection,
                            slot_region=self.layout.slots[3],
                            window_offset=window_offset,
                            window_bbox=window_bbox,
                            img=scrolled_img,
                        )

                # --- Step 3.5: Anti-Loss Safeguard Lock ---
                # Double-check all visible slots before refreshing. If ANY target coin is found,
                # BUY IT IMMEDIATELY and DO NOT REFRESH until it is secured!
                for idx, slot_det in enumerate(visible_detections):
                    if getattr(slot_det, "disponivel", True) and slot_det.tipo in [COIN_MARCA_PAGINAS, COIN_MEDALHAS, COIN_AMIZADE]:
                        print(f"\n[🔒 TRAVA DE SEGURANÇA] Item pendente detectado no Slot {slot_det.slot}: {slot_det.tipo.upper()}!")
                        self._process_slot_buy(
                            detection=slot_det,
                            slot_region=self.layout.slots[idx],
                            window_offset=window_offset,
                            window_bbox=window_bbox,
                            img=img,
                        )

                # --- Step 4: Refresh Shop ---
                print("\n[*] Atualizando Loja (Renovar)...")
                refresh_result = self.buyer.click_refresh_button(self.layout, window_offset=window_offset)
                if not refresh_result.success:
                    print(f"[⚠️] RENOVAÇÃO NÃO EXECUTADA: {refresh_result.message}")
                    print("     Nenhuma Pedra Celeste foi contabilizada; a tela será verificada novamente.")
                    time.sleep(self.config.get("ANIMATION_WAIT_MS", 800) / 1000.0)
                    continue

                confirm_result = self.buyer.confirm_refresh_popup(window_bbox=window_bbox, layout=self.layout)
                if not confirm_result.success:
                    print(f"[⚠️] RENOVAÇÃO NÃO CONFIRMADA: {confirm_result.message}")
                    print("     Nenhuma Pedra Celeste foi contabilizada; a tela será verificada novamente.")
                    time.sleep(self.config.get("ANIMATION_WAIT_MS", 800) / 1000.0)
                    continue

                self.stats["refreshes"] += 1
                self.stats["skystones_spent"] += 3

                self.print_dashboard()
                time.sleep(self.config.get("ANIMATION_WAIT_MS", 800) / 1000.0)

            print("\n[✓] Meta de renovações atingida! Bot finalizado com segurança.")
            self.print_dashboard()

        except KeyboardInterrupt:
            print("\n[!] Bot interrompido pelo usuário (Ctrl+C).")
            self.print_dashboard()
        except pyautogui.FailSafeException:
            print("\n[!] FAIL-SAFE ATIVADO: Mouse no canto da tela. Parada de emergência executada.")
            self.print_dashboard()
        except Exception as e:
            print(f"\n[!] Erro inesperado no loop: {e}")
            self.print_dashboard()


def main():
    parser = argparse.ArgumentParser(description="Epic Seven Secret Shop Bot")
    parser.add_argument("--mode", choices=["dry-run", "live", "calibrate"], default="dry-run",
                        help="Modo de execução: 'calibrate' para testar moldura, 'dry-run' para simular, 'live' para comprar de verdade")
    parser.add_argument("--dry-run", action="store_true", help="Atalho para modo dry-run")
    parser.add_argument("--live", action="store_true", help="Atalho para modo live (compra real)")
    parser.add_argument("--calibrate", action="store_true", help="Atalho para modo de calibração")
    parser.add_argument("--max-refreshes", type=int, default=50, help="Limite máximo de renovações de loja")
    parser.add_argument("--config", default="config.yaml", help="Caminho do arquivo config.yaml")

    args = parser.parse_args()

    # Determine mode
    mode = args.mode
    if args.calibrate:
        mode = "calibrate"
    elif args.live:
        mode = "live"
    elif args.dry_run:
        mode = "dry-run"

    config = load_config(args.config)
    bot = ShopBot(config=config, mode=mode, max_refreshes=args.max_refreshes)

    if mode == "calibrate":
        bot.calibrate()
    else:
        bot.run_loop()


if __name__ == "__main__":
    main()
