"""Adapters: todos falam a mesma lingua -> {slot,currency,price,available,confidence}.
Nao move mouse, nao clica.
"""
from __future__ import annotations
import time, re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

def load_bgr(path: str|Path):
    p=Path(path)
    if not p.is_absolute():
        p=ROOT/p
    pil=Image.open(p).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def normalize_currency(s):
    from benchmark.src.dataset import normalize_currency as nc
    return nc(s)

@dataclass
class PredSlot:
    slot: int
    currency: str  # friendship|bookmarks|mystic_medals|ignorar|unknown
    price: Optional[int]
    available: Optional[bool]
    confidence: float

# ---------- Baseline OpenCV ----------
class OpenCVAdapter:
    name="opencv_template"
    group="baseline"
    def __init__(self):
        from app.templates import TemplateManager
        from app.detector import CoinDetector
        self.tm=TemplateManager(ROOT/"templates")
        self.det=CoinDetector(self.tm, confidence_threshold=0.85)
    def load(self): t0=time.perf_counter(); return time.perf_counter()-t0
    def infer(self, image_path: str|Path, slot_index: int=1):
        img=load_bgr(image_path)
        h,w=img.shape[:2]
        # crop test: arquivo ja eh crop -> usar slot fake cobrindo tudo
        if "165924" not in str(image_path):
            # crop: vender como slot unico
            from app.slots import SlotRegion
            reg=SlotRegion(name="crop",slot_index=slot_index,x0=0,y0=0,x1=w,y1=h, icon_box=(max(0,int(w*0.05)),0,min(w,int(w*0.35)),h), price_box=(int(w*0.55),0,w,int(h*0.45)), button_box=(int(w*0.55),int(h*0.45),w,h))
            det=self.det.detect_slot(img,reg,slot_index)
            cur=normalize_currency(det.tipo)
            return PredSlot(slot=slot_index,currency=cur,price=det.preco,available=det.disponivel,confidence=float(det.confianca))
        else:
            from app.slots import ShopLayout
            layout=ShopLayout(templates_dir=ROOT/"templates")
            layout.adjust_for_resolution(img)
            reg=layout.slots[slot_index-1]
            det=self.det.detect_slot(img,reg,slot_index)
            cur=normalize_currency(det.tipo)
            return PredSlot(slot=slot_index,currency=cur,price=det.preco,available=det.disponivel,confidence=float(det.confianca))

# ---------- Tesseract OCR adapter via pytesseract ----------
import os
_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/c/Program Files/Tesseract-OCR/tesseract.exe",
]
def _find_tesseract():
    import shutil
    exe = shutil.which("tesseract")
    if exe:
        return exe
    for c in _TESSERACT_CANDIDATES:
        if os.path.isfile(c):
            return c
    return None

class TesseractAdapter:
    name="tesseract_price_only"
    group="ocr"
    def __init__(self):
        self._exe = _find_tesseract()
    def load(self):
        try:
            import pytesseract
            if self._exe:
                pytesseract.pytesseract.tesseract_cmd = self._exe
            # test: does tesseract respond?
            pytesseract.get_tesseract_version()
            return 0.0
        except Exception as e:
            raise RuntimeError(f"Tesseract not usable: {e}")
    def infer_price(self, image_path: str|Path):
        """Read price from image ROI via pytesseract. Returns (price:int|None, confidence:float).
        ponytail: upscale 3x + PSM6 + regex. Upgrade: tunning por resolucao/cor do slot.
        """
        try:
            import pytesseract, cv2
            if self._exe:
                pytesseract.pytesseract.tesseract_cmd = self._exe
            img = load_bgr(image_path)
            h, w = img.shape[:2]
            # upscale 3x for small crops (80px high)
            big = cv2.resize(img, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            # PSM 6: uniform block of text; no whitelist — allows full parsing
            raw = pytesseract.image_to_string(gray, config="--psm 6")
            # extract number with thousand separator: 18.000 or 18,000
            import re
            m = re.search(r'(\d{1,3}(?:[.,]\d{3})+)', raw)
            if not m:
                return None, 0.0
            price = int(m.group(1).replace('.', '').replace(',', ''))
            conf = 0.85 if len(str(price)) >= 3 else 0.5
            return price, conf
        except Exception:
            return None, 0.0

# ---------- ONNX placeholder - avalia se onnxruntime + modelo existir ----------
class ONNXAdapter:
    name="yolo_onnx_action_detector"
    group="yolo"
    def __init__(self):
        self.available=False
        self.reason=""
        try:
            import onnxruntime
            self.ort_ver=onnxruntime.__version__
        except Exception as e:
            self.reason=str(e)
            return
        # procurar modelo
        cands=list((ROOT/"models"/"vision").glob("*.onnx"))
        if not cands:
            self.reason="nenhum .onnx em models/vision"
            return
        self.model_path=cands[0]
        self.available=True
    def load(self):
        if not self.available: return 0
        import onnxruntime as ort
        t0=time.perf_counter()
        try:
            self.sess=ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        except Exception as e:
            self.reason=str(e); self.available=False
        return time.perf_counter()-t0
    def infer(self, image_path, slot_index=1):
        return PredSlot(slot=slot_index,currency="unknown",price=None,available=None,confidence=0)

# ---------- VLM adapters via python direct (no network) ----------
# Cada VLM: tenta carregar localmente via transformers. Se nao houver cache, marca FORA DA CATEGORIA / nao disponivel.
# Nao instala modelo sem verificar tamanho.

VLM_CANDIDATES = [
    # nome, hf_id, grupo nominal, tamanho FP16 MB, tamanho INT8, INT4, original formato
    ("smolvlm-256m", "HuggingFaceTB/SmolVLM-256M-Instruct", "C", 512, 256, 128, "FP16"),
    ("smolvlm-500m", "HuggingFaceTB/SmolVLM-500M-Instruct", "B/C", 1000, 500, 250, "FP16"),
    ("smolvlm2-256m", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", "C", 512, 256, 128, "FP16"),
    ("smolvlm2-500m", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "B/C", 1000, 500, 250, "FP16"),
    ("internvl2-1b", "OpenGVLab/InternVL2-1B", "C", 2000, 1000, 500, "FP16"),
    ("internvl2_5-1b", "OpenGVLab/InternVL2_5-1B", "C", 2000, 1000, 500, "FP16"),
    ("tinylava-1.5b", "bczhou/TinyLLaVA-3.1B", "FORA", 6200, 3100, 1550, "FP16"),
]

PROMPT = """Analyze this Epic Seven Secret Shop screenshot. Ignore heroes/equipment/artifacts.
Only consider currencies: friendship (Pontos de Amizade), bookmarks (Marca-Paginas da Alianca), mystic_medals (Medalhas Misticas).
For the visible slot(s), return JSON only:
{"slot":1,"currency":"friendship|bookmarks|mystic_medals|ignorar","price":18000,"available":true,"confidence":0.94}
If price not visible put null. If slot has no relevant currency put "ignorar". No explanation."""

def vlm_local_status(hf_id):
    # sem rede: apenas checa cache local. Sem huggingface_hub -> not_cached
    try:
        from huggingface_hub import scan_cache_dir
        cache=scan_cache_dir()
        for r in cache.repos:
            if hf_id.lower() in r.repo_id.lower():
                sz=sum(f.size_on_disk for f in r.revisions[0].files) if r.revisions else 0
                return "cached", sz
        return "not_cached", 0
    except ModuleNotFoundError:
        # huggingface_hub nao instalado -> nao da para verificar cache, tratar como not_cached
        return "not_cached", 0
    except Exception as e:
        return f"error:{e}", 0

class VLMAdapter:
    def __init__(self, name, hf_id, group):
        self.name=name; self.hf_id=hf_id; self.group=group
        self.loaded=False; self.load_time=0; self.size_mb=0
    def try_load(self):
        import time
        t0=time.perf_counter()
        status, sz = vlm_local_status(self.hf_id)
        self.load_time=time.perf_counter()-t0
        if status=="cached":
            self.size_mb=round(sz/1024/1024,1)
            # tentar carregar realmente
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                # nao baixa da rede - local_files_only
                self.processor=AutoProcessor.from_pretrained(self.hf_id, local_files_only=True, trust_remote_code=True)
                self.model=AutoModelForVision2Seq.from_pretrained(self.hf_id, local_files_only=True, trust_remote_code=True)
                self.loaded=True
                return True
            except Exception as e:
                self.load_error=str(e); return False
        else:
            self.load_error=status; return False
    def infer(self, image_path, slot_index=1):
        if not self.loaded:
            return PredSlot(slot=slot_index,currency="unknown",price=None,available=None,confidence=0)
        # inferencia real (placeholder: nao executado sem cache)
        return PredSlot(slot=slot_index,currency="unknown",price=None,available=None,confidence=0)
