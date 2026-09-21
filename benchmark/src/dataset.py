from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parents[2]
GT_FILE = ROOT / "benchmark" / "ground_truth.json"

ALIASES = {
    "friendship": ["amizade","friendship","pontos de amizade","friendship points"],
    "bookmarks": ["marca_paginas","bookmarks","covenant","marca-paginas","marca paginas"],
    "mystic_medals": ["medalhas_misticas","mystic_medals","mystic","medalhas"],
    "ignorar": ["ignorar","ignore","none","null","desconhecido","unknown","heroi","equipamento","artefato"],
}

# normaliza label de moeda
_CURRENCY_MAP = {}
for k, vals in ALIASES.items():
    for v in vals:
        _CURRENCY_MAP[v.lower()] = k
_CURRENCY_MAP["friendship"]="friendship"
_CURRENCY_MAP["bookmarks"]="bookmarks"
_CURRENCY_MAP["mystic_medals"]="mystic_medals"
_CURRENCY_MAP["ignorar"]="ignorar"

RELEVANT = {"friendship","bookmarks","mystic_medals"}

def normalize_currency(s):
    if s is None: return "ignorar"
    s=str(s).strip().lower().replace("-","_").replace(" ","_")
    if s in _CURRENCY_MAP: return _CURRENCY_MAP[s]
    # substr match
    for k,vals in ALIASES.items():
        for v in vals:
            if v.lower() in s or s in v.lower():
                return k
    return "ignorar" if s in ("null","none","") else s

@dataclass
class GTSlot:
    file: str
    image_type: str  # full_shop | crop_slot
    slot: int
    currency: str | None  # normalized
    price: int | None
    available: bool | None

def load_ground_truth():
    data=json.loads(GT_FILE.read_text(encoding="utf-8"))
    slots=[]
    for img in data["images"]:
        f=img["file"]
        t=img.get("type","unknown")
        for s in img["slots"]:
            if s.get("currency") is None:  # excluir truncado
                continue
            cur=normalize_currency(s["currency"])
            slots.append(GTSlot(file=f, image_type=t, slot=s["slot"], currency=cur, price=s.get("price"), available=s.get("available")))
    return slots, data

def dataset_summary():
    slots,data=load_ground_truth()
    sdir=ROOT/"Screenshot"
    full=len([i for i in data["images"] if i.get("type")=="full_shop"])
    crops=len([i for i in data["images"] if i.get("type")=="crop_slot"])
    # count files on disk
    pngs=list(sdir.glob("*.png")) if sdir.exists() else []
    summary={
        "screenshots_completas_gt": full,
        "crops_slots_gt": crops,
        "pngs_em_disco": len(pngs),
        "gt_slots_validos": len(slots),
        "friendship": len([s for s in slots if s.currency=="friendship"]),
        "bookmarks": len([s for s in slots if s.currency=="bookmarks"]),
        "mystic_medals": len([s for s in slots if s.currency=="mystic_medals"]),
        "ignorar": len([s for s in slots if s.currency=="ignorar"]),
        "disponivel_true": len([s for s in slots if s.available is True]),
        "disponivel_false": len([s for s in slots if s.available is False]),
        "arquivos": [p.name for p in pngs],
    }
    return summary
