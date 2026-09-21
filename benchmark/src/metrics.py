from __future__ import annotations
from dataclasses import dataclass
import statistics

def precision_recall_f1(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp>0 else 0
    r=tp/(tp+fn) if tp+fn>0 else 0
    f1=2*p*r/(p+r) if p+r>0 else 0
    return p,r,f1

def epic_shop_accuracy(slots):
    """slots: list de dict {gt,pred}. moeda+preco+disponibilidade+slot corretos = 1. UNKNOWN nunca conta como acerto."""
    if not slots: return 0
    ok=0
    for s in slots:
        pred=s["pred"]; gt=s["gt"]
        if pred.currency=="unknown" or pred.currency=="UNKNOWN":
            continue
        moeda = pred.currency==gt.currency
        preco = pred.price==gt.price
        disp = pred.available==gt.available
        slot_ok = pred.slot==gt.slot
        if moeda and preco and disp and slot_ok:
            ok+=1
    return ok/len(slots)

def price_stats(slots):
    exact=0; incorreto=0; vazio=0; invalido=0
    for s in slots:
        pred=s["pred"]; gt=s["gt"]
        if pred.price is None: vazio+=1
        elif gt.price is None: invalido+=1
        elif pred.price==gt.price: exact+=1
        else: incorreto+=1
    return {"exato":exact,"incorreto":incorreto,"vazio":vazio,"invalido":invalido,
            "acc": exact/max(1,len(slots))}

def availability_stats(slots):
    ok=0
    for s in slots:
        if s["pred"].available==s["gt"].available: ok+=1
    return {"acc": ok/max(1,len(slots)), "corretos":ok,"total":len(slots)}

def latency_stats(times):
    if not times: return {}
    s=sorted(times)
    def pct(p): 
        import math
        k=(len(s)-1)*p/100; f=math.floor(k); c=math.ceil(k)
        if f==c: return s[int(k)]
        return s[f]*(c-k)+s[c]*(k-f)
    return {"mean":statistics.mean(s),"p50":pct(50),"p95":pct(95),"min":min(s),"max":max(s),"ips": 1/statistics.mean(s) if statistics.mean(s)>0 else 0}

def confusion_by_currency(slots, currencies=("friendship","bookmarks","mystic_medals","ignorar")):
    out={}
    for cur in currencies:
        tp=sum(1 for s in slots if s["gt"].currency==cur and s["pred"].currency==cur)
        fp=sum(1 for s in slots if s["gt"].currency!=cur and s["pred"].currency==cur)
        fn=sum(1 for s in slots if s["gt"].currency==cur and s["pred"].currency!=cur)
        p,r,f1=precision_recall_f1(tp,fp,fn)
        out[cur]={"tp":tp,"fp":fp,"fn":fn,"precision":round(p,3),"recall":round(r,3),"f1":round(f1,3)}
    return out
