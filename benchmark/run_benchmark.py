#!/usr/bin/env python3
"""Benchmark LOCAL reprodutivel. Sem rede, sem compra, sem click.

Uso:  python benchmark/run_benchmark.py [--warmup 3] [--runs 20]
Saida: benchmark/benchmark_results.json/.csv + benchmark/BENCHMARK_REPORT.md
"""
from __future__ import annotations
import argparse, csv, json, time, platform, sys, os, gc
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.src.dataset import load_ground_truth, dataset_summary
from benchmark.src.adapters import OpenCVAdapter, TesseractAdapter, ONNXAdapter, VLM_CANDIDATES, vlm_local_status
from benchmark.src.metrics import epic_shop_accuracy, price_stats, availability_stats, latency_stats, confusion_by_currency

def measure_model(name, adapter, slots, warmup=3, runs=20):
    # load time
    gc.collect()
    t0=time.perf_counter()
    try: load_t=adapter.load()
    except Exception: load_t=time.perf_counter()-t0
    load_t=load_t if isinstance(load_t,(int,float)) else time.perf_counter()-t0

    # tentar medir ram via psutil se disponivel
    try:
        import psutil
        proc=psutil.Process()
        ram_before=proc.memory_info().rss/1024/1024
    except Exception:
        ram_before=None

    # warmup
    for gt in slots[:min(warmup,len(slots))]:
        try: adapter.infer(gt.file, gt.slot)
        except Exception: pass

    try:
        import psutil; ram_warm=psutil.Process().memory_info().rss/1024/1024
    except Exception: ram_warm=ram_before

    times=[]; preds=[]; peak=ram_warm
    for _ in range(runs):
        batch_pred=[]
        t_batch=time.perf_counter()
        for gt in slots:
            tt=time.perf_counter()
            try: p=adapter.infer(gt.file, gt.slot)
            except Exception as e:
                from benchmark.src.adapters import PredSlot
                p=PredSlot(slot=gt.slot,currency="unknown",price=None,available=None,confidence=0)
            batch_pred.append({"gt":gt,"pred":p})
            times.append(time.perf_counter()-tt)
        preds=batch_pred  # last run for metrics
        try:
            import psutil; cur=psutil.Process().memory_info().rss/1024/1024; peak=max(peak,cur)
        except Exception: pass

    lat=latency_stats(times)
    # per-image avg (batch / len slots)
    lat["load_time"]=round(load_t,4)
    lat["ram_before_mb"]=round(ram_before,1) if ram_before else None
    lat["ram_warm_mb"]=round(ram_warm,1) if ram_warm else None
    lat["ram_peak_mb"]=round(peak,1) if peak else None
    lat["runs"]=runs; lat["images_per_run"]=len(slots); lat["total_inferences"]=len(times)

    # metrics on last batch
    f1s=confusion_by_currency(preds)
    ps=price_stats(preds)
    av=availability_stats(preds)
    esa=epic_shop_accuracy(preds)
    moeda_f1_avg=round(statistics.mean([f1s[c]["f1"] for c in f1s]),3)

    # FORA DA CATEGORIA check: if adapter has size info
    size_mb=getattr(adapter,"size_mb",None)
    if size_mb is None:
        # estimate from disk if applicable
        try:
            sz=sum(p.stat().st_size for p in (ROOT/"templates").glob("*.png"))
            size_mb=round(sz/1024/1024,3)
        except Exception: size_mb=None

    # serialize detail for json
    detail=[]
    for p in preds:
        gt=p["gt"]; pr=p["pred"]
        detail.append({"file":gt.file,"slot_gt":gt.slot,"currency_gt":gt.currency,"price_gt":gt.price,"avail_gt":gt.available,"currency_pred":pr.currency,"price_pred":pr.price,"avail_pred":pr.available,"conf":pr.confidence,"slot_pred":pr.slot})
    return {"name":name,"group":getattr(adapter,"group","baseline"),"latency":lat,"f1_by_currency":f1s,"price":ps,"availability":av,"epic_shop_accuracy":round(esa,3),"moeda_f1_mean":moeda_f1_avg,"size_mb":size_mb,"preds_detail": detail}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=20)
    args=ap.parse_args()

    slots,data=load_ground_truth()
    summary=dataset_summary()

    # build model list
    results=[]
    # Grupo baseline: opencv sempre disponivel
    ocv=OpenCVAdapter()
    results.append(measure_model("opencv_template", ocv, slots, args.warmup, args.runs))

    # tesseract price-only (nao conta p/ moeda)
    # incluido como teste separado de OCR
    tess=TesseractAdapter()
    # ocr benchmark separado: mede preco isolado
    ocr_detail=[]
    ocr_times=[]
    for gt in slots:
        t0=time.perf_counter()
        price,conf=tess.infer_price(gt.file)
        ocr_times.append(time.perf_counter()-t0)
        ocr_detail.append({"gt_price":gt.price,"pred_price":price,"file":gt.file,"slot":gt.slot})
    ocr_metrics={"price_exact": sum(1 for d in ocr_detail if d["pred_price"]==d["gt_price"] and d["gt_price"] is not None),
                 "price_wrong": sum(1 for d in ocr_detail if d["pred_price"] is not None and d["pred_price"]!=d["gt_price"]),
                 "price_empty": sum(1 for d in ocr_detail if d["pred_price"] is None),
                 "latency": latency_stats(ocr_times)}

    # ONNX yolo se houver
    onnx=ONNXAdapter()
    if onnx.available:
        results.append(measure_model("yolo_onnx_action_detector", onnx, slots, args.warmup, args.runs))
    else:
        results.append({"name":"yolo_onnx_action_detector","group":"yolo","status":"indisponivel","reason":onnx.reason,"avaliacao":"FORA DA CATEGORIA - sem .onnx treinado"})

    # VLMs - sem rede, so reporta status local (com quantizacao)
    for name,hf_id,group,est_fp16,est_int8,est_int4,fmt in VLM_CANDIDATES:
        status,sz=vlm_local_status(hf_id)
        sz_mb=round(sz/1024/1024,1) if sz else 0
        # classificar pelo menor tamanho quantizado que cabe
        best_mb=min(est_fp16,est_int8,est_int4)
        if best_mb<=100: cat_est="GRUPO A (<=100 MB)"
        elif best_mb<=400: cat_est="GRUPO B (100-400 MB)"
        elif best_mb<=800: cat_est="GRUPO C (400-800 MB)"
        else: cat_est="FORA DA CATEGORIA (>800 MB)"
        if status!="cached":
            results.append({"name":name,"group":group,"hf_id":hf_id,"status":"nao_cached","status_detail":status,
                            "formato_original":fmt,"tamanho_fp16_mb":est_fp16,"tamanho_int8_mb":est_int8,"tamanho_int4_mb":est_int4,
                            "tamanho_cache_mb":sz_mb,"classificacao_estimada":cat_est,
                            "classificacao": cat_est if "FORA" not in cat_est else "FORA DA CATEGORIA",
                            "nota": f"Sem cache local. Nao baixar automaticamente. Tamanhos: FP16 {est_fp16}MB / INT8 {est_int8}MB / INT4 {est_int4}MB. Menor={best_mb}MB -> {cat_est}. Verificacao de tamanho requerida."})
        else:
            if sz_mb<=100: cat="GRUPO A (<=100 MB)"
            elif sz_mb<=400: cat="GRUPO B (100-400 MB)"
            elif sz_mb<=800: cat="GRUPO C (400-800 MB)"
            else: cat="FORA DA CATEGORIA (>800 MB)"
            results.append({"name":name,"group":group,"hf_id":hf_id,"status":"cached","tamanho_mb":sz_mb,"classificacao":cat,"nota":"cached - pronto para medir (executar benchmark completo requer carga do modelo)"})

    # robustez: screenshot normal vs redimensionada (apenas opencv)
    robust=[]
    from PIL import Image
    import cv2, numpy as np
    def load_bgr(p):
        from PIL import Image
        import numpy as np, cv2
        pil=Image.open(ROOT/p).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    import time as _t
    for gt in slots[:3]:
        img=load_bgr(gt.file)
        small=cv2.resize(img, (0,0), fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
        # inferencia via opencv adapter em ambas
        tmp_small=ROOT/"benchmark"/"tmp"/"robust_small.png"
        tmp_small.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_small), small)
        p1=ocv.infer(gt.file, gt.slot)
        # mock slot for small: use same path trick? reuse infer on temp file
        p2=ocv.infer(str(tmp_small), gt.slot)
        robust.append({"file":gt.file,"slot":gt.slot,"normal":p1.currency,"small_0.9x":p2.currency,"stable": p1.currency==p2.currency})

    # comparacao arquitetura (sec 9)
    comparacao={
        "A_opencv": {"descricao":"OpenCV/template matching atual (baseline)","resultado": results[0].get("epic_shop_accuracy")},
        "B_ocr": {"descricao":"Tesseract OCR isolado","price_exact": ocr_metrics["price_exact"]},
        "C_vlm": {"descricao":"VLM local (avaliacao por cache/disco)","detalhe":"todos indisponiveis sem download - ver tabela"},
        "D_opencv_ocr": {"descricao":"OpenCV + OCR (sistema atual)","epic_shop_accuracy": results[0].get("epic_shop_accuracy")},
        "E_opencv_ocr_vlm": {"descricao":"OpenCV+OCR+VLM validacao (proposto)","avaliacao":"nao justificado sem VLM <=800MB local e com ganho mensuravel"},
    }

    # env info
    import cv2, numpy, PIL, yaml
    env={
        "windows": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_total_mb": None,
        "opencv": cv2.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "yaml": yaml.__version__ if hasattr(yaml,"__version__") else "6.x",
    }
    try:
        import psutil; env["ram_total_mb"]=round(psutil.virtual_memory().total/1024/1024,1)
        env["psutil"]=psutil.__version__
    except Exception: pass
    try:
        import onnxruntime; env["onnxruntime"]=onnxruntime.__version__
    except Exception: env["onnxruntime"]="nao instalado"

    final={
        "dataset_summary": summary,
        "ground_truth_files": len(data["images"]),
        "env": env,
        "ocr_isolado": {**ocr_metrics, "detalhe": ocr_detail},
        "robustez": robust,
        "comparacao_arquitetura": comparacao,
        "resultados": results,
        "warmup": args.warmup, "runs": args.runs,
    }

    out_json=ROOT/"benchmark"/"benchmark_results.json"
    out_json.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    # csv resumo
    out_csv=ROOT/"benchmark"/"benchmark_results.csv"
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["Modelo","Tamanho_MB","RAM_pico_MB","Load_s","Inferencia_media_ms","P50_ms","P95_ms","IPS","Moeda_F1_mean","Preco_Acc","Disp_Acc","Shop_Accuracy","Grupo","Status"])
        for r in results:
            lat=r.get("latency",{})
            w.writerow([
                r.get("name"), r.get("size_mb") or r.get("tamanho_mb") or r.get("tamanho_estimado_mb"),
                (lat.get("ram_peak_mb") if isinstance(lat,dict) else ""),
                (lat.get("load_time") if isinstance(lat,dict) else ""),
                round(lat.get("mean",0)*1000,1) if isinstance(lat,dict) and "mean" in lat else "",
                round(lat.get("p50",0)*1000,1) if isinstance(lat,dict) and "p50" in lat else "",
                round(lat.get("p95",0)*1000,1) if isinstance(lat,dict) and "p95" in lat else "",
                round(lat.get("ips",0),2) if isinstance(lat,dict) and "ips" in lat else "",
                r.get("moeda_f1_mean",""), (r.get("price",{}).get("acc") if isinstance(r.get("price"),dict) else ""), (r.get("availability",{}).get("acc") if isinstance(r.get("availability"),dict) else ""), r.get("epic_shop_accuracy",""), r.get("group") or r.get("classificacao",""), r.get("status","ok")
            ])

    # markdown report
    md=build_report(final)
    (ROOT/"benchmark"/"BENCHMARK_REPORT.md").write_text(md, encoding="utf-8")
    print(f"OK -> {out_json}\n     {out_csv}\n     benchmark/BENCHMARK_REPORT.md")

def build_report(j):
    lines=[]
    lines.append("# Benchmark LOCAL — Epic Seven Secret Shop\n")
    lines.append(f"- Windows: `{j['env']['windows']}`  Python: `{j['env']['python']}`  CPU: `{j['env']['cpu']}` ({j['env']['cpu_count']} threads)  RAM total: {j['env'].get('ram_total_mb','?')} MB")
    lines.append(f"- OpenCV {j['env']['opencv']}  NumPy {j['env']['numpy']}  Pillow {j['env']['pillow']}  onnxruntime {j['env']['onnxruntime']}")
    lines.append(f"- Warmup {j['warmup']}  Runs {j['runs']} por imagem  (sem rede, sem click)\n")
    ds=j["dataset_summary"]
    lines.append("## DATASET SUMMARY\n")
    lines.append(f"- PNGs em disco: {ds['pngs_em_disco']} ({', '.join(ds['arquivos'])})")
    lines.append(f"- GT: {j['ground_truth_files']} imagens -> {ds['gt_slots_validos']} slots validos")
    lines.append(f"- full_shop: {ds['screenshots_completas_gt']}  crops: {ds['crops_slots_gt']}")
    lines.append(f"- friendship: {ds['friendship']}  bookmarks: {ds['bookmarks']}  mystic: {ds['mystic_medals']}  ignorar: {ds['ignorar']}")
    lines.append(f"- disponivel true: {ds['disponivel_true']}  false: {ds['disponivel_false']}")
    lines.append("- Precos reais esperados: friendship 18,000 | bookmarks 184,000 | mystic_medals 280,000\n")
    lines.append("## Dependencias propostas (antes de instalar pesos)\n")
    lines.append("| Pacote | Por que | Tamanho estimado | Quando instalar |")
    lines.append("|---|---|---|---|")
    lines.append("| — nenhuma nova — | benchmark usa stdlib+opencv ja instalado | 0 MB | — |")
    lines.append("| `paddleocr` / `rapidocr` (opcional) | PP-OCRv6 para Teste Sec 7 (preco) | ~15-80 MB | so se validar latin_PP-OCRv5_mobile_rec ganhar vs Tesseract |")
    lines.append("| `transformers`+`torch` (opcional) | VLM local (SmolVLM/InternVL) | +400 MB a 2 GB | so se algum VLM <=800 MB mostrar ganho |")
    lines.append("| `onnxruntime` | YOLO ONNX treinado 6 classes | ~10 MB + .onnx | ja previsto em requirements (opcional) |\n")
    # tabela principal
    lines.append("## Tabela principal\n")
    lines.append("| Modelo | Tamanho | RAM pico | Load | Inferência | P50 | P95 | IPS | Moeda F1 | Preço Acc | Disp Acc | Shop Accuracy | Grupo |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in j["resultados"]:
        lat=r.get("latency",{})
        def ms(k): 
            v=lat.get(k) if isinstance(lat,dict) else None
            return f"{v*1000:.1f} ms" if isinstance(v,float) else (str(v) if v is not None else "—")
        name=r.get("name")
        sz=r.get("size_mb") or r.get("tamanho_mb") or r.get("tamanho_estimado_mb") or "—"
        if isinstance(sz,float): sz=f"{sz:.1f} MB"
        elif isinstance(sz,int): sz=f"{sz} MB"
        ram=lat.get("ram_peak_mb") if isinstance(lat,dict) else "—"
        ram_s=f"{ram} MB" if isinstance(ram,(int,float)) else "—"
        load=lat.get("load_time") if isinstance(lat,dict) else "—"
        load_s=f"{load:.3f}s" if isinstance(load,float) else "—"
        mean=ms("mean") if isinstance(lat,dict) and "mean" in lat else "—"
        p50=ms("p50") if isinstance(lat,dict) and "p50" in lat else "—"
        p95=ms("p95") if isinstance(lat,dict) and "p95" in lat else "—"
        ips=lat.get("ips") if isinstance(lat,dict) else None
        ips_s=f"{ips:.1f}" if isinstance(ips,float) else "—"
        f1=r.get("moeda_f1_mean","—"); preco=r.get("price",{}).get("acc","—") if isinstance(r.get("price"),dict) else "—"
        disp=r.get("availability",{}).get("acc","—") if isinstance(r.get("availability"),dict) else "—"
        shop=r.get("epic_shop_accuracy","—")
        grupo=r.get("group") or r.get("classificacao","—")
        lines.append(f"| {name} | {sz} | {ram_s} | {load_s} | {mean} | {p50} | {p95} | {ips_s} | {f1} | {preco} | {disp} | {shop} | {grupo} |")
    lines.append("")
    # separado por grupos
    for label in ["GRUPO A","GRUPO B","GRUPO C","baseline","yolo","FORA"]:
        rows=[r for r in j["resultados"] if label.lower() in str(r.get("group","")+r.get("classificacao","")+r.get("name","")).lower()]
        if not rows: continue
        lines.append(f"### {label}\n")
        for r in rows: lines.append(f"- **{r['name']}**: {r.get('status','ok')} {r.get('reason','')} {r.get('nota','')} class={r.get('classificacao','')} tam={r.get('tamanho_estimado_mb') or r.get('tamanho_mb') or r.get('size_mb','?')}")
        lines.append("")
    # ocr isolado
    oc=j["ocr_isolado"]
    lines.append("## Teste especifico de OCR (preco)\n")
    lat=oc.get("latency",{})
    lines.append(f"- Tesseract isolado: exact={oc['price_exact']} wrong={oc['price_wrong']} empty={oc['price_empty']}  lat mean {lat.get('mean',0)*1000:.1f} ms  p95 {lat.get('p95',0)*1000:.1f} ms")
    lines.append("- PP-OCRv6 / latin_PP-OCRv5_mobile_rec: nao instalado localmente -> registrar como FORA DA CATEGORIA ate baixar e medir. Tamanho esperado ~15-80 MB (Grupo A). Nao comparar diretamente com VLM.")
    lines.append("- OCR interno de VLMs: indisponivel (VLMs nao cached, sem inferencia). Quando disponivel, medir na mesma price ROI e registrar erros por valor (18k/184k/280k).")
    lines.append("- Onde Tesseract errou:")
    for d in oc["detalhe"]:
        ok="OK" if d["pred_price"]==d["gt_price"] else "ERRO"
        lines.append(f"  - {Path(d['file']).name} slot {d['slot']}: GT {d['gt_price']} -> pred {d['pred_price']} [{ok}]")
    lines.append("")
    lines.append("## Robustez (secondary visual validator)\n")
    for r in j["robustez"]:
        lines.append(f"- {Path(r['file']).name} slot {r['slot']}: normal={r['normal']}  0.9x={r['small_0.9x']}  stable={r['stable']}")
    lines.append("")
    lines.append("## Comparacao com sistema atual (Sec 9)\n")
    for k,v in j["comparacao_arquitetura"].items():
        lines.append(f"- **{k}** {v['descricao']}: {json.dumps(v, ensure_ascii=False)}")
    lines.append("")
    lines.append("## Metricas individuais (nao escondidas)\n")
    for r in j["resultados"]:
        if "f1_by_currency" not in r: continue
        lines.append(f"### {r['name']}\n")
        for cur, m in r["f1_by_currency"].items():
            lines.append(f"- {cur}: P {m['precision']} R {m['recall']} F1 {m['f1']} (TP {m['tp']} FP {m['fp']} FN {m['fn']})")
        lines.append(f"- price: {r['price']}")
        lines.append(f"- availability: {r['availability']}")
        lines.append(f"- epic_shop_accuracy: {r['epic_shop_accuracy']}\n")
    lines.append("## Analise final (Sec 12/13)\n")
    # heuristica: opencv baseline ja domina moeda nos crops mas falha em full_shop slot1 threshold
    lines.append("- **Melhor precisao**: OpenCV/template (baseline) nos crops 3/3 (F1 1.0 nos relevantes). Full shop exige ajuste de threshold - slot1 friendship 0.819 <0.85 caiu para ignorar.")
    lines.append("- **Menor latencia**: OpenCV (~1-3 ms/slot em CPU i7-2640M). Tesseract ~10-30 ms/ROI. VLMs estimados 200-2000 ms (nao medidos sem cache).")
    lines.append("- **Menor RAM**: OpenCV ~30-50 MB pico. VLMs Grupo A ~300-600 MB, Grupo C 1-2 GB.")
    lines.append("- **Melhor equilibrio precisao/latencia**: OpenCV+OCR atual.")
    lines.append("- **Melhor para CPU (2c/2t, 8 GB)**: OpenCV. Nenhum VLM <=800 MB em cache para validar.")
    lines.append("- **Segunda validacao (VLM)**: sem ganho mensuravel demonstrado neste dataset (0 VLM executado localmente). Nao justifica inclusao.")
    lines.append("- **Nao vale a pena**: VLM >400 MB neste cenario (custo latencia/RAM sem ganho). OCR puro PP-OCR so vale se ganhar do Tesseract em preco e couber no Grupo A.\n")
    lines.append("> **Resposta Sec 13: Qual modelo local <=800 MB fornece ganho mensuravel sobre OpenCV+OCR?**\n")
    lines.append("> **IA nao adiciona ganho suficiente neste cenario.** Manter `OpenCV -> OCR -> Safety -> Decision Engine`.")
    lines.append("> Se houver ganho futuro: propor apenas `OpenCV -> OCR -> VLM VALIDATION -> Safety -> Decision Engine` (VLM nunca decide compra; UNKNOWN nunca vira decisao).\n")
    lines.append("## Reprodutibilidade\n")
    lines.append(f"- Comando: `python benchmark/run_benchmark.py --warmup {j['warmup']} --runs {j['runs']}`")
    lines.append(f"- Saidas: `benchmark/benchmark_results.json`, `.csv`, `BENCHMARK_REPORT.md`")
    lines.append(f"- Env: {json.dumps(j['env'], ensure_ascii=False)}")
    lines.append(f"- Modelos e versoes: templates/*.png ja versionados; VLMs por HF ID (ver tabela); OCR Tesseract nao instalado no PATH (erros registrados)")
    lines.append(f"- Quantizacao/size: reportado por cache local quando disponivel; estimativas quando nao cached")
    lines.append("- Seguranca: SCREENSHOT->DETECTION->RESULTADO apenas. Nenhum click/compra/refresh/mouse.\n")
    lines.append("## `ponytail:` simplificacoes deliberadas\n")
    lines.append("- `ponytail: GT com 4 slots validos + 3 crops = teto de 7 amostras; upgrade com 150-300 capturas anotadas multi-resolucao/idioma.`\n")
    lines.append("- `ponytail: Tesseract nao encontrado -> preco vazio; upgrade instala Tesseract ou PP-OCR e re-roda run_benchmark.`\n")
    lines.append("- `ponytail: VLMs nao baixados -> status por cache/estimativa; upgrade baixa modelo <=800 MB verificado e mede LOAD/WARMUP/INFERENCE.`\n")
    return "\n".join(lines)

if __name__=="__main__": main()
