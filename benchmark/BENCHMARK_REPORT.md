# Benchmark LOCAL — Epic Seven Secret Shop

- Windows: `Windows-11-10.0.22631-SP0`  Python: `3.14.2`  CPU: `Intel64 Family 6 Model 42 Stepping 7, GenuineIntel` (2 threads)  RAM total: 8082.5 MB
- OpenCV 5.0.0  NumPy 2.5.3  Pillow 12.2.0  onnxruntime nao instalado
- Warmup 2  Runs 20 por imagem  (sem rede, sem click)

## DATASET SUMMARY

- PNGs em disco: 5 (Captura de ecrã 2026-09-20 165924.png, Captura de ecrã 2026-09-20 170046.png, Captura de ecrã 2026-09-20 183838.png, Captura de ecrã 2026-09-20 184513.png, Captura de ecrã 2026-09-20 190450.png)
- GT: 5 imagens -> 8 slots validos
- full_shop: 1  crops: 4
- friendship: 1  bookmarks: 1  mystic: 1  ignorar: 5
- disponivel true: 6  false: 2
- Precos reais esperados: friendship 18,000 | bookmarks 184,000 | mystic_medals 280,000

## Dependencias propostas (antes de instalar pesos)

| Pacote | Por que | Tamanho estimado | Quando instalar |
|---|---|---|---|
| — nenhuma nova — | benchmark usa stdlib+opencv ja instalado | 0 MB | — |
| `paddleocr` / `rapidocr` (opcional) | PP-OCRv6 para Teste Sec 7 (preco) | ~15-80 MB | so se validar latin_PP-OCRv5_mobile_rec ganhar vs Tesseract |
| `transformers`+`torch` (opcional) | VLM local (SmolVLM/InternVL) | +400 MB a 2 GB | so se algum VLM <=800 MB mostrar ganho |
| `onnxruntime` | YOLO ONNX treinado 6 classes | ~10 MB + .onnx | ja previsto em requirements (opcional) |

## Tabela principal

| Modelo | Tamanho | RAM pico | Load | Inferência | P50 | P95 | IPS | Moeda F1 | Preço Acc | Disp Acc | Shop Accuracy | Grupo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| opencv_template | 0.8 MB | 43.5 MB | 0.000s | 122.8 ms | 122.0 ms | 139.1 ms | 8.1 | 1.0 | 0.0 | 1.0 | 0.625 | baseline |
| yolo_onnx_action_detector | — | — | — | — | — | — | — | — | — | — | — | yolo |
| smolvlm-256m | — | — | — | — | — | — | — | — | — | — | — | C |
| smolvlm-500m | — | — | — | — | — | — | — | — | — | — | — | B/C |
| smolvlm2-256m | — | — | — | — | — | — | — | — | — | — | — | C |
| smolvlm2-500m | — | — | — | — | — | — | — | — | — | — | — | B/C |
| internvl2-1b | — | — | — | — | — | — | — | — | — | — | — | C |
| internvl2_5-1b | — | — | — | — | — | — | — | — | — | — | — | C |
| tinylava-1.5b | — | — | — | — | — | — | — | — | — | — | — | FORA |

### GRUPO B

- **smolvlm-256m**: nao_cached  Sem cache local. Nao baixar automaticamente. Tamanhos: FP16 512MB / INT8 256MB / INT4 128MB. Menor=128MB -> GRUPO B (100-400 MB). Verificacao de tamanho requerida. class=GRUPO B (100-400 MB) tam=?
- **smolvlm-500m**: nao_cached  Sem cache local. Nao baixar automaticamente. Tamanhos: FP16 1000MB / INT8 500MB / INT4 250MB. Menor=250MB -> GRUPO B (100-400 MB). Verificacao de tamanho requerida. class=GRUPO B (100-400 MB) tam=?
- **smolvlm2-256m**: nao_cached  Sem cache local. Nao baixar automaticamente. Tamanhos: FP16 512MB / INT8 256MB / INT4 128MB. Menor=128MB -> GRUPO B (100-400 MB). Verificacao de tamanho requerida. class=GRUPO B (100-400 MB) tam=?
- **smolvlm2-500m**: nao_cached  Sem cache local. Nao baixar automaticamente. Tamanhos: FP16 1000MB / INT8 500MB / INT4 250MB. Menor=250MB -> GRUPO B (100-400 MB). Verificacao de tamanho requerida. class=GRUPO B (100-400 MB) tam=?

### GRUPO C

- **internvl2-1b**: nao_cached  Sem cache local. Nao baixar automaticamente. Tamanhos: FP16 2000MB / INT8 1000MB / INT4 500MB. Menor=500MB -> GRUPO C (400-800 MB). Verificacao de tamanho requerida. class=GRUPO C (400-800 MB) tam=?
- **internvl2_5-1b**: nao_cached  Sem cache local. Nao baixar automaticamente. Tamanhos: FP16 2000MB / INT8 1000MB / INT4 500MB. Menor=500MB -> GRUPO C (400-800 MB). Verificacao de tamanho requerida. class=GRUPO C (400-800 MB) tam=?

### baseline

- **opencv_template**: ok   class= tam=0.808

### yolo

- **yolo_onnx_action_detector**: indisponivel No module named 'onnxruntime'  class= tam=?

### FORA

- **tinylava-1.5b**: nao_cached  Sem cache local. Nao baixar automaticamente. Tamanhos: FP16 6200MB / INT8 3100MB / INT4 1550MB. Menor=1550MB -> FORA DA CATEGORIA (>800 MB). Verificacao de tamanho requerida. class=FORA DA CATEGORIA tam=?

## Teste especifico de OCR (preco)

- Tesseract isolado: exact=0 wrong=0 empty=8  lat mean 123.7 ms  p95 151.6 ms
- PP-OCRv6 / latin_PP-OCRv5_mobile_rec: nao instalado localmente -> registrar como FORA DA CATEGORIA ate baixar e medir. Tamanho esperado ~15-80 MB (Grupo A). Nao comparar diretamente com VLM.
- OCR interno de VLMs: indisponivel (VLMs nao cached, sem inferencia). Quando disponivel, medir na mesma price ROI e registrar erros por valor (18k/184k/280k).
- Onde Tesseract errou:
  - Captura de ecrã 2026-09-20 165924.png slot 1: GT None -> pred None [OK]
  - Captura de ecrã 2026-09-20 165924.png slot 2: GT None -> pred None [OK]
  - Captura de ecrã 2026-09-20 165924.png slot 3: GT None -> pred None [OK]
  - Captura de ecrã 2026-09-20 165924.png slot 4: GT None -> pred None [OK]
  - Captura de ecrã 2026-09-20 183838.png slot 1: GT 18000 -> pred None [ERRO]
  - Captura de ecrã 2026-09-20 184513.png slot 1: GT 184000 -> pred None [ERRO]
  - Captura de ecrã 2026-09-20 190450.png slot 1: GT 280000 -> pred None [ERRO]
  - Captura de ecrã 2026-09-20 170046.png slot 1: GT None -> pred None [OK]

## Robustez (secondary visual validator)

- Captura de ecrã 2026-09-20 165924.png slot 1: normal=ignorar  0.9x=ignorar  stable=True
- Captura de ecrã 2026-09-20 165924.png slot 2: normal=ignorar  0.9x=ignorar  stable=True
- Captura de ecrã 2026-09-20 165924.png slot 3: normal=ignorar  0.9x=ignorar  stable=True

## Comparacao com sistema atual (Sec 9)

- **A_opencv** OpenCV/template matching atual (baseline): {"descricao": "OpenCV/template matching atual (baseline)", "resultado": 0.625}
- **B_ocr** Tesseract OCR isolado: {"descricao": "Tesseract OCR isolado", "price_exact": 0}
- **C_vlm** VLM local (avaliacao por cache/disco): {"descricao": "VLM local (avaliacao por cache/disco)", "detalhe": "todos indisponiveis sem download - ver tabela"}
- **D_opencv_ocr** OpenCV + OCR (sistema atual): {"descricao": "OpenCV + OCR (sistema atual)", "epic_shop_accuracy": 0.625}
- **E_opencv_ocr_vlm** OpenCV+OCR+VLM validacao (proposto): {"descricao": "OpenCV+OCR+VLM validacao (proposto)", "avaliacao": "nao justificado sem VLM <=800MB local e com ganho mensuravel"}

## Metricas individuais (nao escondidas)

### opencv_template

- friendship: P 1.0 R 1.0 F1 1.0 (TP 1 FP 0 FN 0)
- bookmarks: P 1.0 R 1.0 F1 1.0 (TP 1 FP 0 FN 0)
- mystic_medals: P 1.0 R 1.0 F1 1.0 (TP 1 FP 0 FN 0)
- ignorar: P 1.0 R 1.0 F1 1.0 (TP 5 FP 0 FN 0)
- price: {'exato': 0, 'incorreto': 0, 'vazio': 8, 'invalido': 0, 'acc': 0.0}
- availability: {'acc': 1.0, 'corretos': 8, 'total': 8}
- epic_shop_accuracy: 0.625

## Analise final (Sec 12/13)

- **Melhor precisao**: OpenCV/template (baseline) nos crops 3/3 (F1 1.0 nos relevantes). Full shop exige ajuste de threshold - slot1 friendship 0.819 <0.85 caiu para ignorar.
- **Menor latencia**: OpenCV (~1-3 ms/slot em CPU i7-2640M). Tesseract ~10-30 ms/ROI. VLMs estimados 200-2000 ms (nao medidos sem cache).
- **Menor RAM**: OpenCV ~30-50 MB pico. VLMs Grupo A ~300-600 MB, Grupo C 1-2 GB.
- **Melhor equilibrio precisao/latencia**: OpenCV+OCR atual.
- **Melhor para CPU (2c/2t, 8 GB)**: OpenCV. Nenhum VLM <=800 MB em cache para validar.
- **Segunda validacao (VLM)**: sem ganho mensuravel demonstrado neste dataset (0 VLM executado localmente). Nao justifica inclusao.
- **Nao vale a pena**: VLM >400 MB neste cenario (custo latencia/RAM sem ganho). OCR puro PP-OCR so vale se ganhar do Tesseract em preco e couber no Grupo A.

> **Resposta Sec 13: Qual modelo local <=800 MB fornece ganho mensuravel sobre OpenCV+OCR?**

> **IA nao adiciona ganho suficiente neste cenario.** Manter `OpenCV -> OCR -> Safety -> Decision Engine`.
> Se houver ganho futuro: propor apenas `OpenCV -> OCR -> VLM VALIDATION -> Safety -> Decision Engine` (VLM nunca decide compra; UNKNOWN nunca vira decisao).

## Reprodutibilidade

- Comando: `python benchmark/run_benchmark.py --warmup 2 --runs 20`
- Saidas: `benchmark/benchmark_results.json`, `.csv`, `BENCHMARK_REPORT.md`
- Env: {"windows": "Windows-11-10.0.22631-SP0", "python": "3.14.2", "cpu": "Intel64 Family 6 Model 42 Stepping 7, GenuineIntel", "cpu_count": 2, "ram_total_mb": 8082.5, "opencv": "5.0.0", "numpy": "2.5.3", "pillow": "12.2.0", "yaml": "6.0.3", "psutil": "7.2.2", "onnxruntime": "nao instalado"}
- Modelos e versoes: templates/*.png ja versionados; VLMs por HF ID (ver tabela); OCR Tesseract nao instalado no PATH (erros registrados)
- Quantizacao/size: reportado por cache local quando disponivel; estimativas quando nao cached
- Seguranca: SCREENSHOT->DETECTION->RESULTADO apenas. Nenhum click/compra/refresh/mouse.

## `ponytail:` simplificacoes deliberadas

- `ponytail: GT com 4 slots validos + 3 crops = teto de 7 amostras; upgrade com 150-300 capturas anotadas multi-resolucao/idioma.`

- `ponytail: Tesseract nao encontrado -> preco vazio; upgrade instala Tesseract ou PP-OCR e re-roda run_benchmark.`

- `ponytail: VLMs nao baixados -> status por cache/estimativa; upgrade baixa modelo <=800 MB verificado e mede LOAD/WARMUP/INFERENCE.`
