# Diretório do Modelo de IA Visual (YOLO ONNX)

Este diretório está configurado e pronto para receber um modelo local treinado para deteção de elementos de interface do Epic Seven.

## Estado da Integração
- O pipeline em `app/action_identifier.py` e `app/vision_backends/yolo_onnx.py` já está 100% implementado.
- Se nenhum modelo estiver presente, o bot utiliza automaticamente os detetores visuais calibrados baseados em template matching e análise cromática (HSV).

## Estrutura dos Ficheiros Esperados

Copie para este diretório:
1. `model.onnx` — Modelo exportado no formato ONNX.
2. `labels.txt` — Ficheiro de texto contendo as classes na ordem exata dos IDs do modelo (uma por linha).

### Classes Suportadas (`labels.txt`):
```text
refresh_button
buy_button
purchase_confirm_button
refresh_confirm_button
purchase_modal
refresh_modal
```

## Como Ativar no `config.yaml`
```yaml
VISUAL_AI:
  ENABLED: true
  BACKEND: "yolo_onnx"
  MODEL_PATH: "models/vision/model.onnx"
  LABELS_PATH: "models/vision/labels.txt"
  MIN_CONFIDENCE: 0.90
  INPUT_SIZE: 640
```

## Dependência
Caso queira ativar a inferência ONNX, instale:
```powershell
pip install onnxruntime
```
