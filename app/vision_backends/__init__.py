"""Pluggable local vision backends used to identify game actions."""

from .contracts import ActionClass, ActionCandidate, VisionBackend
from .yolo_onnx import YoloOnnxBackend

__all__ = ["ActionClass", "ActionCandidate", "VisionBackend", "YoloOnnxBackend"]
