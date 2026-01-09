"""
Trakt - TensorFlow OCR for ONVIF Cameras

A TensorFlow-based application for performing OCR on video streams
from ONVIF-compliant IP cameras.
"""

__version__ = "1.0.0"
__author__ = "albertmwangwa"

from .camera_handler import ONVIFCameraHandler
from .ocr_engine import OCREngine, TensorFlowOCRModel
from .web_api import app, run_server, update_state

__all__ = [
    "ONVIFCameraHandler",
    "OCREngine",
    "TensorFlowOCRModel",
    "app",
    "run_server",
    "update_state",
]
