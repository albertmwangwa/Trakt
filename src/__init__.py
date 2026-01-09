"""
Trakt - TensorFlow OCR for ONVIF Cameras

A TensorFlow-based application for performing OCR on video streams
from ONVIF-compliant IP cameras.
"""

__version__ = "1.0.0"
__author__ = "albertmwangwa"

# Lazy imports to avoid requiring all dependencies at import time
__all__ = [
    "ONVIFCameraHandler",
    "OCREngine",
    "TensorFlowOCRModel",
    "app",
    "run_server",
    "update_state",
    "DatabaseManager",
    "AlertManager",
    "MultiCameraManager",
]


def __getattr__(name):
    """Lazy import for module attributes."""
    if name == "ONVIFCameraHandler":
        from .camera_handler import ONVIFCameraHandler
        return ONVIFCameraHandler
    elif name == "OCREngine":
        from .ocr_engine import OCREngine
        return OCREngine
    elif name == "TensorFlowOCRModel":
        from .ocr_engine import TensorFlowOCRModel
        return TensorFlowOCRModel
    elif name == "app":
        from .web_api import app
        return app
    elif name == "run_server":
        from .web_api import run_server
        return run_server
    elif name == "update_state":
        from .web_api import update_state
        return update_state
    elif name == "DatabaseManager":
        from .database import DatabaseManager
        return DatabaseManager
    elif name == "AlertManager":
        from .alert_manager import AlertManager
        return AlertManager
    elif name == "MultiCameraManager":
        from .multi_camera_manager import MultiCameraManager
        return MultiCameraManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
