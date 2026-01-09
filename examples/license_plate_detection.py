"""
Example: License Plate Detection

This example demonstrates detecting license plates from camera feed.
"""

import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import re

import cv2

from src.camera_handler import ONVIFCameraHandler
from src.ocr_engine import OCREngine

logging.basicConfig(level=logging.INFO)

# License plate pattern (customize for your region)
LICENSE_PLATE_PATTERN = r"[A-Z]{2,3}[0-9]{3,4}"

# Camera configuration
CAMERA_CONFIG = {
    "host": "192.168.1.100",
    "port": 80,
    "username": "admin",
    "password": "password",
}

# OCR configuration optimized for license plates
OCR_CONFIG = {
    "engine": "easyocr",
    "confidence_threshold": 0.6,
    "min_text_length": 4,
    "filter_patterns": [LICENSE_PLATE_PATTERN],
    "easyocr": {"languages": ["en"], "gpu": False},
}


def is_license_plate(text: str) -> bool:
    """Check if text matches license plate pattern."""
    return bool(re.match(LICENSE_PLATE_PATTERN, text.upper()))


def main():
    """Run license plate detection example."""
    # Initialize camera
    camera = ONVIFCameraHandler(**CAMERA_CONFIG)

    if not camera.connect():
        print("Failed to connect to camera")
        return

    if not camera.get_stream_uri():
        print("Failed to get stream URI")
        return

    if not camera.start_stream():
        print("Failed to start stream")
        return

    # Initialize OCR
    ocr = OCREngine(OCR_CONFIG)

    detected_plates = set()

    try:
        while True:
            ret, frame = camera.read_frame()

            if not ret:
                continue

            # Perform OCR
            results = ocr.detect_text(frame)

            # Filter for license plates
            for result in results:
                text = result["text"].upper().replace(" ", "")

                if is_license_plate(text) and text not in detected_plates:
                    detected_plates.add(text)
                    print(
                        f"🚗 Detected License Plate: {text} "
                        f"(confidence: {result['confidence']:.2f})"
                    )

            # Show annotated frame
            if results:
                annotated = ocr.annotate_frame(frame, results)
                cv2.imshow("License Plate Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print(f"\nTotal unique license plates detected: {len(detected_plates)}")
        for plate in sorted(detected_plates):
            print(f"  - {plate}")


if __name__ == "__main__":
    main()
