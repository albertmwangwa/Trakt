"""
Example: Text Detection Pipeline with EAST Detector

This example demonstrates using the EAST text detector to identify text regions
before performing OCR, which can improve accuracy and performance.
"""

import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse  # noqa: E402
import logging  # noqa: E402

import cv2  # noqa: E402

from src.camera_handler import ONVIFCameraHandler  # noqa: E402
from src.ocr_engine import OCREngine  # noqa: E402

logging.basicConfig(level=logging.INFO)

# Camera configuration
CAMERA_CONFIG = {
    "host": "192.168.1.100",
    "port": 80,
    "username": "admin",
    "password": "password",
}

# OCR configuration with text detection enabled
OCR_CONFIG = {
    "engine": "easyocr",
    "confidence_threshold": 0.5,
    "min_text_length": 2,
    "use_text_detection": True,  # Enable EAST text detection
    "text_detection": {
        "east_model_path": "./models/frozen_east_text_detection.pb",
        "confidence_threshold": 0.5,
        "nms_threshold": 0.4,
        "input_width": 320,
        "input_height": 320,
        "region_padding": 5,
    },
    "preprocessing": {
        "apply_deskewing": True,
        "apply_contrast_enhancement": False,
        "apply_noise_removal": False,
        "apply_illumination_normalization": True,
    },
    "easyocr": {"languages": ["en"], "gpu": False},
}


def main():
    """Run text detection pipeline example."""
    parser = argparse.ArgumentParser(
        description="Text Detection Pipeline with EAST Detector"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process (default: 100)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated frames to output directory",
    )
    args = parser.parse_args()

    # Check if EAST model exists
    if not os.path.exists(OCR_CONFIG["text_detection"]["east_model_path"]):
        print("\n⚠️  EAST model not found!")
        print(
            f"Please download the model from: "
            "https://github.com/oyyd/frozen-east-text-detection.pb/raw/master/frozen_east_text_detection.pb"
        )
        print(f"And place it at: {OCR_CONFIG['text_detection']['east_model_path']}")
        print("\nFalling back to standard OCR without text detection...\n")
        OCR_CONFIG["use_text_detection"] = False

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

    # Initialize OCR with text detection
    ocr = OCREngine(OCR_CONFIG)

    # Create output directory if saving frames
    if args.save_frames:
        os.makedirs("./output/text_detection_examples", exist_ok=True)

    frame_count = 0
    total_detections = 0

    try:
        print("\n🚀 Starting text detection pipeline...")
        print(
            f"Text detection: {'ENABLED' if OCR_CONFIG['use_text_detection'] else 'DISABLED'}"
        )
        print("Press 'q' to quit\n")

        while frame_count < args.frames:
            ret, frame = camera.read_frame()

            if not ret:
                continue

            frame_count += 1

            # Perform OCR with text detection
            results = ocr.detect_text(frame)

            if results:
                total_detections += len(results)
                print(f"\n📷 Frame {frame_count}:")
                print(f"   Detected {len(results)} text regions")

                for i, result in enumerate(results, 1):
                    print(
                        f"   {i}. '{result['text']}' "
                        f"(confidence: {result['confidence']:.2f})"
                    )
                    if "region" in result:
                        x1, y1, x2, y2 = result["region"]
                        print(f"      Region: ({x1}, {y1}) -> ({x2}, {y2})")

                # Annotate frame
                annotated = ocr.annotate_frame(frame, results)

                # Save frame if requested
                if args.save_frames:
                    filename = f"./output/text_detection_examples/frame_{frame_count:04d}.jpg"
                    cv2.imwrite(filename, annotated)

                # Show preview
                cv2.imshow("Text Detection Pipeline", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Show frame even if no detections
                cv2.imshow("Text Detection Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print(f"\n📊 Summary:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average per frame: {total_detections / max(frame_count, 1):.2f}")

        if args.save_frames:
            print(f"\n💾 Frames saved to: ./output/text_detection_examples/")


if __name__ == "__main__":
    main()
