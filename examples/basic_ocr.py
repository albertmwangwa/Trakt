"""
Example: Basic OCR from ONVIF Camera

This example demonstrates basic usage of the Trakt OCR application.
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera_handler import ONVIFCameraHandler
from src.ocr_engine import OCREngine
import cv2
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)

# Camera configuration
CAMERA_HOST = "192.168.1.100"
CAMERA_PORT = 80
CAMERA_USER = "admin"
CAMERA_PASS = "password"

# OCR configuration
ocr_config = {
    'engine': 'easyocr',
    'confidence_threshold': 0.5,
    'min_text_length': 2,
    'easyocr': {
        'languages': ['en'],
        'gpu': False
    }
}


def main():
    """Run basic OCR example."""
    parser = argparse.ArgumentParser(description='Basic OCR from ONVIF Camera')
    parser.add_argument('--frames', type=int, default=100, 
                       help='Number of frames to process (default: 100)')
    args = parser.parse_args()
    
    # Initialize camera
    camera = ONVIFCameraHandler(
        CAMERA_HOST, CAMERA_PORT, CAMERA_USER, CAMERA_PASS
    )
    
    # Connect to camera
    if not camera.connect():
        print("Failed to connect to camera")
        return
    
    # Get stream URI
    if not camera.get_stream_uri():
        print("Failed to get stream URI")
        return
    
    # Start streaming
    if not camera.start_stream():
        print("Failed to start stream")
        return
    
    # Initialize OCR
    ocr = OCREngine(ocr_config)
    
    # Process frames
    frame_count = 0
    try:
        while frame_count < args.frames:
            ret, frame = camera.read_frame()
            
            if not ret:
                continue
            
            frame_count += 1
            
            # Perform OCR
            results = ocr.detect_text(frame)
            
            if results:
                print(f"\nFrame {frame_count}:")
                for result in results:
                    print(f"  Text: '{result['text']}' "
                          f"(confidence: {result['confidence']:.2f})")
                
                # Show annotated frame
                annotated = ocr.annotate_frame(frame, results)
                cv2.imshow('OCR Results', annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
