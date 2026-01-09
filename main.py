"""
Main Application Module

TensorFlow-based OCR application for ONVIF cameras.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import coloredlogs
import cv2
import yaml

from src.camera_handler import ONVIFCameraHandler
from src.ocr_engine import OCREngine, TensorFlowOCRModel
from src.database import DatabaseManager
from src.alert_manager import AlertManager
from src.multi_camera_manager import MultiCameraManager


class TraktOCRApp:
    """Main application class for Trakt OCR."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize Trakt OCR application.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.camera = None
        self.ocr_engine = None
        self.tf_model = None
        self.database_manager = None
        self.alert_manager = None
        self.multi_camera_manager = None

        # Runtime state
        self.running = False
        self.frame_count = 0
        self.detection_count = 0

        # Create output directories
        self._create_output_dirs()

        # Initialize database if enabled
        self._initialize_database()

        # Initialize alert manager if enabled
        self._initialize_alerts()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    def _setup_logging(self):
        """Setup logging configuration."""
        output_config = self.config.get("output", {})
        log_level = output_config.get("log_level", "INFO")
        log_file = output_config.get("log_file", "./output/trakt.log")

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        # Add colored logs for console
        coloredlogs.install(
            level=log_level, fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def _create_output_dirs(self):
        """Create output directories."""
        output_config = self.config.get("output", {})

        if output_config.get("save_results"):
            os.makedirs(
                output_config.get("results_dir", "./output/results"), exist_ok=True
            )

        if output_config.get("save_frames"):
            os.makedirs(
                output_config.get("frames_dir", "./output/frames"), exist_ok=True
            )

    def _initialize_database(self):
        """Initialize database if enabled."""
        db_config = self.config.get("database", {})

        if db_config.get("enabled", False):
            db_url = db_config.get("url", "sqlite:///./output/trakt.db")
            self.logger.info(f"Initializing database: {db_url}")

            try:
                self.database_manager = DatabaseManager(db_url)
                self.logger.info("Database initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize database: {e}")

    def _initialize_alerts(self):
        """Initialize alert manager if enabled."""
        alert_config = self.config.get("alerts", {})

        if alert_config.get("enabled", False):
            self.logger.info("Initializing alert manager...")

            try:
                self.alert_manager = AlertManager(
                    alert_config, database_manager=self.database_manager
                )
                self.logger.info("Alert manager initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize alert manager: {e}")

    def initialize_camera(self) -> bool:
        """
        Initialize ONVIF camera connection.

        Returns:
            True if successful, False otherwise
        """
        camera_config = self.config.get("camera", {})

        self.logger.info("Initializing camera connection...")

        self.camera = ONVIFCameraHandler(
            host=camera_config.get("host"),
            port=camera_config.get("port"),
            username=camera_config.get("username"),
            password=camera_config.get("password"),
        )

        # Connect to camera
        if not self.camera.connect():
            return False

        # Get stream URI
        stream_profile = camera_config.get("stream_profile", 0)
        transport = camera_config.get("transport", "tcp")

        if not self.camera.get_stream_uri(stream_profile, transport):
            return False

        # Start streaming
        if not self.camera.start_stream():
            return False

        # Log camera info
        info = self.camera.get_camera_info()
        self.logger.info(f"Camera Info: {json.dumps(info, indent=2)}")

        return True

    def initialize_ocr(self):
        """Initialize OCR engine."""
        self.logger.info("Initializing OCR engine...")

        ocr_config = self.config.get("ocr", {})
        self.ocr_engine = OCREngine(ocr_config)

        self.logger.info("OCR engine initialized")

    def initialize_tensorflow_model(self):
        """Initialize TensorFlow model if configured."""
        tf_config = self.config.get("tensorflow", {})

        if not tf_config.get("use_pretrained", False):
            self.logger.info("Custom TensorFlow model not enabled")
            return

        model_path = tf_config.get("model_path")
        if model_path and os.path.exists(model_path):
            self.logger.info("Initializing TensorFlow model...")
            self.tf_model = TensorFlowOCRModel(model_path, tf_config)
        else:
            self.logger.warning("TensorFlow model path not found")

    def process_frame(self, frame, camera_id=None):
        """
        Process a single frame for OCR.

        Args:
            frame: Input frame from camera
            camera_id: Optional camera ID for database storage
        """
        self.frame_count += 1

        # Perform OCR
        results = self.ocr_engine.detect_text(frame)

        if results:
            self.detection_count += len(results)
            self.logger.info(
                f"Frame {self.frame_count}: Detected {len(results)} text regions"
            )

            # Log detected text
            for result in results:
                self.logger.debug(
                    f"  Text: '{result['text']}' "
                    f"(confidence: {result['confidence']:.2f})"
                )

            # Save results to database
            if self.database_manager and camera_id:
                for result in results:
                    detection = self.database_manager.add_detection(
                        camera_id=camera_id,
                        frame_number=self.frame_count,
                        text=result["text"],
                        confidence=result["confidence"],
                        bbox=result.get("bbox"),
                        matched_pattern=result.get("matched_pattern"),
                    )

                    # Check for alerts
                    if self.alert_manager and detection:
                        self.alert_manager.check_text(
                            text=result["text"],
                            camera_name=f"Camera_{camera_id}",
                            detection_id=detection.id,
                        )

            # Save results to file (legacy support)
            self._save_results(results)

            # Annotate and save frame
            if self.config.get("output", {}).get("save_frames"):
                annotated = self.ocr_engine.annotate_frame(frame, results)
                self._save_frame(annotated)

        # Show preview if enabled
        if self.config.get("app", {}).get("show_preview"):
            annotated = self.ocr_engine.annotate_frame(frame, results)
            scale = self.config.get("app", {}).get("preview_scale", 0.5)
            preview = cv2.resize(annotated, None, fx=scale, fy=scale)
            cv2.imshow("Trakt OCR Preview", preview)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False

    def _save_results(self, results):
        """Save OCR results to file."""
        output_config = self.config.get("output", {})

        if not output_config.get("save_results"):
            return

        results_dir = output_config.get("results_dir", "./output/results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/results_{timestamp}.json"

        data = {
            "timestamp": timestamp,
            "frame_number": self.frame_count,
            "detections": results,
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def _save_frame(self, frame):
        """Save annotated frame."""
        output_config = self.config.get("output", {})
        save_interval = output_config.get("save_interval", 30)

        if self.frame_count % save_interval != 0:
            return

        frames_dir = output_config.get("frames_dir", "./output/frames")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{frames_dir}/frame_{self.frame_count}_{timestamp}.jpg"

        cv2.imwrite(filename, frame)
        self.logger.debug(f"Saved frame: {filename}")

    def run(self):
        """Run the main application loop."""
        self.logger.info("Starting Trakt OCR Application")

        try:
            # Check if multi-camera mode is enabled
            app_config = self.config.get("app", {})
            multi_camera_mode = app_config.get("multi_camera_mode", False)

            if multi_camera_mode and "cameras" in self.config:
                self._run_multi_camera()
            else:
                self._run_single_camera()

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def _run_single_camera(self):
        """Run in single camera mode."""
        # Initialize camera
        if not self.initialize_camera():
            self.logger.error("Failed to initialize camera")
            return

        # Initialize OCR
        self.initialize_ocr()

        # Initialize TensorFlow model (optional)
        self.initialize_tensorflow_model()

        # Register camera in database
        camera_id = None
        if self.database_manager:
            camera_config = self.config.get("camera", {})
            camera_db = self.database_manager.add_camera(
                name="default_camera",
                host=camera_config.get("host"),
                port=camera_config.get("port"),
                username=camera_config.get("username"),
                stream_profile=camera_config.get("stream_profile", 0),
            )
            if camera_db:
                camera_id = camera_db.id

        # Main processing loop
        self.running = True
        camera_config = self.config.get("camera", {})
        fps_limit = camera_config.get("fps_limit", 5)
        frame_delay = 1.0 / fps_limit if fps_limit > 0 else 0

        self.logger.info("Starting main processing loop...")

        while self.running:
            start_time = time.time()

            # Read frame
            ret, frame = self.camera.read_frame()

            if not ret or frame is None:
                self.logger.warning("Failed to read frame")
                time.sleep(1)
                continue

            # Process frame
            self.process_frame(frame, camera_id=camera_id)

            # Maintain FPS limit
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    def _run_multi_camera(self):
        """Run in multi-camera mode."""
        cameras_config = self.config.get("cameras", [])

        if not cameras_config:
            self.logger.error("No cameras configured for multi-camera mode")
            return

        self.logger.info(f"Starting multi-camera mode with {len(cameras_config)} cameras")

        # Initialize OCR config
        ocr_config = self.config.get("ocr", {})

        # Create multi-camera manager
        self.multi_camera_manager = MultiCameraManager(
            cameras_config=cameras_config,
            ocr_config=ocr_config,
            database_manager=self.database_manager,
            alert_manager=self.alert_manager,
        )

        # Initialize all cameras
        if not self.multi_camera_manager.initialize_cameras():
            self.logger.error("Failed to initialize any cameras")
            return

        # Keep main thread alive
        self.running = True
        self.logger.info("All cameras running. Press Ctrl+C to stop.")

        try:
            while self.running:
                time.sleep(1)

                # Print status periodically
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    status = self.multi_camera_manager.get_status()
                    self.logger.info(
                        f"Status: {status['total_cameras']} cameras active"
                    )

        except KeyboardInterrupt:
            self.logger.info("Stopping multi-camera mode...")
            self.multi_camera_manager.stop_all()

    def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up resources...")

        if self.camera:
            self.camera.release()

        if self.multi_camera_manager:
            self.multi_camera_manager.stop_all()

        cv2.destroyAllWindows()

        self.logger.info(f"Processed {self.frame_count} frames")
        self.logger.info(f"Total detections: {self.detection_count}")
        self.logger.info("Application stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trakt - TensorFlow OCR for ONVIF Cameras"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    args = parser.parse_args()

    # Create and run application
    app = TraktOCRApp(config_path=args.config)
    app.run()


if __name__ == "__main__":
    main()
