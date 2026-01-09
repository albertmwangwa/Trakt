"""
Multi-Camera Manager Module

This module handles multiple simultaneous camera connections and processing.
"""

import logging
import threading
import time
from typing import Dict, List, Optional
from queue import Queue

from .camera_handler import ONVIFCameraHandler
from .ocr_engine import OCREngine


class CameraInstance:
    """Represents a single camera instance with its processing thread."""

    def __init__(
        self,
        camera_id: int,
        name: str,
        config: dict,
        ocr_engine: OCREngine,
        database_manager=None,
        alert_manager=None,
    ):
        """
        Initialize camera instance.

        Args:
            camera_id: Database camera ID
            name: Camera name
            config: Camera configuration
            ocr_engine: OCR engine instance
            database_manager: Optional database manager
            alert_manager: Optional alert manager
        """
        self.camera_id = camera_id
        self.name = name
        self.config = config
        self.ocr_engine = ocr_engine
        self.database_manager = database_manager
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(f"{__name__}.{name}")

        # Camera handler
        self.handler = None

        # Processing state
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.detection_count = 0

        # Frame queue for async processing
        self.frame_queue = Queue(maxsize=10)

    def start(self) -> bool:
        """
        Start camera processing.

        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Initialize camera handler
            self.handler = ONVIFCameraHandler(
                host=self.config.get("host"),
                port=self.config.get("port"),
                username=self.config.get("username"),
                password=self.config.get("password"),
            )

            # Connect to camera
            if not self.handler.connect():
                self.logger.error(f"Failed to connect to camera {self.name}")
                return False

            # Get stream URI
            stream_profile = self.config.get("stream_profile", 0)
            transport = self.config.get("transport", "tcp")

            if not self.handler.get_stream_uri(stream_profile, transport):
                self.logger.error(f"Failed to get stream URI for camera {self.name}")
                return False

            # Start stream
            if not self.handler.start_stream():
                self.logger.error(f"Failed to start stream for camera {self.name}")
                return False

            # Start processing thread
            self.running = True
            self.thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.thread.start()

            self.logger.info(f"Camera {self.name} started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start camera {self.name}: {e}")
            return False

    def stop(self):
        """Stop camera processing."""
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

        if self.handler:
            self.handler.release()

        self.logger.info(
            f"Camera {self.name} stopped. "
            f"Processed {self.frame_count} frames, "
            f"{self.detection_count} detections"
        )

    def _processing_loop(self):
        """Main processing loop for the camera."""
        fps_limit = self.config.get("fps_limit", 5)
        frame_delay = 1.0 / fps_limit if fps_limit > 0 else 0

        while self.running:
            try:
                start_time = time.time()

                # Read frame
                ret, frame = self.handler.read_frame()

                if not ret or frame is None:
                    self.logger.warning(f"Failed to read frame from {self.name}")
                    time.sleep(1)
                    continue

                # Process frame
                self._process_frame(frame)

                # Maintain FPS limit
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)

            except Exception as e:
                self.logger.error(f"Error in processing loop for {self.name}: {e}")
                time.sleep(1)

    def _process_frame(self, frame):
        """
        Process a single frame.

        Args:
            frame: Input frame
        """
        self.frame_count += 1

        # Perform OCR
        results = self.ocr_engine.detect_text(frame)

        if results:
            self.detection_count += len(results)
            self.logger.debug(
                f"Frame {self.frame_count}: Detected {len(results)} text regions"
            )

            # Save results to database
            if self.database_manager:
                for result in results:
                    detection = self.database_manager.add_detection(
                        camera_id=self.camera_id,
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
                            camera_name=self.name,
                            detection_id=detection.id,
                        )

    def get_status(self) -> dict:
        """
        Get camera status.

        Returns:
            Dictionary with camera status information
        """
        return {
            "name": self.name,
            "camera_id": self.camera_id,
            "running": self.running,
            "frame_count": self.frame_count,
            "detection_count": self.detection_count,
            "connected": self.handler is not None and self.handler.capture is not None,
        }


class MultiCameraManager:
    """Manager for multiple simultaneous cameras."""

    def __init__(
        self,
        cameras_config: List[dict],
        ocr_config: dict,
        database_manager=None,
        alert_manager=None,
    ):
        """
        Initialize multi-camera manager.

        Args:
            cameras_config: List of camera configurations
            ocr_config: OCR configuration
            database_manager: Optional database manager
            alert_manager: Optional alert manager
        """
        self.cameras_config = cameras_config
        self.ocr_config = ocr_config
        self.database_manager = database_manager
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)

        # Camera instances
        self.cameras: Dict[str, CameraInstance] = {}

        # Create OCR engines (one per camera for thread safety)
        self.ocr_engines: Dict[str, OCREngine] = {}

    def initialize_cameras(self) -> bool:
        """
        Initialize all configured cameras.

        Returns:
            True if at least one camera initialized successfully
        """
        self.logger.info(f"Initializing {len(self.cameras_config)} cameras...")

        success_count = 0

        for camera_config in self.cameras_config:
            name = camera_config.get("name", f"Camera_{len(self.cameras) + 1}")

            try:
                # Register camera in database
                camera_id = None
                if self.database_manager:
                    camera_db = self.database_manager.add_camera(
                        name=name,
                        host=camera_config.get("host"),
                        port=camera_config.get("port"),
                        username=camera_config.get("username"),
                        stream_profile=camera_config.get("stream_profile", 0),
                    )
                    if camera_db:
                        camera_id = camera_db.id

                # Create OCR engine for this camera
                ocr_engine = OCREngine(self.ocr_config)
                self.ocr_engines[name] = ocr_engine

                # Create camera instance
                camera_instance = CameraInstance(
                    camera_id=camera_id or 0,
                    name=name,
                    config=camera_config,
                    ocr_engine=ocr_engine,
                    database_manager=self.database_manager,
                    alert_manager=self.alert_manager,
                )

                # Start camera
                if camera_instance.start():
                    self.cameras[name] = camera_instance
                    success_count += 1
                    self.logger.info(f"Successfully initialized camera: {name}")
                else:
                    self.logger.error(f"Failed to initialize camera: {name}")

            except Exception as e:
                self.logger.error(f"Error initializing camera {name}: {e}")

        self.logger.info(
            f"Initialized {success_count}/{len(self.cameras_config)} cameras"
        )
        return success_count > 0

    def stop_all(self):
        """Stop all cameras."""
        self.logger.info("Stopping all cameras...")

        for name, camera in self.cameras.items():
            try:
                camera.stop()
            except Exception as e:
                self.logger.error(f"Error stopping camera {name}: {e}")

        self.cameras.clear()
        self.logger.info("All cameras stopped")

    def get_status(self) -> dict:
        """
        Get status of all cameras.

        Returns:
            Dictionary with status information for all cameras
        """
        return {
            "total_cameras": len(self.cameras),
            "cameras": [camera.get_status() for camera in self.cameras.values()],
        }

    def get_camera(self, name: str) -> Optional[CameraInstance]:
        """
        Get camera instance by name.

        Args:
            name: Camera name

        Returns:
            CameraInstance or None if not found
        """
        return self.cameras.get(name)
