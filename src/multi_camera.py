"""
Multi-Camera Manager Module

This module provides support for managing multiple simultaneous camera streams
with thread-safe OCR processing.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    id: str
    host: str
    port: int = 80
    username: str = "admin"
    password: str = "password"
    stream_profile: int = 0
    transport: str = "tcp"
    fps_limit: int = 5
    enabled: bool = True
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"Camera_{self.id}"


@dataclass
class CameraState:
    """Runtime state for a single camera."""

    config: CameraConfig
    handler: any = None
    thread: threading.Thread = None
    running: bool = False
    frame_count: int = 0
    detection_count: int = 0
    last_frame_time: datetime = None
    error_count: int = 0
    status: str = "stopped"
    session_id: int = None


class MultiCameraManager:
    """Manager for handling multiple simultaneous camera streams."""

    def __init__(
        self,
        ocr_engine,
        database=None,
        alert_manager=None,
        frame_callback: Callable = None,
        detection_callback: Callable = None,
    ):
        """
        Initialize multi-camera manager.

        Args:
            ocr_engine: OCREngine instance for text detection
            database: Optional DatabaseManager for storing results
            alert_manager: Optional AlertManager for pattern alerts
            frame_callback: Optional callback for processed frames
            detection_callback: Optional callback for detection results
        """
        self.ocr_engine = ocr_engine
        self.database = database
        self.alert_manager = alert_manager
        self.frame_callback = frame_callback
        self.detection_callback = detection_callback
        self.logger = logging.getLogger(__name__)

        self._cameras: Dict[str, CameraState] = {}
        self._lock = threading.Lock()
        self._running = False
        self._result_queue = queue.Queue(maxsize=1000)
        self._result_thread: Optional[threading.Thread] = None

    def add_camera(self, config: CameraConfig) -> bool:
        """
        Add a camera to the manager.

        Args:
            config: Camera configuration

        Returns:
            True if camera was added successfully
        """
        with self._lock:
            if config.id in self._cameras:
                self.logger.warning(f"Camera {config.id} already exists")
                return False

            self._cameras[config.id] = CameraState(config=config)
            self.logger.info(f"Added camera: {config.id} ({config.name})")
            return True

    def add_cameras_from_config(self, cameras_config: List[Dict]) -> int:
        """
        Add multiple cameras from configuration.

        Args:
            cameras_config: List of camera configuration dictionaries

        Returns:
            Number of cameras added
        """
        count = 0
        for cam_config in cameras_config:
            try:
                config = CameraConfig(
                    id=cam_config.get("id", f"cam_{count}"),
                    host=cam_config.get("host"),
                    port=cam_config.get("port", 80),
                    username=cam_config.get("username", "admin"),
                    password=cam_config.get("password", "password"),
                    stream_profile=cam_config.get("stream_profile", 0),
                    transport=cam_config.get("transport", "tcp"),
                    fps_limit=cam_config.get("fps_limit", 5),
                    enabled=cam_config.get("enabled", True),
                    name=cam_config.get("name", ""),
                )
                if self.add_camera(config):
                    count += 1
            except Exception as e:
                self.logger.error(f"Failed to add camera from config: {e}")

        return count

    def remove_camera(self, camera_id: str) -> bool:
        """
        Remove a camera from the manager.

        Args:
            camera_id: Camera identifier

        Returns:
            True if camera was removed
        """
        with self._lock:
            if camera_id not in self._cameras:
                return False

            state = self._cameras[camera_id]

            # Stop camera if running
            if state.running:
                self._stop_camera_internal(state)

            del self._cameras[camera_id]
            self.logger.info(f"Removed camera: {camera_id}")
            return True

    def start_all(self) -> int:
        """
        Start all enabled cameras.

        Returns:
            Number of cameras started
        """
        self._running = True
        self._start_result_processor()

        started = 0
        with self._lock:
            for camera_id, state in self._cameras.items():
                if state.config.enabled and not state.running:
                    if self._start_camera_internal(state):
                        started += 1

        self.logger.info(f"Started {started} cameras")
        return started

    def stop_all(self):
        """Stop all cameras."""
        self._running = False

        with self._lock:
            for state in self._cameras.values():
                if state.running:
                    self._stop_camera_internal(state)

        # Stop result processor
        if self._result_thread and self._result_thread.is_alive():
            self._result_queue.put(None)  # Signal to stop
            self._result_thread.join(timeout=5)

        self.logger.info("All cameras stopped")

    def start_camera(self, camera_id: str) -> bool:
        """
        Start a specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            True if camera started successfully
        """
        with self._lock:
            if camera_id not in self._cameras:
                return False

            state = self._cameras[camera_id]
            if state.running:
                return True

            return self._start_camera_internal(state)

    def stop_camera(self, camera_id: str) -> bool:
        """
        Stop a specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            True if camera stopped
        """
        with self._lock:
            if camera_id not in self._cameras:
                return False

            state = self._cameras[camera_id]
            if not state.running:
                return True

            return self._stop_camera_internal(state)

    def _start_camera_internal(self, state: CameraState) -> bool:
        """Start a camera (must be called with lock held)."""
        try:
            from .camera_handler import ONVIFCameraHandler

            config = state.config

            # Create camera handler
            handler = ONVIFCameraHandler(
                host=config.host,
                port=config.port,
                username=config.username,
                password=config.password,
            )

            # Connect to camera
            if not handler.connect():
                state.status = "connection_failed"
                state.error_count += 1
                return False

            # Get stream URI
            if not handler.get_stream_uri(config.stream_profile, config.transport):
                state.status = "stream_failed"
                state.error_count += 1
                return False

            # Start streaming
            if not handler.start_stream():
                state.status = "stream_failed"
                state.error_count += 1
                return False

            state.handler = handler
            state.running = True
            state.status = "running"

            # Start camera session in database
            if self.database:
                state.session_id = self.database.start_camera_session(config.id)

            # Start processing thread
            state.thread = threading.Thread(
                target=self._camera_processing_loop,
                args=(state,),
                name=f"Camera-{config.id}",
                daemon=True,
            )
            state.thread.start()

            self.logger.info(f"Camera {config.id} started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start camera {state.config.id}: {e}")
            state.status = "error"
            state.error_count += 1
            return False

    def _stop_camera_internal(self, state: CameraState) -> bool:
        """Stop a camera (must be called with lock held)."""
        try:
            state.running = False

            # Wait for thread to stop
            if state.thread and state.thread.is_alive():
                state.thread.join(timeout=5)

            # Release camera resources
            if state.handler:
                state.handler.release()
                state.handler = None

            # End camera session in database
            if self.database and state.session_id:
                self.database.end_camera_session(
                    state.session_id, state.frame_count, state.detection_count
                )

            state.status = "stopped"
            self.logger.info(f"Camera {state.config.id} stopped")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping camera {state.config.id}: {e}")
            return False

    def _camera_processing_loop(self, state: CameraState):
        """Main processing loop for a camera."""
        config = state.config
        fps_limit = config.fps_limit
        frame_delay = 1.0 / fps_limit if fps_limit > 0 else 0

        self.logger.info(f"Starting processing loop for camera {config.id}")

        while state.running and self._running:
            try:
                start_time = time.time()

                # Read frame
                ret, frame = state.handler.read_frame()

                if not ret or frame is None:
                    state.error_count += 1
                    if state.error_count > 10:
                        self.logger.error(
                            f"Camera {config.id}: Too many read errors, stopping"
                        )
                        break
                    time.sleep(1)
                    continue

                # Reset error count on successful read
                state.error_count = 0
                state.frame_count += 1
                state.last_frame_time = datetime.now()

                # Perform OCR
                results = self.ocr_engine.detect_text(frame)

                if results:
                    state.detection_count += len(results)

                    # Queue results for processing
                    result_data = {
                        "camera_id": config.id,
                        "frame_number": state.frame_count,
                        "timestamp": datetime.now().isoformat(),
                        "detections": results,
                        "frame": frame if self.frame_callback else None,
                    }

                    try:
                        self._result_queue.put_nowait(result_data)
                    except queue.Full:
                        self.logger.warning(
                            f"Result queue full, dropping frame from {config.id}"
                        )

                # Maintain FPS limit
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)

            except Exception as e:
                self.logger.error(f"Error in camera {config.id} loop: {e}")
                state.error_count += 1
                time.sleep(1)

        state.status = "stopped"
        self.logger.info(f"Processing loop ended for camera {config.id}")

    def _start_result_processor(self):
        """Start the result processing thread."""
        self._result_thread = threading.Thread(
            target=self._result_processing_loop, name="ResultProcessor", daemon=True
        )
        self._result_thread.start()

    def _result_processing_loop(self):
        """Process results from all cameras."""
        while self._running:
            try:
                result = self._result_queue.get(timeout=1)

                if result is None:  # Stop signal
                    break

                camera_id = result["camera_id"]
                detections = result["detections"]
                frame_number = result["frame_number"]

                # Save to database
                if self.database and detections:
                    self.database.save_detections_batch(
                        camera_id, detections, frame_number
                    )

                # Check for alerts
                if self.alert_manager and detections:
                    self.alert_manager.check_and_alert(
                        camera_id, detections, frame_number
                    )

                # Call detection callback
                if self.detection_callback:
                    self.detection_callback(camera_id, detections, frame_number)

                # Call frame callback
                if self.frame_callback and result.get("frame") is not None:
                    annotated = self.ocr_engine.annotate_frame(
                        result["frame"], detections
                    )
                    self.frame_callback(camera_id, annotated, frame_number)

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in result processor: {e}")

    def get_camera_status(self, camera_id: str = None) -> Dict:
        """
        Get camera status.

        Args:
            camera_id: Specific camera ID, or None for all cameras

        Returns:
            Status dictionary
        """
        with self._lock:
            if camera_id:
                if camera_id not in self._cameras:
                    return {}
                state = self._cameras[camera_id]
                return self._get_camera_state_dict(state)
            else:
                return {
                    cid: self._get_camera_state_dict(state)
                    for cid, state in self._cameras.items()
                }

    def _get_camera_state_dict(self, state: CameraState) -> Dict:
        """Convert camera state to dictionary."""
        return {
            "id": state.config.id,
            "name": state.config.name,
            "host": state.config.host,
            "enabled": state.config.enabled,
            "status": state.status,
            "running": state.running,
            "frame_count": state.frame_count,
            "detection_count": state.detection_count,
            "error_count": state.error_count,
            "last_frame_time": (
                state.last_frame_time.isoformat() if state.last_frame_time else None
            ),
        }

    def get_statistics(self) -> Dict:
        """
        Get aggregate statistics for all cameras.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_frames = sum(s.frame_count for s in self._cameras.values())
            total_detections = sum(s.detection_count for s in self._cameras.values())
            running_cameras = sum(1 for s in self._cameras.values() if s.running)

            return {
                "total_cameras": len(self._cameras),
                "running_cameras": running_cameras,
                "total_frames": total_frames,
                "total_detections": total_detections,
                "cameras": {
                    cid: {
                        "frames": s.frame_count,
                        "detections": s.detection_count,
                        "status": s.status,
                    }
                    for cid, s in self._cameras.items()
                },
            }
