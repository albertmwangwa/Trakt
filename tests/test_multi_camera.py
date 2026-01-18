"""
Tests for Multi-Camera Manager Module
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import threading
import time

# Add src to path before import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import directly from the module file bypassing __init__.py
import importlib.util

spec = importlib.util.spec_from_file_location(
    "multi_camera", os.path.join(os.path.dirname(__file__), "..", "src", "multi_camera.py")
)
multi_camera = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_camera)

CameraConfig = multi_camera.CameraConfig
CameraState = multi_camera.CameraState
MultiCameraManager = multi_camera.MultiCameraManager


class TestCameraConfig(unittest.TestCase):
    """Test cases for CameraConfig class."""

    def test_initialization(self):
        """Test camera config initialization."""
        config = CameraConfig(
            id="cam_1",
            host="192.168.1.100",
            port=80,
            username="admin",
            password="secret",
            stream_profile=0,
            transport="tcp",
            fps_limit=5,
            enabled=True,
            name="Front Camera",
        )

        self.assertEqual(config.id, "cam_1")
        self.assertEqual(config.host, "192.168.1.100")
        self.assertEqual(config.port, 80)
        self.assertEqual(config.name, "Front Camera")
        self.assertTrue(config.enabled)

    def test_default_name(self):
        """Test default name generation."""
        config = CameraConfig(id="test_cam", host="192.168.1.1")

        self.assertEqual(config.name, "Camera_test_cam")

    def test_default_values(self):
        """Test default values."""
        config = CameraConfig(id="cam_1", host="192.168.1.1")

        self.assertEqual(config.port, 80)
        self.assertEqual(config.username, "admin")
        self.assertEqual(config.password, "password")
        self.assertEqual(config.stream_profile, 0)
        self.assertEqual(config.transport, "tcp")
        self.assertEqual(config.fps_limit, 5)
        self.assertTrue(config.enabled)


class TestCameraState(unittest.TestCase):
    """Test cases for CameraState class."""

    def test_initialization(self):
        """Test camera state initialization."""
        config = CameraConfig(id="cam_1", host="192.168.1.1")
        state = CameraState(config=config)

        self.assertEqual(state.config, config)
        self.assertIsNone(state.handler)
        self.assertIsNone(state.thread)
        self.assertFalse(state.running)
        self.assertEqual(state.frame_count, 0)
        self.assertEqual(state.detection_count, 0)
        self.assertEqual(state.error_count, 0)
        self.assertEqual(state.status, "stopped")


class TestMultiCameraManager(unittest.TestCase):
    """Test cases for MultiCameraManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ocr_engine = Mock()
        self.mock_ocr_engine.detect_text = Mock(return_value=[])
        self.mock_ocr_engine.annotate_frame = Mock(return_value=None)

    def test_initialization(self):
        """Test manager initialization."""
        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)

        self.assertIsNotNone(manager)
        self.assertEqual(len(manager._cameras), 0)

    def test_add_camera(self):
        """Test adding a camera."""
        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)
        config = CameraConfig(id="cam_1", host="192.168.1.100")

        result = manager.add_camera(config)

        self.assertTrue(result)
        self.assertEqual(len(manager._cameras), 1)
        self.assertIn("cam_1", manager._cameras)

    def test_add_duplicate_camera(self):
        """Test adding a duplicate camera."""
        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)
        config = CameraConfig(id="cam_1", host="192.168.1.100")

        manager.add_camera(config)
        result = manager.add_camera(config)

        self.assertFalse(result)
        self.assertEqual(len(manager._cameras), 1)

    def test_add_cameras_from_config(self):
        """Test adding multiple cameras from config."""
        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)
        cameras_config = [
            {"id": "cam_1", "host": "192.168.1.100", "name": "Camera 1"},
            {"id": "cam_2", "host": "192.168.1.101", "name": "Camera 2"},
            {"id": "cam_3", "host": "192.168.1.102", "enabled": False},
        ]

        count = manager.add_cameras_from_config(cameras_config)

        self.assertEqual(count, 3)
        self.assertEqual(len(manager._cameras), 3)

    def test_remove_camera(self):
        """Test removing a camera."""
        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)
        config = CameraConfig(id="cam_1", host="192.168.1.100")
        manager.add_camera(config)

        result = manager.remove_camera("cam_1")

        self.assertTrue(result)
        self.assertEqual(len(manager._cameras), 0)

    def test_remove_nonexistent_camera(self):
        """Test removing a non-existent camera."""
        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)

        result = manager.remove_camera("nonexistent")

        self.assertFalse(result)

    def test_get_camera_status_all(self):
        """Test getting status of all cameras."""
        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)
        manager.add_camera(CameraConfig(id="cam_1", host="192.168.1.100"))
        manager.add_camera(CameraConfig(id="cam_2", host="192.168.1.101"))

        status = manager.get_camera_status()

        self.assertEqual(len(status), 2)
        self.assertIn("cam_1", status)
        self.assertIn("cam_2", status)

    def test_get_camera_status_single(self):
        """Test getting status of a single camera."""
        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)
        manager.add_camera(CameraConfig(id="cam_1", host="192.168.1.100", name="Test Camera"))

        status = manager.get_camera_status("cam_1")

        self.assertIsInstance(status, dict)
        self.assertEqual(status["id"], "cam_1")
        self.assertEqual(status["name"], "Test Camera")
        self.assertEqual(status["status"], "stopped")

    def test_get_camera_status_nonexistent(self):
        """Test getting status of non-existent camera."""
        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)

        status = manager.get_camera_status("nonexistent")

        self.assertEqual(status, {})

    def test_get_statistics(self):
        """Test getting aggregate statistics."""
        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)
        manager.add_camera(CameraConfig(id="cam_1", host="192.168.1.100"))
        manager.add_camera(CameraConfig(id="cam_2", host="192.168.1.101"))

        stats = manager.get_statistics()

        self.assertEqual(stats["total_cameras"], 2)
        self.assertEqual(stats["running_cameras"], 0)
        self.assertEqual(stats["total_frames"], 0)
        self.assertEqual(stats["total_detections"], 0)
        self.assertIn("cameras", stats)

    def test_callbacks(self):
        """Test that callbacks are properly set."""
        detection_callback = Mock()
        frame_callback = Mock()

        manager = MultiCameraManager(
            ocr_engine=self.mock_ocr_engine,
            detection_callback=detection_callback,
            frame_callback=frame_callback,
        )

        self.assertEqual(manager.detection_callback, detection_callback)
        self.assertEqual(manager.frame_callback, frame_callback)


class TestMultiCameraManagerWithMockedCamera(unittest.TestCase):
    """Test cases for MultiCameraManager with mocked camera handler."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ocr_engine = Mock()
        self.mock_ocr_engine.detect_text = Mock(return_value=[])

    @patch.object(MultiCameraManager, "_start_camera_internal")
    def test_start_all(self, mock_start):
        """Test starting all cameras."""
        mock_start.return_value = True

        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)
        manager.add_camera(CameraConfig(id="cam_1", host="192.168.1.100"))
        manager.add_camera(CameraConfig(id="cam_2", host="192.168.1.101"))

        started = manager.start_all()

        self.assertEqual(started, 2)
        self.assertEqual(mock_start.call_count, 2)

        # Cleanup
        manager._running = False

    @patch.object(MultiCameraManager, "_stop_camera_internal")
    def test_stop_all(self, mock_stop):
        """Test stopping all cameras."""
        mock_stop.return_value = True

        manager = MultiCameraManager(ocr_engine=self.mock_ocr_engine)
        manager.add_camera(CameraConfig(id="cam_1", host="192.168.1.100"))
        manager._cameras["cam_1"].running = True

        manager.stop_all()

        mock_stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
