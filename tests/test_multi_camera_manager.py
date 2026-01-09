"""
Tests for Multi-Camera Manager Module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time

from src.multi_camera_manager import CameraInstance, MultiCameraManager


class TestCameraInstance(unittest.TestCase):
    """Test cases for CameraInstance class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "host": "192.168.1.100",
            "port": 80,
            "username": "admin",
            "password": "password",
            "stream_profile": 0,
            "transport": "tcp",
            "fps_limit": 5,
        }
        self.mock_ocr_engine = Mock()

    @patch("src.multi_camera_manager.ONVIFCameraHandler")
    def test_initialization(self, mock_handler):
        """Test camera instance initialization."""
        instance = CameraInstance(
            camera_id=1,
            name="Test Camera",
            config=self.config,
            ocr_engine=self.mock_ocr_engine,
        )

        self.assertEqual(instance.camera_id, 1)
        self.assertEqual(instance.name, "Test Camera")
        self.assertFalse(instance.running)
        self.assertEqual(instance.frame_count, 0)

    @patch("src.multi_camera_manager.ONVIFCameraHandler")
    def test_start_success(self, mock_handler_class):
        """Test successful camera start."""
        # Mock handler
        mock_handler = Mock()
        mock_handler.connect.return_value = True
        mock_handler.get_stream_uri.return_value = "rtsp://test"
        mock_handler.start_stream.return_value = True
        mock_handler_class.return_value = mock_handler

        instance = CameraInstance(
            camera_id=1,
            name="Test Camera",
            config=self.config,
            ocr_engine=self.mock_ocr_engine,
        )

        result = instance.start()

        self.assertTrue(result)
        self.assertTrue(instance.running)
        self.assertIsNotNone(instance.thread)

        # Clean up
        instance.stop()

    @patch("src.multi_camera_manager.ONVIFCameraHandler")
    def test_start_failure(self, mock_handler_class):
        """Test failed camera start."""
        # Mock handler that fails to connect
        mock_handler = Mock()
        mock_handler.connect.return_value = False
        mock_handler_class.return_value = mock_handler

        instance = CameraInstance(
            camera_id=1,
            name="Test Camera",
            config=self.config,
            ocr_engine=self.mock_ocr_engine,
        )

        result = instance.start()

        self.assertFalse(result)
        self.assertFalse(instance.running)

    @patch("src.multi_camera_manager.ONVIFCameraHandler")
    def test_get_status(self, mock_handler_class):
        """Test getting camera status."""
        instance = CameraInstance(
            camera_id=1,
            name="Test Camera",
            config=self.config,
            ocr_engine=self.mock_ocr_engine,
        )

        status = instance.get_status()

        self.assertEqual(status["name"], "Test Camera")
        self.assertEqual(status["camera_id"], 1)
        self.assertFalse(status["running"])
        self.assertEqual(status["frame_count"], 0)


class TestMultiCameraManager(unittest.TestCase):
    """Test cases for MultiCameraManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cameras_config = [
            {
                "name": "Camera 1",
                "host": "192.168.1.100",
                "port": 80,
                "username": "admin",
                "password": "password",
            },
            {
                "name": "Camera 2",
                "host": "192.168.1.101",
                "port": 80,
                "username": "admin",
                "password": "password",
            },
        ]
        self.ocr_config = {"engine": "easyocr"}

    def test_initialization(self):
        """Test multi-camera manager initialization."""
        manager = MultiCameraManager(
            cameras_config=self.cameras_config, ocr_config=self.ocr_config
        )

        self.assertEqual(len(manager.cameras_config), 2)
        self.assertEqual(len(manager.cameras), 0)

    @patch("src.multi_camera_manager.OCREngine")
    @patch("src.multi_camera_manager.CameraInstance")
    def test_initialize_cameras(self, mock_instance_class, mock_ocr_class):
        """Test camera initialization."""
        # Mock camera instances
        mock_instance1 = Mock()
        mock_instance1.start.return_value = True
        mock_instance2 = Mock()
        mock_instance2.start.return_value = True

        mock_instance_class.side_effect = [mock_instance1, mock_instance2]

        manager = MultiCameraManager(
            cameras_config=self.cameras_config, ocr_config=self.ocr_config
        )

        result = manager.initialize_cameras()

        self.assertTrue(result)
        self.assertEqual(len(manager.cameras), 2)

        # Clean up
        manager.stop_all()

    @patch("src.multi_camera_manager.OCREngine")
    @patch("src.multi_camera_manager.CameraInstance")
    def test_stop_all_cameras(self, mock_instance_class, mock_ocr_class):
        """Test stopping all cameras."""
        # Mock camera instances
        mock_instance1 = Mock()
        mock_instance1.start.return_value = True
        mock_instance1.stop.return_value = None

        mock_instance_class.return_value = mock_instance1

        manager = MultiCameraManager(
            cameras_config=[self.cameras_config[0]], ocr_config=self.ocr_config
        )

        manager.initialize_cameras()
        manager.stop_all()

        self.assertEqual(len(manager.cameras), 0)
        mock_instance1.stop.assert_called()

    @patch("src.multi_camera_manager.OCREngine")
    @patch("src.multi_camera_manager.CameraInstance")
    def test_get_status(self, mock_instance_class, mock_ocr_class):
        """Test getting manager status."""
        mock_instance = Mock()
        mock_instance.start.return_value = True
        mock_instance.get_status.return_value = {
            "name": "Camera 1",
            "running": True,
        }

        mock_instance_class.return_value = mock_instance

        manager = MultiCameraManager(
            cameras_config=[self.cameras_config[0]], ocr_config=self.ocr_config
        )

        manager.initialize_cameras()
        status = manager.get_status()

        self.assertEqual(status["total_cameras"], 1)
        self.assertEqual(len(status["cameras"]), 1)

        # Clean up
        manager.stop_all()

    @patch("src.multi_camera_manager.OCREngine")
    @patch("src.multi_camera_manager.CameraInstance")
    def test_get_camera_by_name(self, mock_instance_class, mock_ocr_class):
        """Test getting camera by name."""
        mock_instance = Mock()
        mock_instance.start.return_value = True
        mock_instance.name = "Camera 1"

        mock_instance_class.return_value = mock_instance

        manager = MultiCameraManager(
            cameras_config=[self.cameras_config[0]], ocr_config=self.ocr_config
        )

        manager.initialize_cameras()
        camera = manager.get_camera("Camera 1")

        self.assertIsNotNone(camera)
        self.assertEqual(camera.name, "Camera 1")

        # Clean up
        manager.stop_all()


if __name__ == "__main__":
    unittest.main()
