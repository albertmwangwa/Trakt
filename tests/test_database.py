"""
Tests for Database Module
"""

import unittest
import os
import tempfile
from datetime import datetime

from src.database import DatabaseManager, Camera, Detection, Alert


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database for testing
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.db_url = f"sqlite:///{self.db_path}"
        self.db_manager = DatabaseManager(self.db_url)

    def tearDown(self):
        """Clean up test fixtures."""
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_database_initialization(self):
        """Test database initialization."""
        self.assertIsNotNone(self.db_manager)
        self.assertIsNotNone(self.db_manager.engine)

    def test_add_camera(self):
        """Test adding a camera."""
        camera = self.db_manager.add_camera(
            name="Test Camera",
            host="192.168.1.100",
            port=80,
            username="admin",
        )

        self.assertIsNotNone(camera)
        self.assertEqual(camera.name, "Test Camera")
        self.assertEqual(camera.host, "192.168.1.100")
        self.assertEqual(camera.port, 80)

    def test_get_camera_by_name(self):
        """Test retrieving camera by name."""
        # Add camera
        self.db_manager.add_camera(
            name="Test Camera", host="192.168.1.100", port=80
        )

        # Retrieve camera
        camera = self.db_manager.get_camera_by_name("Test Camera")

        self.assertIsNotNone(camera)
        self.assertEqual(camera.name, "Test Camera")

    def test_get_active_cameras(self):
        """Test retrieving active cameras."""
        # Add cameras
        self.db_manager.add_camera(
            name="Camera 1", host="192.168.1.100", port=80, active=True
        )
        self.db_manager.add_camera(
            name="Camera 2", host="192.168.1.101", port=80, active=False
        )

        # Get active cameras
        cameras = self.db_manager.get_active_cameras()

        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0].name, "Camera 1")

    def test_add_detection(self):
        """Test adding a detection."""
        # First add a camera
        camera = self.db_manager.add_camera(
            name="Test Camera", host="192.168.1.100", port=80
        )

        # Add detection
        detection = self.db_manager.add_detection(
            camera_id=camera.id,
            frame_number=1,
            text="ABC123",
            confidence=0.95,
            bbox=[10, 20, 100, 50],
            matched_pattern="[A-Z]{3}[0-9]{3}",
        )

        self.assertIsNotNone(detection)
        self.assertEqual(detection.text, "ABC123")
        self.assertEqual(detection.confidence, 0.95)
        self.assertEqual(detection.bbox_x1, 10)

    def test_add_alert(self):
        """Test adding an alert."""
        # Add camera and detection
        camera = self.db_manager.add_camera(
            name="Test Camera", host="192.168.1.100", port=80
        )
        detection = self.db_manager.add_detection(
            camera_id=camera.id,
            frame_number=1,
            text="ABC123",
            confidence=0.95,
        )

        # Add alert
        alert = self.db_manager.add_alert(
            detection_id=detection.id,
            alert_type="email",
            pattern="[A-Z]{3}[0-9]{3}",
            message="License plate detected",
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.alert_type, "email")
        self.assertFalse(alert.sent)

    def test_mark_alert_sent(self):
        """Test marking alert as sent."""
        # Add camera, detection, and alert
        camera = self.db_manager.add_camera(
            name="Test Camera", host="192.168.1.100", port=80
        )
        detection = self.db_manager.add_detection(
            camera_id=camera.id, frame_number=1, text="ABC123", confidence=0.95
        )
        alert = self.db_manager.add_alert(
            detection_id=detection.id,
            alert_type="email",
            pattern="[A-Z]{3}[0-9]{3}",
        )

        # Mark as sent
        self.db_manager.mark_alert_sent(alert.id, success=True)

        # Verify
        session = self.db_manager.get_session()
        updated_alert = session.query(Alert).filter_by(id=alert.id).first()
        session.close()

        self.assertTrue(updated_alert.sent)
        self.assertIsNotNone(updated_alert.sent_at)

    def test_get_recent_detections(self):
        """Test retrieving recent detections."""
        # Add camera and detections
        camera = self.db_manager.add_camera(
            name="Test Camera", host="192.168.1.100", port=80
        )

        for i in range(5):
            self.db_manager.add_detection(
                camera_id=camera.id,
                frame_number=i,
                text=f"Text {i}",
                confidence=0.9,
            )

        # Get recent detections
        detections = self.db_manager.get_recent_detections(limit=3)

        self.assertEqual(len(detections), 3)

    def test_get_detection_stats(self):
        """Test getting detection statistics."""
        # Add camera and detections
        camera = self.db_manager.add_camera(
            name="Test Camera", host="192.168.1.100", port=80
        )

        self.db_manager.add_detection(
            camera_id=camera.id, frame_number=1, text="Test 1", confidence=0.8
        )
        self.db_manager.add_detection(
            camera_id=camera.id, frame_number=2, text="Test 2", confidence=0.9
        )

        # Get stats
        stats = self.db_manager.get_detection_stats(camera_id=camera.id)

        self.assertEqual(stats["total_detections"], 2)
        self.assertAlmostEqual(stats["average_confidence"], 0.85, places=2)


if __name__ == "__main__":
    unittest.main()
