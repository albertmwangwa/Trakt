"""
Tests for Database Module
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime

# Add src to path before import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import directly from the module file bypassing __init__.py
import importlib.util

spec = importlib.util.spec_from_file_location(
    "database", os.path.join(os.path.dirname(__file__), "..", "src", "database.py")
)
database = importlib.util.module_from_spec(spec)
spec.loader.exec_module(database)

DatabaseManager = database.DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database file
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.config = {"enabled": True, "path": self.db_path}
        self.db = DatabaseManager(self.config)

    def tearDown(self):
        """Tear down test fixtures."""
        self.db.close()
        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test database initialization."""
        self.assertTrue(self.db.enabled)
        self.assertEqual(self.db.db_path, self.db_path)
        self.assertTrue(os.path.exists(self.db_path))

    def test_initialization_disabled(self):
        """Test database initialization when disabled."""
        db = DatabaseManager({"enabled": False})
        self.assertFalse(db.enabled)

    def test_save_detection(self):
        """Test saving a single detection."""
        detection_id = self.db.save_detection(
            camera_id="cam_1",
            text="TEST123",
            confidence=0.95,
            bbox=[100, 100, 200, 150],
            frame_number=1,
            matched_pattern="[A-Z]+[0-9]+",
        )

        self.assertIsNotNone(detection_id)
        self.assertIsInstance(detection_id, int)

    def test_save_detections_batch(self):
        """Test saving multiple detections."""
        detections = [
            {"text": "ABC", "confidence": 0.9, "bbox": [0, 0, 50, 20]},
            {"text": "DEF", "confidence": 0.85, "bbox": [60, 0, 110, 20]},
            {"text": "123", "confidence": 0.95, "bbox": [120, 0, 170, 20]},
        ]

        count = self.db.save_detections_batch("cam_1", detections, frame_number=10)
        self.assertEqual(count, 3)

    def test_save_detections_batch_empty(self):
        """Test saving empty detections list."""
        count = self.db.save_detections_batch("cam_1", [], frame_number=1)
        self.assertEqual(count, 0)

    def test_get_detections(self):
        """Test querying detections."""
        # Save some detections
        self.db.save_detection("cam_1", "TEST1", 0.9)
        self.db.save_detection("cam_2", "TEST2", 0.85)
        self.db.save_detection("cam_1", "TEST3", 0.95)

        # Get all detections
        all_detections = self.db.get_detections()
        self.assertEqual(len(all_detections), 3)

        # Get detections for specific camera
        cam1_detections = self.db.get_detections(camera_id="cam_1")
        self.assertEqual(len(cam1_detections), 2)

    def test_get_detections_pagination(self):
        """Test pagination of detections."""
        # Save several detections
        for i in range(10):
            self.db.save_detection("cam_1", f"TEXT{i}", 0.9)

        # Get with limit
        detections = self.db.get_detections(limit=5)
        self.assertEqual(len(detections), 5)

        # Get with offset
        detections_offset = self.db.get_detections(limit=5, offset=5)
        self.assertEqual(len(detections_offset), 5)

    def test_save_alert(self):
        """Test saving an alert."""
        alert_id = self.db.save_alert(
            camera_id="cam_1",
            alert_type="pattern_match",
            pattern="[0-9]{4,}",
            detected_text="12345",
            confidence=0.92,
        )

        self.assertIsNotNone(alert_id)
        self.assertIsInstance(alert_id, int)

    def test_get_alerts(self):
        """Test querying alerts."""
        self.db.save_alert("cam_1", "pattern_match", "[0-9]+", "123", 0.9)
        self.db.save_alert("cam_2", "pattern_match", "[A-Z]+", "ABC", 0.85)

        # Get all alerts
        alerts = self.db.get_alerts()
        self.assertEqual(len(alerts), 2)

        # Get alerts for specific camera
        cam1_alerts = self.db.get_alerts(camera_id="cam_1")
        self.assertEqual(len(cam1_alerts), 1)

    def test_update_alert_status(self):
        """Test updating alert status."""
        alert_id = self.db.save_alert("cam_1", "pattern_match", pattern="test")

        # Update status
        success = self.db.update_alert_status(alert_id, "acknowledged")
        self.assertTrue(success)

        # Verify update
        alerts = self.db.get_alerts(status="acknowledged")
        self.assertEqual(len(alerts), 1)

    def test_camera_session(self):
        """Test camera session management."""
        # Start session
        session_id = self.db.start_camera_session("cam_1")
        self.assertIsNotNone(session_id)

        # End session
        success = self.db.end_camera_session(session_id, frame_count=100, detection_count=25)
        self.assertTrue(success)

    def test_get_statistics(self):
        """Test getting statistics."""
        # Add some data
        self.db.save_detection("cam_1", "TEST1", 0.9)
        self.db.save_detection("cam_1", "TEST2", 0.8)
        self.db.save_alert("cam_1", "pattern_match")

        stats = self.db.get_statistics()

        self.assertEqual(stats["total_detections"], 2)
        self.assertEqual(stats["total_alerts"], 1)
        self.assertIn("average_confidence", stats)

    def test_get_statistics_by_camera(self):
        """Test getting statistics filtered by camera."""
        self.db.save_detection("cam_1", "TEST1", 0.9)
        self.db.save_detection("cam_2", "TEST2", 0.8)

        stats = self.db.get_statistics(camera_id="cam_1")
        self.assertEqual(stats["total_detections"], 1)

    def test_disabled_database_operations(self):
        """Test that operations return None/empty when disabled."""
        db = DatabaseManager({"enabled": False})

        self.assertIsNone(db.save_detection("cam_1", "TEST", 0.9))
        self.assertEqual(db.save_detections_batch("cam_1", [{"text": "X"}]), 0)
        self.assertIsNone(db.save_alert("cam_1", "test"))
        self.assertEqual(db.get_detections(), [])
        self.assertEqual(db.get_alerts(), [])
        self.assertEqual(db.get_statistics(), {})


if __name__ == "__main__":
    unittest.main()
