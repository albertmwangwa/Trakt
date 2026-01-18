"""
Tests for Alert System Module
"""

import os
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

# Add src to path before import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import directly from the module file bypassing __init__.py
import importlib.util

spec = importlib.util.spec_from_file_location(
    "alerts", os.path.join(os.path.dirname(__file__), "..", "src", "alerts.py")
)
alerts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alerts)

AlertHandler = alerts.AlertHandler
AlertManager = alerts.AlertManager
AlertPattern = alerts.AlertPattern
CallbackAlertHandler = alerts.CallbackAlertHandler
LoggingAlertHandler = alerts.LoggingAlertHandler
WebhookAlertHandler = alerts.WebhookAlertHandler


class TestAlertPattern(unittest.TestCase):
    """Test cases for AlertPattern class."""

    def test_initialization(self):
        """Test pattern initialization."""
        pattern = AlertPattern(
            name="test_pattern",
            pattern="[0-9]+",
            priority="high",
            enabled=True,
            cooldown_seconds=30,
        )

        self.assertEqual(pattern.name, "test_pattern")
        self.assertEqual(pattern.pattern, "[0-9]+")
        self.assertEqual(pattern.priority, "high")
        self.assertTrue(pattern.enabled)
        self.assertEqual(pattern.cooldown_seconds, 30)

    def test_matches(self):
        """Test pattern matching."""
        pattern = AlertPattern(name="numbers", pattern="[0-9]{3,}")

        self.assertTrue(pattern.matches("ABC123"))
        self.assertTrue(pattern.matches("12345"))
        self.assertFalse(pattern.matches("AB"))
        self.assertFalse(pattern.matches("12"))

    def test_matches_disabled(self):
        """Test that disabled patterns don't match."""
        pattern = AlertPattern(name="numbers", pattern="[0-9]+", enabled=False)

        self.assertFalse(pattern.matches("12345"))

    def test_can_trigger_no_cooldown(self):
        """Test triggering without cooldown."""
        pattern = AlertPattern(name="test", pattern=".*", cooldown_seconds=0)

        self.assertTrue(pattern.can_trigger("cam_1"))
        pattern.mark_triggered("cam_1")
        self.assertTrue(pattern.can_trigger("cam_1"))  # Still true, no cooldown

    def test_can_trigger_with_cooldown(self):
        """Test triggering with cooldown."""
        pattern = AlertPattern(name="test", pattern=".*", cooldown_seconds=60)

        self.assertTrue(pattern.can_trigger("cam_1"))
        pattern.mark_triggered("cam_1")
        self.assertFalse(pattern.can_trigger("cam_1"))  # Still in cooldown

        # Different camera should still be triggerable
        self.assertTrue(pattern.can_trigger("cam_2"))


class TestLoggingAlertHandler(unittest.TestCase):
    """Test cases for LoggingAlertHandler class."""

    def test_handle(self):
        """Test logging handler."""
        handler = LoggingAlertHandler("WARNING")

        alert = {
            "alert_type": "pattern_match",
            "camera_id": "cam_1",
            "pattern": "[0-9]+",
            "detected_text": "12345",
            "confidence": 0.95,
        }

        result = handler.handle(alert)
        self.assertTrue(result)


class TestCallbackAlertHandler(unittest.TestCase):
    """Test cases for CallbackAlertHandler class."""

    def test_handle_success(self):
        """Test callback handler success."""
        callback = Mock(return_value=True)
        handler = CallbackAlertHandler(callback)

        alert = {"text": "test"}
        result = handler.handle(alert)

        self.assertTrue(result)
        callback.assert_called_once_with(alert)

    def test_handle_failure(self):
        """Test callback handler with failing callback."""
        callback = Mock(side_effect=Exception("Test error"))
        handler = CallbackAlertHandler(callback)

        alert = {"text": "test"}
        result = handler.handle(alert)

        self.assertFalse(result)


class TestAlertManager(unittest.TestCase):
    """Test cases for AlertManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "enabled": True,
            "patterns": [
                {
                    "name": "license_plate",
                    "pattern": "[A-Z]{2,3}[0-9]{3,4}",
                    "priority": "high",
                    "enabled": True,
                },
                {
                    "name": "numbers",
                    "pattern": "[0-9]{4,}",
                    "priority": "normal",
                    "enabled": True,
                },
            ],
            "handlers": {"logging": {"enabled": True, "level": "WARNING"}},
        }

    def test_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager(self.config)

        self.assertTrue(manager.enabled)
        self.assertEqual(len(manager.patterns), 2)
        self.assertGreaterEqual(len(manager.handlers), 1)  # At least logging handler

    def test_initialization_disabled(self):
        """Test disabled alert manager."""
        manager = AlertManager({"enabled": False})

        self.assertFalse(manager.enabled)
        self.assertEqual(len(manager.patterns), 0)

    def test_add_pattern(self):
        """Test adding a pattern."""
        manager = AlertManager({"enabled": True})

        pattern = AlertPattern(name="custom", pattern="CUSTOM.*")
        manager.add_pattern(pattern)

        self.assertEqual(len(manager.patterns), 1)

    def test_remove_pattern(self):
        """Test removing a pattern."""
        manager = AlertManager(self.config)

        initial_count = len(manager.patterns)
        success = manager.remove_pattern("license_plate")

        self.assertTrue(success)
        self.assertEqual(len(manager.patterns), initial_count - 1)

    def test_remove_nonexistent_pattern(self):
        """Test removing a non-existent pattern."""
        manager = AlertManager(self.config)

        success = manager.remove_pattern("nonexistent")
        self.assertFalse(success)

    def test_check_and_alert_no_match(self):
        """Test check_and_alert with no matching patterns."""
        manager = AlertManager(self.config)

        detections = [{"text": "HELLO WORLD", "confidence": 0.9}]

        alerts = manager.check_and_alert("cam_1", detections)
        self.assertEqual(len(alerts), 0)

    def test_check_and_alert_with_match(self):
        """Test check_and_alert with matching patterns."""
        manager = AlertManager(self.config)

        detections = [
            {"text": "ABC123", "confidence": 0.95, "bbox": [0, 0, 100, 50]},
            {"text": "12345678", "confidence": 0.9},
        ]

        alerts = manager.check_and_alert("cam_1", detections)
        self.assertGreaterEqual(len(alerts), 1)

    def test_check_and_alert_disabled(self):
        """Test check_and_alert when disabled."""
        manager = AlertManager({"enabled": False})

        detections = [{"text": "12345678", "confidence": 0.9}]
        alerts = manager.check_and_alert("cam_1", detections)

        self.assertEqual(len(alerts), 0)

    def test_check_and_alert_empty_detections(self):
        """Test check_and_alert with empty detections."""
        manager = AlertManager(self.config)

        alerts = manager.check_and_alert("cam_1", [])
        self.assertEqual(len(alerts), 0)

    def test_get_patterns(self):
        """Test getting patterns as dictionaries."""
        manager = AlertManager(self.config)

        patterns = manager.get_patterns()
        self.assertEqual(len(patterns), 2)
        self.assertIsInstance(patterns[0], dict)
        self.assertIn("name", patterns[0])
        self.assertIn("pattern", patterns[0])

    def test_set_pattern_enabled(self):
        """Test enabling/disabling a pattern."""
        manager = AlertManager(self.config)

        success = manager.set_pattern_enabled("license_plate", False)
        self.assertTrue(success)

        patterns = manager.get_patterns()
        plate_pattern = next(p for p in patterns if p["name"] == "license_plate")
        self.assertFalse(plate_pattern["enabled"])

    def test_set_pattern_enabled_nonexistent(self):
        """Test enabling non-existent pattern."""
        manager = AlertManager(self.config)

        success = manager.set_pattern_enabled("nonexistent", False)
        self.assertFalse(success)

    def test_add_handler(self):
        """Test adding a custom handler."""
        manager = AlertManager({"enabled": True})

        callback = Mock(return_value=True)
        handler = CallbackAlertHandler(callback)
        manager.add_handler(handler)

        self.assertIn(handler, manager.handlers)


class TestWebhookAlertHandler(unittest.TestCase):
    """Test cases for WebhookAlertHandler class."""

    @patch.object(alerts.requests, "post")
    def test_handle_success(self, mock_post):
        """Test successful webhook call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        handler = WebhookAlertHandler("http://example.com/webhook")
        alert = {"alert_type": "test", "camera_id": "cam_1"}

        result = handler.handle(alert)
        self.assertTrue(result)

    @patch.object(alerts.requests, "post")
    def test_handle_failure(self, mock_post):
        """Test failed webhook call."""
        mock_post.side_effect = alerts.requests.RequestException("Connection refused")

        handler = WebhookAlertHandler("http://example.com/webhook")
        alert = {"alert_type": "test", "camera_id": "cam_1"}

        result = handler.handle(alert)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
