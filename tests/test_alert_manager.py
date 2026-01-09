"""
Tests for Alert Manager Module
"""

import unittest
from unittest.mock import Mock, patch
import time

from src.alert_manager import AlertManager


class TestAlertManager(unittest.TestCase):
    """Test cases for AlertManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "enabled": True,
            "patterns": [
                {"pattern": "[A-Z]{3}[0-9]{3}", "name": "license_plate"},
                {"pattern": "EMERGENCY", "name": "emergency"},
            ],
            "alert_types": ["log"],
            "cooldown_seconds": 2,
        }
        self.alert_manager = AlertManager(self.config)

    def test_initialization(self):
        """Test alert manager initialization."""
        self.assertTrue(self.alert_manager.enabled)
        self.assertEqual(len(self.alert_manager.alert_patterns), 2)
        self.assertEqual(self.alert_manager.cooldown_seconds, 2)

    def test_check_text_match(self):
        """Test text matching against patterns."""
        # Should match license plate pattern
        result = self.alert_manager.check_text("ABC123", camera_name="Test Camera")
        self.assertTrue(result)

    def test_check_text_no_match(self):
        """Test text not matching any patterns."""
        result = self.alert_manager.check_text("NoMatch", camera_name="Test Camera")
        self.assertFalse(result)

    def test_cooldown_period(self):
        """Test alert cooldown functionality."""
        # First alert should succeed
        result1 = self.alert_manager.check_text("ABC123", camera_name="Test Camera")
        self.assertTrue(result1)

        # Immediate second alert should be blocked by cooldown
        result2 = self.alert_manager.check_text("ABC123", camera_name="Test Camera")
        self.assertFalse(result2)

        # Wait for cooldown to expire
        time.sleep(2.1)

        # Third alert should succeed after cooldown
        result3 = self.alert_manager.check_text("ABC123", camera_name="Test Camera")
        self.assertTrue(result3)

    def test_disabled_alerts(self):
        """Test that alerts don't trigger when disabled."""
        config = self.config.copy()
        config["enabled"] = False
        alert_manager = AlertManager(config)

        result = alert_manager.check_text("ABC123", camera_name="Test Camera")
        self.assertFalse(result)

    @patch("src.alert_manager.smtplib.SMTP")
    def test_email_alert(self, mock_smtp):
        """Test email alert sending."""
        config = self.config.copy()
        config["alert_types"] = ["email"]
        config["email"] = {
            "enabled": True,
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "username": "test@test.com",
            "password": "password",
            "from_email": "test@test.com",
            "to_emails": ["recipient@test.com"],
        }

        alert_manager = AlertManager(config)

        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Trigger alert
        result = alert_manager.check_text("ABC123", camera_name="Test Camera")

        self.assertTrue(result)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()

    @patch("src.alert_manager.requests.post")
    def test_webhook_alert(self, mock_post):
        """Test webhook alert sending."""
        config = self.config.copy()
        config["alert_types"] = ["webhook"]
        config["webhook"] = {
            "enabled": True,
            "url": "https://test.com/webhook",
            "method": "POST",
            "timeout": 10,
        }

        alert_manager = AlertManager(config)

        # Mock successful webhook response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Trigger alert
        result = alert_manager.check_text("ABC123", camera_name="Test Camera")

        self.assertTrue(result)
        mock_post.assert_called_once()

    def test_log_alert(self):
        """Test log alert functionality."""
        with self.assertLogs(level="WARNING") as log:
            self.alert_manager.check_text("ABC123", camera_name="Test Camera")

        self.assertTrue(any("ALERT" in message for message in log.output))

    def test_database_integration(self):
        """Test database integration for alerts."""
        mock_db = Mock()
        mock_db.add_alert.return_value = Mock(id=1)

        config = self.config.copy()
        alert_manager = AlertManager(config, database_manager=mock_db)

        # Trigger alert with detection ID
        alert_manager.check_text(
            "ABC123", camera_name="Test Camera", detection_id=1
        )

        # Verify database was called
        mock_db.add_alert.assert_called()


if __name__ == "__main__":
    unittest.main()
