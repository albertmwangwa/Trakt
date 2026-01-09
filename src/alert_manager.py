"""
Alert Manager Module

This module provides alert functionality for specific text pattern matches.
Supports email, webhook, and log-based alerts.
"""

import json
import logging
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List
from datetime import datetime

import requests


class AlertManager:
    """Manager for handling alerts when specific text patterns are detected."""

    def __init__(self, config: dict, database_manager=None):
        """
        Initialize alert manager.

        Args:
            config: Alert configuration dictionary
            database_manager: Optional database manager for storing alerts
        """
        self.config = config
        self.database_manager = database_manager
        self.logger = logging.getLogger(__name__)

        # Alert configuration
        self.enabled = config.get("enabled", False)
        self.alert_patterns = config.get("patterns", [])
        self.alert_types = config.get("alert_types", ["log"])

        # Email configuration
        self.email_config = config.get("email", {})

        # Webhook configuration
        self.webhook_config = config.get("webhook", {})

        # Alert cooldown to prevent spam
        self.cooldown_seconds = config.get("cooldown_seconds", 60)
        self.last_alert_time = {}

        if self.enabled:
            self.logger.info(
                f"Alert manager initialized with {len(self.alert_patterns)} patterns"
            )

    def check_text(
        self, text: str, camera_name: str = None, detection_id: int = None
    ) -> bool:
        """
        Check if text matches any alert patterns.

        Args:
            text: Text to check
            camera_name: Name of camera that detected the text
            detection_id: Optional detection ID for database reference

        Returns:
            True if alert was triggered, False otherwise
        """
        if not self.enabled or not self.alert_patterns:
            return False

        for pattern in self.alert_patterns:
            pattern_config = pattern if isinstance(pattern, dict) else {"pattern": pattern}
            regex_pattern = pattern_config.get("pattern")
            pattern_name = pattern_config.get("name", regex_pattern)

            if re.search(regex_pattern, text):
                # Check cooldown
                cooldown_key = f"{camera_name}:{pattern_name}:{text}"
                if self._is_in_cooldown(cooldown_key):
                    self.logger.debug(
                        f"Alert for pattern '{pattern_name}' is in cooldown period"
                    )
                    return False

                self.logger.info(
                    f"Text '{text}' matched alert pattern: {pattern_name}"
                )

                # Trigger alerts
                self._trigger_alerts(text, pattern_name, camera_name, detection_id)

                # Update cooldown
                self.last_alert_time[cooldown_key] = datetime.utcnow()

                return True

        return False

    def _is_in_cooldown(self, key: str) -> bool:
        """
        Check if alert is in cooldown period.

        Args:
            key: Cooldown key

        Returns:
            True if in cooldown, False otherwise
        """
        if key not in self.last_alert_time:
            return False

        elapsed = (datetime.utcnow() - self.last_alert_time[key]).total_seconds()
        return elapsed < self.cooldown_seconds

    def _trigger_alerts(
        self,
        text: str,
        pattern: str,
        camera_name: str = None,
        detection_id: int = None,
    ):
        """
        Trigger all configured alert types.

        Args:
            text: Detected text
            pattern: Matched pattern
            camera_name: Camera name
            detection_id: Detection ID
        """
        alert_data = {
            "text": text,
            "pattern": pattern,
            "camera": camera_name,
            "timestamp": datetime.utcnow().isoformat(),
        }

        for alert_type in self.alert_types:
            try:
                if alert_type == "log":
                    self._send_log_alert(alert_data)
                elif alert_type == "email":
                    self._send_email_alert(alert_data)
                elif alert_type == "webhook":
                    self._send_webhook_alert(alert_data)

                # Store alert in database
                if self.database_manager and detection_id:
                    self.database_manager.add_alert(
                        detection_id=detection_id,
                        alert_type=alert_type,
                        pattern=pattern,
                        message=json.dumps(alert_data),
                    )

            except Exception as e:
                self.logger.error(f"Failed to send {alert_type} alert: {e}")

    def _send_log_alert(self, alert_data: dict):
        """
        Send log-based alert.

        Args:
            alert_data: Alert data dictionary
        """
        self.logger.warning(
            f"🚨 ALERT: Pattern '{alert_data['pattern']}' detected! "
            f"Text: '{alert_data['text']}' "
            f"Camera: {alert_data.get('camera', 'Unknown')} "
            f"Time: {alert_data['timestamp']}"
        )

    def _send_email_alert(self, alert_data: dict):
        """
        Send email alert.

        Args:
            alert_data: Alert data dictionary
        """
        if not self.email_config.get("enabled", False):
            return

        smtp_server = self.email_config.get("smtp_server")
        smtp_port = self.email_config.get("smtp_port", 587)
        username = self.email_config.get("username")
        password = self.email_config.get("password")
        from_email = self.email_config.get("from_email", username)
        to_emails = self.email_config.get("to_emails", [])

        if not all([smtp_server, username, password, to_emails]):
            self.logger.warning("Email configuration incomplete, skipping email alert")
            return

        # Create email
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = ", ".join(to_emails)
        msg["Subject"] = f"Trakt Alert: Pattern '{alert_data['pattern']}' Detected"

        body = f"""
        Alert Notification from Trakt OCR System

        Pattern Matched: {alert_data['pattern']}
        Detected Text: {alert_data['text']}
        Camera: {alert_data.get('camera', 'Unknown')}
        Timestamp: {alert_data['timestamp']}

        This is an automated alert from your Trakt OCR monitoring system.
        """

        msg.attach(MIMEText(body, "plain"))

        # Send email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)

            self.logger.info(f"Email alert sent to {to_emails}")

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
            raise

    def _send_webhook_alert(self, alert_data: dict):
        """
        Send webhook alert.

        Args:
            alert_data: Alert data dictionary
        """
        if not self.webhook_config.get("enabled", False):
            return

        url = self.webhook_config.get("url")
        method = self.webhook_config.get("method", "POST").upper()
        headers = self.webhook_config.get("headers", {"Content-Type": "application/json"})
        timeout = self.webhook_config.get("timeout", 10)

        if not url:
            self.logger.warning("Webhook URL not configured, skipping webhook alert")
            return

        # Send webhook
        try:
            if method == "POST":
                response = requests.post(
                    url, json=alert_data, headers=headers, timeout=timeout
                )
            elif method == "GET":
                response = requests.get(
                    url, params=alert_data, headers=headers, timeout=timeout
                )
            else:
                self.logger.error(f"Unsupported webhook method: {method}")
                return

            response.raise_for_status()
            self.logger.info(f"Webhook alert sent to {url}")

        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            raise

    def get_alert_history(
        self, camera_id: int = None, limit: int = 50
    ) -> List[Dict]:
        """
        Get alert history from database.

        Args:
            camera_id: Optional camera ID filter
            limit: Maximum number of results

        Returns:
            List of alert records
        """
        if not self.database_manager:
            return []

        # This would query the database for alert history
        # Implementation depends on database schema
        return []
