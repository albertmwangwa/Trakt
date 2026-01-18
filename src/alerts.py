"""
Alert System Module

This module provides an alert system for detecting specific text patterns
and triggering notifications through various handlers.
"""

import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Optional

import requests


class AlertHandler(ABC):
    """Abstract base class for alert handlers."""

    @abstractmethod
    def handle(self, alert: Dict) -> bool:
        """
        Handle an alert.

        Args:
            alert: Alert dictionary with detection details

        Returns:
            True if alert was handled successfully
        """
        pass


class LoggingAlertHandler(AlertHandler):
    """Alert handler that logs alerts to the application log."""

    def __init__(self, log_level: str = "WARNING"):
        """
        Initialize logging alert handler.

        Args:
            log_level: Logging level for alerts
        """
        self.logger = logging.getLogger(__name__)
        self.log_level = getattr(logging, log_level.upper(), logging.WARNING)

    def handle(self, alert: Dict) -> bool:
        """Log the alert."""
        message = (
            f"ALERT [{alert.get('alert_type', 'unknown')}] "
            f"Camera: {alert.get('camera_id', 'unknown')} | "
            f"Pattern: {alert.get('pattern', 'N/A')} | "
            f"Text: '{alert.get('detected_text', '')}' | "
            f"Confidence: {alert.get('confidence', 0):.2f}"
        )
        self.logger.log(self.log_level, message)
        return True


class WebhookAlertHandler(AlertHandler):
    """Alert handler that sends alerts to a webhook URL."""

    def __init__(self, url: str, headers: Dict = None, timeout: int = 10):
        """
        Initialize webhook alert handler.

        Args:
            url: Webhook URL to send alerts to
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
        """
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def handle(self, alert: Dict) -> bool:
        """Send alert to webhook."""
        try:
            payload = {
                "timestamp": alert.get("timestamp", datetime.now().isoformat()),
                "alert_type": alert.get("alert_type", "pattern_match"),
                "camera_id": alert.get("camera_id", "unknown"),
                "pattern": alert.get("pattern"),
                "detected_text": alert.get("detected_text"),
                "confidence": alert.get("confidence"),
                "bbox": alert.get("bbox"),
            }

            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                self.logger.debug(f"Alert sent to webhook: {self.url}")
                return True
            else:
                self.logger.warning(
                    f"Webhook returned status {response.status_code}"
                )
                return False
        except requests.RequestException as e:
            self.logger.error(f"Failed to send alert to webhook: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending webhook alert: {e}")
            return False


class CallbackAlertHandler(AlertHandler):
    """Alert handler that calls a user-defined callback function."""

    def __init__(self, callback: Callable[[Dict], bool]):
        """
        Initialize callback alert handler.

        Args:
            callback: Callback function to call with alert data
        """
        self.callback = callback
        self.logger = logging.getLogger(__name__)

    def handle(self, alert: Dict) -> bool:
        """Call the callback with alert data."""
        try:
            return self.callback(alert)
        except Exception as e:
            self.logger.error(f"Callback alert handler error: {e}")
            return False


class AlertPattern:
    """Represents an alert pattern configuration."""

    def __init__(
        self,
        name: str,
        pattern: str,
        priority: str = "normal",
        enabled: bool = True,
        cooldown_seconds: int = 0,
    ):
        """
        Initialize alert pattern.

        Args:
            name: Pattern name/identifier
            pattern: Regex pattern to match
            priority: Alert priority (low, normal, high, critical)
            enabled: Whether pattern is enabled
            cooldown_seconds: Minimum seconds between alerts for same pattern
        """
        self.name = name
        self.pattern = pattern
        self.regex = re.compile(pattern)
        self.priority = priority
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered: Dict[str, datetime] = {}  # camera_id -> last trigger time

    def matches(self, text: str) -> bool:
        """Check if text matches the pattern."""
        return self.enabled and self.regex.search(text) is not None

    def can_trigger(self, camera_id: str) -> bool:
        """Check if pattern can trigger based on cooldown."""
        if self.cooldown_seconds <= 0:
            return True

        last = self.last_triggered.get(camera_id)
        if last is None:
            return True

        elapsed = (datetime.now() - last).total_seconds()
        return elapsed >= self.cooldown_seconds

    def mark_triggered(self, camera_id: str):
        """Mark pattern as triggered for cooldown tracking."""
        self.last_triggered[camera_id] = datetime.now()


class AlertManager:
    """Manager for the alert system."""

    def __init__(self, config: dict = None, database=None):
        """
        Initialize alert manager.

        Args:
            config: Alert configuration dictionary
            database: Optional DatabaseManager instance for storing alerts
        """
        self.config = config or {}
        self.database = database
        self.enabled = self.config.get("enabled", True)
        self.logger = logging.getLogger(__name__)

        self.patterns: List[AlertPattern] = []
        self.handlers: List[AlertHandler] = []
        self._lock = threading.Lock()

        if self.enabled:
            self._initialize_patterns()
            self._initialize_handlers()

    def _initialize_patterns(self):
        """Initialize alert patterns from configuration."""
        patterns_config = self.config.get("patterns", [])

        for p in patterns_config:
            try:
                pattern = AlertPattern(
                    name=p.get("name", p.get("pattern", "unknown")),
                    pattern=p.get("pattern"),
                    priority=p.get("priority", "normal"),
                    enabled=p.get("enabled", True),
                    cooldown_seconds=p.get("cooldown_seconds", 0),
                )
                self.patterns.append(pattern)
                self.logger.debug(f"Registered alert pattern: {pattern.name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize pattern {p}: {e}")

        self.logger.info(f"Initialized {len(self.patterns)} alert patterns")

    def _initialize_handlers(self):
        """Initialize alert handlers from configuration."""
        handlers_config = self.config.get("handlers", {})

        # Always add logging handler if enabled
        if handlers_config.get("logging", {}).get("enabled", True):
            log_level = handlers_config.get("logging", {}).get("level", "WARNING")
            self.handlers.append(LoggingAlertHandler(log_level))
            self.logger.debug("Added logging alert handler")

        # Add webhook handler if configured
        webhook_config = handlers_config.get("webhook", {})
        if webhook_config.get("enabled", False) and webhook_config.get("url"):
            self.handlers.append(
                WebhookAlertHandler(
                    url=webhook_config["url"],
                    headers=webhook_config.get("headers"),
                    timeout=webhook_config.get("timeout", 10),
                )
            )
            self.logger.debug(f"Added webhook alert handler: {webhook_config['url']}")

        self.logger.info(f"Initialized {len(self.handlers)} alert handlers")

    def add_pattern(self, pattern: AlertPattern):
        """Add an alert pattern."""
        with self._lock:
            self.patterns.append(pattern)
            self.logger.debug(f"Added alert pattern: {pattern.name}")

    def remove_pattern(self, name: str) -> bool:
        """Remove an alert pattern by name."""
        with self._lock:
            for i, p in enumerate(self.patterns):
                if p.name == name:
                    del self.patterns[i]
                    self.logger.debug(f"Removed alert pattern: {name}")
                    return True
        return False

    def add_handler(self, handler: AlertHandler):
        """Add an alert handler."""
        with self._lock:
            self.handlers.append(handler)

    def check_and_alert(
        self,
        camera_id: str,
        detections: List[Dict],
        frame_number: int = None,
    ) -> List[Dict]:
        """
        Check detections against alert patterns and trigger alerts.

        Args:
            camera_id: Camera identifier
            detections: List of OCR detection results
            frame_number: Current frame number

        Returns:
            List of triggered alerts
        """
        if not self.enabled or not detections:
            return []

        triggered_alerts = []

        with self._lock:
            for detection in detections:
                text = detection.get("text", "")
                confidence = detection.get("confidence", 0)
                bbox = detection.get("bbox")

                for pattern in self.patterns:
                    if not pattern.enabled:
                        continue

                    if pattern.matches(text) and pattern.can_trigger(camera_id):
                        alert = {
                            "timestamp": datetime.now().isoformat(),
                            "alert_type": "pattern_match",
                            "camera_id": camera_id,
                            "pattern": pattern.pattern,
                            "pattern_name": pattern.name,
                            "priority": pattern.priority,
                            "detected_text": text,
                            "confidence": confidence,
                            "bbox": bbox,
                            "frame_number": frame_number,
                        }

                        # Trigger all handlers
                        for handler in self.handlers:
                            try:
                                handler.handle(alert)
                            except Exception as e:
                                self.logger.error(f"Alert handler error: {e}")

                        # Save to database if available
                        if self.database:
                            self.database.save_alert(
                                camera_id=camera_id,
                                alert_type="pattern_match",
                                pattern=pattern.pattern,
                                detected_text=text,
                                confidence=confidence,
                            )

                        # Mark pattern as triggered for cooldown
                        pattern.mark_triggered(camera_id)
                        triggered_alerts.append(alert)

        return triggered_alerts

    def get_patterns(self) -> List[Dict]:
        """Get all registered patterns as dictionaries."""
        with self._lock:
            return [
                {
                    "name": p.name,
                    "pattern": p.pattern,
                    "priority": p.priority,
                    "enabled": p.enabled,
                    "cooldown_seconds": p.cooldown_seconds,
                }
                for p in self.patterns
            ]

    def set_pattern_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a pattern by name."""
        with self._lock:
            for p in self.patterns:
                if p.name == name:
                    p.enabled = enabled
                    return True
        return False

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """
        Get recent alerts from the database.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alert records
        """
        if self.database:
            return self.database.get_alerts(limit=limit)
        return []
