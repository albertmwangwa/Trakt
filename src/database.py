"""
Database Module for OCR Results Storage

This module provides database integration for storing OCR detection results
using SQLite as the default database backend.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional


class DatabaseManager:
    """Manager for database operations with SQLite backend."""

    def __init__(self, config: dict = None):
        """
        Initialize database manager.

        Args:
            config: Database configuration dictionary
        """
        self.config = config or {}
        self.db_path = self.config.get("path", "./output/trakt.db")
        self.enabled = self.config.get("enabled", True)
        self.logger = logging.getLogger(__name__)
        self._local = threading.local()

        if self.enabled:
            self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursor."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()

    def _initialize_database(self):
        """Initialize database tables."""
        try:
            import os

            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            with self._get_cursor() as cursor:
                # Create detections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        frame_number INTEGER,
                        text TEXT NOT NULL,
                        confidence REAL,
                        bbox TEXT,
                        matched_pattern TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create alerts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        pattern TEXT,
                        detected_text TEXT,
                        confidence REAL,
                        status TEXT DEFAULT 'pending',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create camera_sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS camera_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        frame_count INTEGER DEFAULT 0,
                        detection_count INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'active'
                    )
                """)

                # Create indexes for common queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_detections_camera_id
                    ON detections(camera_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_detections_timestamp
                    ON detections(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_status
                    ON alerts(status)
                """)

            self.logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            self.enabled = False

    def save_detection(
        self,
        camera_id: str,
        text: str,
        confidence: float,
        bbox: List[int] = None,
        frame_number: int = None,
        matched_pattern: str = None,
    ) -> Optional[int]:
        """
        Save a detection result to the database.

        Args:
            camera_id: Camera identifier
            text: Detected text
            confidence: Detection confidence score
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            frame_number: Frame number in which detection occurred
            matched_pattern: Regex pattern that matched (if any)

        Returns:
            ID of the inserted record, or None if failed
        """
        if not self.enabled:
            return None

        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO detections
                    (camera_id, timestamp, frame_number, text, confidence, bbox, matched_pattern)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        camera_id,
                        datetime.now().isoformat(),
                        frame_number,
                        text,
                        confidence,
                        json.dumps(bbox) if bbox else None,
                        matched_pattern,
                    ),
                )
                return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Failed to save detection: {e}")
            return None

    def save_detections_batch(
        self, camera_id: str, detections: List[Dict], frame_number: int = None
    ) -> int:
        """
        Save multiple detection results to the database.

        Args:
            camera_id: Camera identifier
            detections: List of detection dictionaries
            frame_number: Frame number in which detections occurred

        Returns:
            Number of records inserted
        """
        if not self.enabled or not detections:
            return 0

        try:
            timestamp = datetime.now().isoformat()
            with self._get_cursor() as cursor:
                records = [
                    (
                        camera_id,
                        timestamp,
                        frame_number,
                        d.get("text", ""),
                        d.get("confidence", 0.0),
                        json.dumps(d.get("bbox")) if d.get("bbox") else None,
                        d.get("matched_pattern"),
                    )
                    for d in detections
                ]
                cursor.executemany(
                    """
                    INSERT INTO detections
                    (camera_id, timestamp, frame_number, text, confidence, bbox, matched_pattern)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    records,
                )
                return len(records)
        except Exception as e:
            self.logger.error(f"Failed to save detections batch: {e}")
            return 0

    def save_alert(
        self,
        camera_id: str,
        alert_type: str,
        pattern: str = None,
        detected_text: str = None,
        confidence: float = None,
    ) -> Optional[int]:
        """
        Save an alert to the database.

        Args:
            camera_id: Camera identifier
            alert_type: Type of alert (e.g., 'pattern_match', 'threshold')
            pattern: Pattern that triggered the alert
            detected_text: Text that triggered the alert
            confidence: Detection confidence

        Returns:
            ID of the inserted record, or None if failed
        """
        if not self.enabled:
            return None

        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO alerts
                    (camera_id, timestamp, alert_type, pattern, detected_text, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        camera_id,
                        datetime.now().isoformat(),
                        alert_type,
                        pattern,
                        detected_text,
                        confidence,
                    ),
                )
                return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Failed to save alert: {e}")
            return None

    def get_detections(
        self,
        camera_id: str = None,
        start_time: str = None,
        end_time: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Query detection results from the database.

        Args:
            camera_id: Filter by camera ID (optional)
            start_time: Filter by start time (ISO format)
            end_time: Filter by end time (ISO format)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of detection records
        """
        if not self.enabled:
            return []

        try:
            query = "SELECT * FROM detections WHERE 1=1"
            params = []

            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            with self._get_cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to query detections: {e}")
            return []

    def get_alerts(
        self,
        camera_id: str = None,
        status: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Query alerts from the database.

        Args:
            camera_id: Filter by camera ID (optional)
            status: Filter by status (optional)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of alert records
        """
        if not self.enabled:
            return []

        try:
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []

            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            with self._get_cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to query alerts: {e}")
            return []

    def update_alert_status(self, alert_id: int, status: str) -> bool:
        """
        Update the status of an alert.

        Args:
            alert_id: Alert ID
            status: New status

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    "UPDATE alerts SET status = ? WHERE id = ?", (status, alert_id)
                )
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Failed to update alert status: {e}")
            return False

    def start_camera_session(self, camera_id: str) -> Optional[int]:
        """
        Start a new camera session.

        Args:
            camera_id: Camera identifier

        Returns:
            Session ID or None if failed
        """
        if not self.enabled:
            return None

        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO camera_sessions (camera_id, start_time, status)
                    VALUES (?, ?, 'active')
                """,
                    (camera_id, datetime.now().isoformat()),
                )
                return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Failed to start camera session: {e}")
            return None

    def end_camera_session(
        self, session_id: int, frame_count: int = 0, detection_count: int = 0
    ) -> bool:
        """
        End a camera session.

        Args:
            session_id: Session ID
            frame_count: Total frames processed
            detection_count: Total detections made

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE camera_sessions
                    SET end_time = ?, frame_count = ?, detection_count = ?, status = 'completed'
                    WHERE id = ?
                """,
                    (datetime.now().isoformat(), frame_count, detection_count, session_id),
                )
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Failed to end camera session: {e}")
            return False

    def get_statistics(self, camera_id: str = None) -> Dict:
        """
        Get detection statistics.

        Args:
            camera_id: Filter by camera ID (optional)

        Returns:
            Dictionary with statistics
        """
        if not self.enabled:
            return {}

        try:
            stats = {}
            with self._get_cursor() as cursor:
                # Total detections
                if camera_id:
                    cursor.execute(
                        "SELECT COUNT(*) FROM detections WHERE camera_id = ?",
                        (camera_id,),
                    )
                else:
                    cursor.execute("SELECT COUNT(*) FROM detections")
                stats["total_detections"] = cursor.fetchone()[0]

                # Total alerts
                if camera_id:
                    cursor.execute(
                        "SELECT COUNT(*) FROM alerts WHERE camera_id = ?", (camera_id,)
                    )
                else:
                    cursor.execute("SELECT COUNT(*) FROM alerts")
                stats["total_alerts"] = cursor.fetchone()[0]

                # Pending alerts
                if camera_id:
                    cursor.execute(
                        "SELECT COUNT(*) FROM alerts WHERE camera_id = ? AND status = 'pending'",
                        (camera_id,),
                    )
                else:
                    cursor.execute(
                        "SELECT COUNT(*) FROM alerts WHERE status = 'pending'"
                    )
                stats["pending_alerts"] = cursor.fetchone()[0]

                # Average confidence
                if camera_id:
                    cursor.execute(
                        "SELECT AVG(confidence) FROM detections WHERE camera_id = ?",
                        (camera_id,),
                    )
                else:
                    cursor.execute("SELECT AVG(confidence) FROM detections")
                avg_conf = cursor.fetchone()[0]
                stats["average_confidence"] = round(avg_conf, 3) if avg_conf else 0.0

            return stats
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}

    def close(self):
        """Close database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
            self.logger.info("Database connection closed")
