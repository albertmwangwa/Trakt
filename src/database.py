"""
Database Module

This module provides database integration for storing OCR results,
camera information, and alert history.
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


def utcnow():
    """Return current UTC time."""
    return datetime.utcnow()


class Camera(Base):
    """Camera configuration and information."""

    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    host = Column(String(100), nullable=False)
    port = Column(Integer, nullable=False)
    username = Column(String(100))
    stream_profile = Column(Integer, default=0)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    detections = relationship("Detection", back_populates="camera")


class Detection(Base):
    """OCR detection results."""

    __tablename__ = "detections"

    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    matched_pattern = Column(String(200))
    timestamp = Column(DateTime, default=utcnow)

    # Relationships
    camera = relationship("Camera", back_populates="detections")
    alerts = relationship("Alert", back_populates="detection")


class Alert(Base):
    """Alert records for pattern matches."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=False)
    alert_type = Column(String(50), nullable=False)  # email, webhook, log
    pattern = Column(String(200), nullable=False)
    message = Column(Text)
    sent = Column(Boolean, default=False)
    sent_at = Column(DateTime)
    error_message = Column(Text)
    created_at = Column(DateTime, default=utcnow)

    # Relationships
    detection = relationship("Detection", back_populates="alerts")


class DatabaseManager:
    """Manager for database operations."""

    def __init__(self, database_url: str = "sqlite:///trakt.db"):
        """
        Initialize database manager.

        Args:
            database_url: SQLAlchemy database URL
        """
        self.logger = logging.getLogger(__name__)
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)
        self.logger.info(f"Database initialized: {database_url}")

    def get_session(self):
        """Get a new database session."""
        return self.SessionLocal()

    def add_camera(
        self,
        name: str,
        host: str,
        port: int,
        username: str = None,
        stream_profile: int = 0,
        active: bool = True,
    ) -> Optional[Camera]:
        """
        Add a new camera to the database.

        Args:
            name: Camera name
            host: Camera IP address
            port: Camera port
            username: Camera username
            stream_profile: Stream profile index
            active: Whether camera is active

        Returns:
            Camera object or None if failed
        """
        session = self.get_session()
        try:
            camera = Camera(
                name=name,
                host=host,
                port=port,
                username=username,
                stream_profile=stream_profile,
                active=active,
            )
            session.add(camera)
            session.commit()
            session.refresh(camera)
            self.logger.info(f"Added camera: {name}")
            return camera
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to add camera: {e}")
            return None
        finally:
            session.close()

    def get_camera_by_name(self, name: str) -> Optional[Camera]:
        """
        Get camera by name.

        Args:
            name: Camera name

        Returns:
            Camera object or None if not found
        """
        session = self.get_session()
        try:
            camera = session.query(Camera).filter_by(name=name).first()
            return camera
        finally:
            session.close()

    def get_active_cameras(self) -> List[Camera]:
        """
        Get all active cameras.

        Returns:
            List of active Camera objects
        """
        session = self.get_session()
        try:
            cameras = session.query(Camera).filter_by(active=True).all()
            return cameras
        finally:
            session.close()

    def add_detection(
        self,
        camera_id: int,
        frame_number: int,
        text: str,
        confidence: float,
        bbox: List[int] = None,
        matched_pattern: str = None,
    ) -> Optional[Detection]:
        """
        Add a detection result to the database.

        Args:
            camera_id: Camera ID
            frame_number: Frame number
            text: Detected text
            confidence: Detection confidence
            bbox: Bounding box [x1, y1, x2, y2]
            matched_pattern: Matched regex pattern

        Returns:
            Detection object or None if failed
        """
        session = self.get_session()
        try:
            detection = Detection(
                camera_id=camera_id,
                frame_number=frame_number,
                text=text,
                confidence=confidence,
                matched_pattern=matched_pattern,
            )

            if bbox and len(bbox) == 4:
                detection.bbox_x1 = bbox[0]
                detection.bbox_y1 = bbox[1]
                detection.bbox_x2 = bbox[2]
                detection.bbox_y2 = bbox[3]

            session.add(detection)
            session.commit()
            session.refresh(detection)
            return detection
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to add detection: {e}")
            return None
        finally:
            session.close()

    def add_alert(
        self,
        detection_id: int,
        alert_type: str,
        pattern: str,
        message: str = None,
    ) -> Optional[Alert]:
        """
        Add an alert record to the database.

        Args:
            detection_id: Detection ID
            alert_type: Type of alert (email, webhook, log)
            pattern: Pattern that triggered the alert
            message: Alert message

        Returns:
            Alert object or None if failed
        """
        session = self.get_session()
        try:
            alert = Alert(
                detection_id=detection_id,
                alert_type=alert_type,
                pattern=pattern,
                message=message,
            )
            session.add(alert)
            session.commit()
            session.refresh(alert)
            return alert
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to add alert: {e}")
            return None
        finally:
            session.close()

    def mark_alert_sent(
        self, alert_id: int, success: bool = True, error_message: str = None
    ):
        """
        Mark an alert as sent.

        Args:
            alert_id: Alert ID
            success: Whether alert was sent successfully
            error_message: Error message if failed
        """
        session = self.get_session()
        try:
            alert = session.query(Alert).filter_by(id=alert_id).first()
            if alert:
                alert.sent = success
                alert.sent_at = datetime.utcnow() if success else None
                alert.error_message = error_message
                session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to mark alert as sent: {e}")
        finally:
            session.close()

    def get_recent_detections(
        self, camera_id: int = None, limit: int = 100
    ) -> List[Detection]:
        """
        Get recent detections.

        Args:
            camera_id: Optional camera ID filter
            limit: Maximum number of results

        Returns:
            List of Detection objects
        """
        session = self.get_session()
        try:
            query = session.query(Detection).order_by(Detection.timestamp.desc())

            if camera_id:
                query = query.filter_by(camera_id=camera_id)

            detections = query.limit(limit).all()
            return detections
        finally:
            session.close()

    def get_detection_stats(self, camera_id: int = None) -> dict:
        """
        Get detection statistics.

        Args:
            camera_id: Optional camera ID filter

        Returns:
            Dictionary with statistics
        """
        session = self.get_session()
        try:
            query = session.query(Detection)

            if camera_id:
                query = query.filter_by(camera_id=camera_id)

            total = query.count()

            # Use the same filtered query for confidence
            avg_query = query.filter(Detection.confidence > 0)
            avg_confidence = [d.confidence for d in avg_query.all()]

            if avg_confidence:
                avg = sum(avg_confidence) / len(avg_confidence)
            else:
                avg = 0.0

            return {
                "total_detections": total,
                "average_confidence": avg,
            }
        finally:
            session.close()
