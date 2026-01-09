"""
ONVIF Camera Handler Module

This module handles connection and streaming from ONVIF-compliant cameras.
"""

import cv2
import numpy as np
from onvif import ONVIFCamera
import logging
from typing import Optional, Tuple
import time


class ONVIFCameraHandler:
    """Handler for ONVIF camera connections and streaming."""

    def __init__(self, host: str, port: int, username: str, password: str):
        """
        Initialize ONVIF camera handler.

        Args:
            host: Camera IP address
            port: ONVIF port (usually 80 or 8080)
            username: Camera username
            password: Camera password
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.camera = None
        self.media_service = None
        self.stream_uri = None
        self.capture = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> bool:
        """
        Connect to the ONVIF camera.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to ONVIF camera at {self.host}:{self.port}")

            # Create ONVIF camera instance
            self.camera = ONVIFCamera(
                self.host,
                self.port,
                self.username,
                self.password,
                "/etc/onvif/wsdl/",  # WSDL directory
            )

            # Get media service
            self.media_service = self.camera.create_media_service()

            self.logger.info("Successfully connected to ONVIF camera")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to ONVIF camera: {e}")
            return False

    def get_stream_uri(
        self, profile_token: int = 0, transport: str = "tcp"
    ) -> Optional[str]:
        """
        Get the RTSP stream URI from the camera.

        Args:
            profile_token: Profile index (0 for main stream, 1 for sub stream)
            transport: Transport protocol ('tcp' or 'udp')

        Returns:
            RTSP stream URI or None if failed
        """
        try:
            # Get available profiles
            profiles = self.media_service.GetProfiles()

            if not profiles or profile_token >= len(profiles):
                self.logger.error(f"Profile {profile_token} not available")
                return None

            token = profiles[profile_token].token

            # Get stream URI
            stream_setup = {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}}

            obj = self.media_service.create_type("GetStreamUri")
            obj.ProfileToken = token
            obj.StreamSetup = stream_setup

            response = self.media_service.GetStreamUri(obj)
            self.stream_uri = response.Uri

            self.logger.info(f"Stream URI: {self.stream_uri}")
            return self.stream_uri

        except Exception as e:
            self.logger.error(f"Failed to get stream URI: {e}")
            return None

    def start_stream(self) -> bool:
        """
        Start capturing video stream.

        Returns:
            True if stream started successfully, False otherwise
        """
        try:
            if not self.stream_uri:
                self.logger.error(
                    "No stream URI available. Call get_stream_uri() first."
                )
                return False

            self.logger.info("Starting video stream...")

            # OpenCV VideoCapture with RTSP
            self.capture = cv2.VideoCapture(self.stream_uri)

            if not self.capture.isOpened():
                self.logger.error("Failed to open video stream")
                return False

            self.logger.info("Video stream started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start stream: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the video stream.

        Returns:
            Tuple of (success, frame)
        """
        if self.capture is None or not self.capture.isOpened():
            return False, None

        ret, frame = self.capture.read()
        return ret, frame

    def release(self):
        """Release camera resources."""
        if self.capture is not None:
            self.capture.release()
            self.logger.info("Camera resources released")

    def get_camera_info(self) -> dict:
        """
        Get camera information.

        Returns:
            Dictionary containing camera information
        """
        info = {
            "host": self.host,
            "port": self.port,
            "connected": self.camera is not None,
            "streaming": self.capture is not None and self.capture.isOpened(),
        }

        if self.camera:
            try:
                device_info = self.camera.devicemgmt.GetDeviceInformation()
                info.update(
                    {
                        "manufacturer": device_info.Manufacturer,
                        "model": device_info.Model,
                        "firmware_version": device_info.FirmwareVersion,
                        "serial_number": device_info.SerialNumber,
                    }
                )
            except Exception as e:
                self.logger.warning(f"Could not retrieve device information: {e}")

        return info
