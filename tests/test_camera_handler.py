"""
Tests for ONVIF Camera Handler
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera_handler import ONVIFCameraHandler


class TestONVIFCameraHandler(unittest.TestCase):
    """Test cases for ONVIFCameraHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.host = "192.168.1.100"
        self.port = 80
        self.username = "admin"
        self.password = "password"
    
    def test_initialization(self):
        """Test camera handler initialization."""
        handler = ONVIFCameraHandler(
            self.host, self.port, self.username, self.password
        )
        
        self.assertEqual(handler.host, self.host)
        self.assertEqual(handler.port, self.port)
        self.assertEqual(handler.username, self.username)
        self.assertEqual(handler.password, self.password)
        self.assertIsNone(handler.camera)
        self.assertIsNone(handler.stream_uri)
    
    @patch('src.camera_handler.ONVIFCamera')
    def test_connect_success(self, mock_onvif):
        """Test successful camera connection."""
        handler = ONVIFCameraHandler(
            self.host, self.port, self.username, self.password
        )
        
        # Mock ONVIF camera
        mock_camera = Mock()
        mock_onvif.return_value = mock_camera
        mock_camera.create_media_service.return_value = Mock()
        
        result = handler.connect()
        
        self.assertTrue(result)
        self.assertIsNotNone(handler.camera)
        self.assertIsNotNone(handler.media_service)
    
    @patch('src.camera_handler.ONVIFCamera')
    def test_connect_failure(self, mock_onvif):
        """Test failed camera connection."""
        handler = ONVIFCameraHandler(
            self.host, self.port, self.username, self.password
        )
        
        # Mock connection failure
        mock_onvif.side_effect = Exception("Connection failed")
        
        result = handler.connect()
        
        self.assertFalse(result)
    
    def test_get_camera_info(self):
        """Test getting camera information."""
        handler = ONVIFCameraHandler(
            self.host, self.port, self.username, self.password
        )
        
        info = handler.get_camera_info()
        
        self.assertEqual(info['host'], self.host)
        self.assertEqual(info['port'], self.port)
        self.assertFalse(info['connected'])
        self.assertFalse(info['streaming'])


if __name__ == '__main__':
    unittest.main()
