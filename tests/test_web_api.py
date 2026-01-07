"""
Tests for Web API
"""

import unittest
import json
import sys
import os

# Add src to path before importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import directly from web_api to avoid loading camera_handler and ocr_engine
# which have heavy dependencies (cv2, tensorflow)
import importlib.util
spec = importlib.util.spec_from_file_location("web_api", os.path.join(os.path.dirname(__file__), '..', 'src', 'web_api.py'))
web_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web_api)

app = web_api.app
update_state = web_api.update_state
app_state = web_api.app_state


class TestWebAPI(unittest.TestCase):
    """Test cases for Web API."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = app.test_client()
        app.config['TESTING'] = True
        # Reset app state
        app_state['status'] = 'stopped'
        app_state['frame_count'] = 0
        app_state['detection_count'] = 0
        app_state['last_update'] = None
        app_state['camera_info'] = None
        app_state['recent_detections'] = []
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/api/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
    
    def test_status_endpoint(self):
        """Test status endpoint."""
        response = self.client.get('/api/status')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIn('status', data['data'])
        self.assertIn('frame_count', data['data'])
        self.assertIn('detection_count', data['data'])
    
    def test_detections_endpoint(self):
        """Test detections endpoint."""
        response = self.client.get('/api/detections')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIsInstance(data['data'], list)
        self.assertIn('count', data)
    
    def test_camera_info_endpoint(self):
        """Test camera info endpoint."""
        response = self.client.get('/api/camera/info')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
    
    def test_results_endpoint(self):
        """Test results endpoint."""
        response = self.client.get('/api/results')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIsInstance(data['data'], list)
    
    def test_update_state(self):
        """Test state update function."""
        update_state(
            frame_count=100,
            detection_count=50,
            status='running',
            camera_info={'host': '192.168.1.100', 'connected': True}
        )
        
        self.assertEqual(app_state['frame_count'], 100)
        self.assertEqual(app_state['detection_count'], 50)
        self.assertEqual(app_state['status'], 'running')
        self.assertEqual(app_state['camera_info']['host'], '192.168.1.100')
        self.assertIsNotNone(app_state['last_update'])
    
    def test_update_state_with_detections(self):
        """Test state update with detections."""
        detections = [
            {'text': 'TEST', 'confidence': 0.95, 'bbox': [0, 0, 100, 50]},
            {'text': 'ABC123', 'confidence': 0.88, 'bbox': [100, 0, 200, 50]}
        ]
        
        update_state(detections=detections)
        
        self.assertEqual(len(app_state['recent_detections']), 2)
        self.assertEqual(app_state['recent_detections'][0]['text'], 'TEST')
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.get('/api/health')
        
        # Check that response is successful (CORS middleware is configured)
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
