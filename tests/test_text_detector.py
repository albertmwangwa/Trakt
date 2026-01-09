"""
Tests for Text Detection Module
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add src to path to import text_detector directly without __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Import directly from the module file bypassing __init__.py
import text_detector
from text_detector import EASTTextDetector, TextRegionPreprocessor


class TestTextRegionPreprocessor(unittest.TestCase):
    """Test cases for TextRegionPreprocessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextRegionPreprocessor()

    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsNotNone(self.preprocessor)

    def test_deskew(self):
        """Test image deskewing."""
        # Create a dummy image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = self.preprocessor.deskew(image)

        # Check output has same shape
        self.assertEqual(result.shape, image.shape)

    def test_enhance_contrast(self):
        """Test contrast enhancement."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = self.preprocessor.enhance_contrast(image)

        # Check output has same shape
        self.assertEqual(result.shape, image.shape)

    def test_remove_noise(self):
        """Test noise removal."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = self.preprocessor.remove_noise(image)

        # Check output has same shape
        self.assertEqual(result.shape, image.shape)

    def test_normalize_illumination(self):
        """Test illumination normalization."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = self.preprocessor.normalize_illumination(image)

        # Check output has same shape
        self.assertEqual(result.shape, image.shape)

    def test_preprocess_text_region_minimal(self):
        """Test minimal preprocessing (default)."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = self.preprocessor.preprocess_text_region(image, apply_all=False)

        # Check output has same shape
        self.assertEqual(result.shape, image.shape)

    def test_preprocess_text_region_full(self):
        """Test full preprocessing."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = self.preprocessor.preprocess_text_region(image, apply_all=True)

        # Check output has same shape
        self.assertEqual(result.shape, image.shape)

    def test_order_points(self):
        """Test point ordering."""
        # Create random points
        pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype="float32")

        result = self.preprocessor._order_points(pts)

        # Check we get 4 points back
        self.assertEqual(result.shape, (4, 2))


class TestEASTTextDetector(unittest.TestCase):
    """Test cases for EASTTextDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "confidence_threshold": 0.5,
            "nms_threshold": 0.4,
            "input_width": 320,
            "input_height": 320,
        }

    def test_initialization_without_model(self):
        """Test detector initialization without model path."""
        detector = EASTTextDetector(config=self.config)

        self.assertIsNone(detector.net)
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertEqual(detector.input_width, 320)

    def test_initialization_with_nonexistent_model(self):
        """Test detector initialization with non-existent model."""
        detector = EASTTextDetector(
            model_path="/nonexistent/model.pb", config=self.config
        )

        # Should not crash, but model should not load
        self.assertIsNone(detector.net)

    def test_detect_without_model(self):
        """Test detection without loaded model."""
        detector = EASTTextDetector(config=self.config)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector.detect(image)

        # Should return empty list when no model loaded
        self.assertEqual(result, [])

    def test_apply_nms_empty(self):
        """Test NMS with empty boxes."""
        detector = EASTTextDetector(config=self.config)

        result = detector._apply_nms([], [], 0.4)

        self.assertEqual(result, [])

    def test_apply_nms_with_boxes(self):
        """Test NMS with boxes."""
        detector = EASTTextDetector(config=self.config)
        boxes = [(10, 10, 50, 50), (15, 15, 55, 55), (100, 100, 150, 150)]
        confidences = [0.9, 0.8, 0.95]

        result = detector._apply_nms(boxes, confidences, 0.4)

        # Should return filtered boxes
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), len(boxes))

    def test_decode_predictions(self):
        """Test prediction decoding."""
        detector = EASTTextDetector(config=self.config)

        # Create dummy predictions
        scores = np.random.rand(1, 1, 80, 80).astype(np.float32)
        geometry = np.random.rand(1, 5, 80, 80).astype(np.float32)

        boxes, confidences = detector._decode_predictions(scores, geometry, 0.9)

        # Should return tuple of boxes and confidences
        self.assertIsInstance(boxes, list)
        self.assertIsInstance(confidences, list)
        self.assertEqual(len(boxes), len(confidences))


if __name__ == "__main__":
    unittest.main()
