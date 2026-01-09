"""
Tests for OCR Engine
"""

import unittest

import numpy as np

from src.ocr_engine import OCREngine


class TestOCREngine(unittest.TestCase):
    """Test cases for OCREngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "engine": "tesseract",
            "confidence_threshold": 0.5,
            "min_text_length": 2,
            "filter_patterns": ["[0-9]+"],
            "tesseract": {"language": "eng", "oem": 3, "psm": 6},
        }

    def test_initialization(self):
        """Test OCR engine initialization."""
        engine = OCREngine(self.config)

        self.assertEqual(engine.engine_type, "tesseract")
        self.assertEqual(engine.confidence_threshold, 0.5)
        self.assertEqual(engine.min_text_length, 2)

    def test_preprocess_frame(self):
        """Test frame preprocessing."""
        engine = OCREngine(self.config)

        # Create a dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        processed = engine.preprocess_frame(frame)

        # Check that output is grayscale
        self.assertEqual(len(processed.shape), 2)
        self.assertEqual(processed.shape[0], 480)
        self.assertEqual(processed.shape[1], 640)

    def test_filter_results_by_length(self):
        """Test filtering results by minimum text length."""
        engine = OCREngine(self.config)

        results = [
            {"text": "A", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"text": "AB", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"text": "ABC", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
        ]

        filtered = engine._filter_results(results)

        # Only results with length >= 2 should pass
        self.assertEqual(len(filtered), 2)
        self.assertNotIn("A", [r["text"] for r in filtered])

    def test_annotate_frame(self):
        """Test frame annotation."""
        engine = OCREngine(self.config)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [{"text": "TEST", "confidence": 0.95, "bbox": [100, 100, 200, 150]}]

        annotated = engine.annotate_frame(frame, results)

        # Check that frame was modified
        self.assertFalse(np.array_equal(frame, annotated))
        self.assertEqual(annotated.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
