"""
Tests for Training Utilities
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

# Add src/training to path to import modules directly without __init__.py side effects
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'training'))

import numpy as np


class TestDataAugmentation(unittest.TestCase):
    """Test cases for DataAugmentation class."""

    def setUp(self):
        """Set up test fixtures."""
        from augmentation import DataAugmentation
        self.augmentation = DataAugmentation()

    def test_initialization(self):
        """Test augmentation initialization."""
        self.assertIsNotNone(self.augmentation)
        self.assertEqual(self.augmentation.rotation_range, 10)

    def test_initialization_with_config(self):
        """Test augmentation initialization with custom config."""
        from augmentation import DataAugmentation
        config = {"rotation_range": 20, "noise_stddev": 15}
        aug = DataAugmentation(config=config)
        self.assertEqual(aug.rotation_range, 20)
        self.assertEqual(aug.noise_stddev, 15)

    def test_rotate(self):
        """Test image rotation."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.augmentation.rotate(image, angle=5)
        self.assertEqual(result.shape, image.shape)

    def test_shift(self):
        """Test image shifting."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.augmentation.shift(image, dx=0.1, dy=0.1)
        self.assertEqual(result.shape, image.shape)

    def test_zoom(self):
        """Test image zooming."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.augmentation.zoom(image, zoom_factor=1.1)
        self.assertEqual(result.shape, image.shape)

    def test_adjust_brightness(self):
        """Test brightness adjustment."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.augmentation.adjust_brightness(image, factor=1.2)
        self.assertEqual(result.shape, image.shape)

    def test_add_noise(self):
        """Test noise addition."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.augmentation.add_noise(image, stddev=10)
        self.assertEqual(result.shape, image.shape)

    def test_blur(self):
        """Test Gaussian blur."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.augmentation.blur(image, kernel_size=3)
        self.assertEqual(result.shape, image.shape)

    def test_augment_random(self):
        """Test random augmentation."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.augmentation.augment(image)
        self.assertEqual(result.shape, image.shape)

    def test_augment_specific(self):
        """Test specific augmentations."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.augmentation.augment(image, augmentations=["rotate", "blur"])
        self.assertEqual(result.shape, image.shape)

    def test_generate_batch(self):
        """Test batch generation."""
        images = np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)
        labels = np.array([f"label_{i}" for i in range(10)])
        
        batch_images, batch_labels = self.augmentation.generate_batch(
            images, labels, batch_size=4, augment=False
        )
        
        self.assertEqual(batch_images.shape[0], 4)
        self.assertEqual(len(batch_labels), 4)


class TestOCRDataset(unittest.TestCase):
    """Test cases for OCRDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        from dataset import OCRDataset
        self.dataset = OCRDataset()

    def test_initialization(self):
        """Test dataset initialization."""
        self.assertIsNotNone(self.dataset)
        self.assertEqual(len(self.dataset), 0)

    def test_initialization_with_config(self):
        """Test dataset initialization with config."""
        from dataset import OCRDataset
        config = {"image_size": [128, 128], "channels": 1}
        ds = OCRDataset(config=config)
        self.assertEqual(ds.image_size, (128, 128))
        self.assertEqual(ds.channels, 1)

    def test_num_classes(self):
        """Test num_classes property."""
        # Default charset length + blank
        self.assertEqual(self.dataset.num_classes, 63)

    def test_encode_decode_label(self):
        """Test label encoding and decoding."""
        text = "ABC123"
        encoded = self.dataset.encode_label(text)
        
        self.assertEqual(len(encoded), self.dataset.max_label_length)
        
        # Decode (without CTC, just direct decode)
        decoded = self.dataset.decode_label(encoded)
        self.assertEqual(decoded, text)

    def test_add_sample(self):
        """Test adding a sample to dataset."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        label = "TEST"
        
        self.dataset.add_sample(image, label)
        
        self.assertEqual(len(self.dataset), 1)
        self.assertEqual(self.dataset.labels[0], label)

    def test_get_item(self):
        """Test getting item from dataset."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        label = "SAMPLE"
        
        self.dataset.add_sample(image, label)
        
        img, lbl = self.dataset[0]
        self.assertEqual(lbl, label)

    def test_split(self):
        """Test dataset splitting."""
        # Add some samples
        for i in range(10):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            self.dataset.add_sample(image, f"TEXT{i}")
        
        train_ds, val_ds, test_ds = self.dataset.split(
            train_ratio=0.6,
            val_ratio=0.2,
            shuffle=False
        )
        
        self.assertEqual(len(train_ds), 6)
        self.assertEqual(len(val_ds), 2)
        self.assertEqual(len(test_ds), 2)

    def test_get_batch(self):
        """Test batch retrieval."""
        for i in range(10):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            self.dataset.add_sample(image, f"TEXT{i}")
        
        images, labels = self.dataset.get_batch(batch_size=4, shuffle=False)
        
        self.assertEqual(images.shape[0], 4)
        self.assertEqual(labels.shape[0], 4)

    def test_save_and_load(self):
        """Test saving and loading dataset."""
        # Add samples
        for i in range(5):
            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            self.dataset.add_sample(image, f"LABEL{i}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            self.dataset.save(tmpdir)
            
            # Verify files exist
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "labels.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "images")))
            
            # Load
            from dataset import OCRDataset
            loaded_ds = OCRDataset.load(tmpdir)
            
            self.assertEqual(len(loaded_ds), 5)


class TestOCRMetrics(unittest.TestCase):
    """Test cases for OCRMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        from metrics import OCRMetrics
        self.metrics = OCRMetrics()

    def test_initialization(self):
        """Test metrics initialization."""
        self.assertIsNotNone(self.metrics)

    def test_character_accuracy_perfect(self):
        """Test character accuracy with perfect predictions."""
        predictions = ["ABC", "123"]
        ground_truths = ["ABC", "123"]
        
        accuracy = self.metrics.character_accuracy(predictions, ground_truths)
        self.assertEqual(accuracy, 1.0)

    def test_character_accuracy_partial(self):
        """Test character accuracy with partial match."""
        predictions = ["ABC"]
        ground_truths = ["ABX"]
        
        accuracy = self.metrics.character_accuracy(predictions, ground_truths)
        self.assertAlmostEqual(accuracy, 2/3, places=5)

    def test_word_accuracy(self):
        """Test word accuracy."""
        predictions = ["ABC", "123", "XYZ"]
        ground_truths = ["ABC", "124", "XYZ"]
        
        accuracy = self.metrics.word_accuracy(predictions, ground_truths)
        self.assertAlmostEqual(accuracy, 2/3, places=5)

    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        self.assertEqual(self.metrics.levenshtein_distance("ABC", "ABC"), 0)
        self.assertEqual(self.metrics.levenshtein_distance("ABC", "ABD"), 1)
        self.assertEqual(self.metrics.levenshtein_distance("ABC", "ABCD"), 1)
        self.assertEqual(self.metrics.levenshtein_distance("", "ABC"), 3)

    def test_character_error_rate(self):
        """Test character error rate."""
        predictions = ["ABC"]
        ground_truths = ["ABC"]
        
        cer = self.metrics.character_error_rate(predictions, ground_truths)
        self.assertEqual(cer, 0.0)

    def test_evaluate(self):
        """Test full evaluation."""
        predictions = ["HELLO", "WORLD"]
        ground_truths = ["HELLO", "WORLD"]
        
        result = self.metrics.evaluate(predictions, ground_truths)
        
        self.assertIn("character_accuracy", result)
        self.assertIn("word_accuracy", result)
        self.assertIn("character_error_rate", result)
        self.assertIn("f1_score", result)

    def test_format_metrics(self):
        """Test metrics formatting."""
        metrics = {
            "character_accuracy": 0.95,
            "word_accuracy": 0.90,
            "character_error_rate": 0.05,
            "f1_score": 0.92
        }
        
        formatted = self.metrics.format_metrics(metrics)
        
        self.assertIn("Character Accuracy", formatted)
        self.assertIn("0.95", formatted)


class TestOCRTrainer(unittest.TestCase):
    """Test cases for OCRTrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        from trainer import OCRTrainer
        self.config = {
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "input_shape": [64, 64, 3],
            "num_classes": 10
        }
        self.trainer = OCRTrainer(config=self.config)

    def test_initialization(self):
        """Test trainer initialization."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.epochs, 2)
        self.assertEqual(self.trainer.batch_size, 4)

    def test_build_cnn_model(self):
        """Test building CNN model."""
        model = self.trainer.build_model(architecture="cnn")
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape[1:], tuple(self.config["input_shape"]))

    def test_build_crnn_model(self):
        """Test building CRNN model."""
        model = self.trainer.build_model(architecture="crnn")
        
        self.assertIsNotNone(model)

    def test_compile_model(self):
        """Test model compilation."""
        self.trainer.build_model(architecture="cnn")
        self.trainer.compile_model(optimizer="adam", loss="categorical_crossentropy")
        
        self.assertIsNotNone(self.trainer.model.optimizer)

    def test_save_and_load_model(self):
        """Test model saving and loading."""
        self.trainer.build_model(architecture="cnn")
        self.trainer.compile_model(optimizer="adam", loss="categorical_crossentropy")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.h5")
            
            # Save
            self.trainer.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Load
            from trainer import OCRTrainer
            new_trainer = OCRTrainer(config=self.config)
            new_trainer.load_model(model_path)
            
            self.assertIsNotNone(new_trainer.model)

    def test_get_training_summary_empty(self):
        """Test training summary before training."""
        summary = self.trainer.get_training_summary()
        self.assertEqual(summary, {})


if __name__ == "__main__":
    unittest.main()
