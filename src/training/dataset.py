"""
Dataset Module

Provides dataset management and loading utilities for OCR training.
"""

import json
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


class OCRDataset:
    """Dataset class for OCR training data."""

    def __init__(
        self,
        data_dir: str = None,
        config: dict = None
    ):
        """
        Initialize OCR dataset.

        Args:
            data_dir: Path to dataset directory
            config: Dataset configuration dictionary
        """
        self.data_dir = data_dir
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Dataset configuration
        self.image_size = tuple(self.config.get("image_size", [224, 224]))
        self.channels = self.config.get("channels", 3)
        self.normalize = self.config.get("normalize", True)
        self.max_label_length = self.config.get("max_label_length", 32)

        # Character set for encoding/decoding
        self.charset = self.config.get(
            "charset",
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.charset)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.charset)}
        self.blank_idx = 0  # CTC blank token

        # Data storage
        self.images = []
        self.labels = []
        self.encoded_labels = []
        self.file_paths = []

    @property
    def num_classes(self) -> int:
        """Return number of classes (charset size + blank)."""
        return len(self.charset) + 1

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label)
        """
        return self.images[idx], self.labels[idx]

    def load_from_directory(
        self,
        data_dir: str = None,
        labels_file: str = None
    ) -> int:
        """
        Load dataset from directory structure.

        Expected structure:
        data_dir/
            images/
                img_001.png
                img_002.png
                ...
            labels.json  # {"img_001.png": "TEXT1", "img_002.png": "TEXT2", ...}

        Args:
            data_dir: Path to dataset directory (uses self.data_dir if None)
            labels_file: Path to labels JSON file

        Returns:
            Number of samples loaded
        """
        data_dir = data_dir or self.data_dir
        if not data_dir:
            self.logger.error("No data directory specified")
            return 0

        images_dir = os.path.join(data_dir, "images")
        labels_file = labels_file or os.path.join(data_dir, "labels.json")

        if not os.path.exists(images_dir):
            self.logger.error(f"Images directory not found: {images_dir}")
            return 0

        if not os.path.exists(labels_file):
            self.logger.error(f"Labels file not found: {labels_file}")
            return 0

        # Load labels
        with open(labels_file, "r") as f:
            labels_dict = json.load(f)

        # Load images and labels
        loaded = 0
        for filename, label in labels_dict.items():
            image_path = os.path.join(images_dir, filename)
            if os.path.exists(image_path):
                image = self._load_image(image_path)
                if image is not None:
                    self.images.append(image)
                    self.labels.append(label)
                    self.encoded_labels.append(self.encode_label(label))
                    self.file_paths.append(image_path)
                    loaded += 1
            else:
                self.logger.warning(f"Image not found: {image_path}")

        self.logger.info(f"Loaded {loaded} samples from {data_dir}")
        return loaded

    def load_from_annotations(
        self,
        annotations: List[Dict],
        images_dir: str = None
    ) -> int:
        """
        Load dataset from annotations list.

        Args:
            annotations: List of dicts with 'image_path' and 'text' keys
            images_dir: Base directory for image paths

        Returns:
            Number of samples loaded
        """
        loaded = 0
        for ann in annotations:
            image_path = ann.get("image_path")
            text = ann.get("text")

            if images_dir and not os.path.isabs(image_path):
                image_path = os.path.join(images_dir, image_path)

            if image_path and text:
                image = self._load_image(image_path)
                if image is not None:
                    self.images.append(image)
                    self.labels.append(text)
                    self.encoded_labels.append(self.encode_label(text))
                    self.file_paths.append(image_path)
                    loaded += 1

        self.logger.info(f"Loaded {loaded} samples from annotations")
        return loaded

    def add_sample(
        self,
        image: np.ndarray,
        label: str,
        file_path: str = None
    ):
        """
        Add a single sample to the dataset.

        Args:
            image: Image array
            label: Text label
            file_path: Optional file path for reference
        """
        processed = self._preprocess_image(image)
        self.images.append(processed)
        self.labels.append(label)
        self.encoded_labels.append(self.encode_label(label))
        self.file_paths.append(file_path or "")

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess image from file.

        Args:
            path: Path to image file

        Returns:
            Preprocessed image array or None if failed
        """
        try:
            if self.channels == 1:
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(path, cv2.IMREAD_COLOR)

            if image is None:
                self.logger.warning(f"Failed to load image: {path}")
                return None

            return self._preprocess_image(image)
        except Exception as e:
            self.logger.error(f"Error loading image {path}: {e}")
            return None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Input image

        Returns:
            Preprocessed image
        """
        # Resize to target size
        resized = cv2.resize(image, self.image_size)

        # Convert grayscale to 3 channel if needed
        if self.channels == 3 and len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        elif self.channels == 1 and len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Normalize pixel values
        if self.normalize:
            resized = resized.astype(np.float32) / 255.0

        return resized

    def encode_label(self, text: str) -> np.ndarray:
        """
        Encode text label to numeric array.

        Args:
            text: Text label

        Returns:
            Encoded label array
        """
        encoded = []
        for char in text:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                # Unknown character, skip
                pass

        # Pad to max length
        encoded = encoded[:self.max_label_length]
        padded = np.zeros(self.max_label_length, dtype=np.int32)
        padded[:len(encoded)] = encoded

        return padded

    def decode_label(self, encoded: np.ndarray) -> str:
        """
        Decode numeric array to text label.

        Args:
            encoded: Encoded label array

        Returns:
            Decoded text
        """
        chars = []
        for idx in encoded:
            if idx == self.blank_idx:
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])

        return "".join(chars)

    def decode_ctc(self, predictions: np.ndarray) -> str:
        """
        Decode CTC output to text.

        Args:
            predictions: CTC output array (sequence of class probabilities)

        Returns:
            Decoded text
        """
        # Get most likely class for each timestep
        argmax = np.argmax(predictions, axis=-1)

        # Remove consecutive duplicates and blanks
        decoded = []
        prev = None
        for idx in argmax:
            if idx != prev and idx != self.blank_idx:
                if idx in self.idx_to_char:
                    decoded.append(self.idx_to_char[idx])
            prev = idx

        return "".join(decoded)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = None
    ) -> Tuple["OCRDataset", "OCRDataset", "OCRDataset"]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if seed is not None:
            np.random.seed(seed)

        n_samples = len(self)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # Create new datasets
        train_ds = self._create_subset(train_indices)
        val_ds = self._create_subset(val_indices)
        test_ds = self._create_subset(test_indices)

        return train_ds, val_ds, test_ds

    def _create_subset(self, indices: np.ndarray) -> "OCRDataset":
        """
        Create a new dataset with subset of samples.

        Args:
            indices: Indices of samples to include

        Returns:
            New OCRDataset with selected samples
        """
        subset = OCRDataset(config=self.config)
        subset.charset = self.charset
        subset.char_to_idx = self.char_to_idx
        subset.idx_to_char = self.idx_to_char

        for idx in indices:
            subset.images.append(self.images[idx])
            subset.labels.append(self.labels[idx])
            subset.encoded_labels.append(self.encoded_labels[idx])
            subset.file_paths.append(self.file_paths[idx])

        return subset

    def get_batch(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of samples.

        Args:
            batch_size: Number of samples in batch
            shuffle: Whether to randomly select samples

        Returns:
            Tuple of (images_batch, labels_batch)
        """
        if shuffle:
            indices = np.random.choice(len(self), batch_size, replace=True)
        else:
            indices = np.arange(min(batch_size, len(self)))

        images = np.array([self.images[i] for i in indices])
        labels = np.array([self.encoded_labels[i] for i in indices])

        return images, labels

    def to_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        augmentation_fn: Callable = None
    ):
        """
        Convert to TensorFlow Dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
            augmentation_fn: Optional augmentation function

        Returns:
            TensorFlow Dataset object
        """
        import tensorflow as tf

        images = np.array(self.images)
        labels = np.array(self.encoded_labels)

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self))

        if augmentation_fn:
            dataset = dataset.map(
                lambda x, y: (augmentation_fn(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def save(self, path: str):
        """
        Save dataset to disk.

        Args:
            path: Path to save directory
        """
        os.makedirs(path, exist_ok=True)

        # Save images
        images_dir = os.path.join(path, "images")
        os.makedirs(images_dir, exist_ok=True)

        labels_dict = {}
        for i, (image, label) in enumerate(zip(self.images, self.labels)):
            filename = f"img_{i:06d}.png"
            image_path = os.path.join(images_dir, filename)

            # Convert back from normalized if needed
            if self.normalize:
                save_image = (image * 255).astype(np.uint8)
            else:
                save_image = image

            cv2.imwrite(image_path, save_image)
            labels_dict[filename] = label

        # Save labels
        labels_file = os.path.join(path, "labels.json")
        with open(labels_file, "w") as f:
            json.dump(labels_dict, f, indent=2)

        # Save config
        config_file = os.path.join(path, "config.json")
        with open(config_file, "w") as f:
            json.dump({
                "image_size": list(self.image_size),
                "channels": self.channels,
                "normalize": self.normalize,
                "max_label_length": self.max_label_length,
                "charset": self.charset
            }, f, indent=2)

        self.logger.info(f"Saved dataset with {len(self)} samples to {path}")

    @classmethod
    def load(cls, path: str) -> "OCRDataset":
        """
        Load dataset from disk.

        Args:
            path: Path to saved dataset directory

        Returns:
            Loaded OCRDataset
        """
        config_file = os.path.join(path, "config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            config = {}

        dataset = cls(data_dir=path, config=config)
        dataset.load_from_directory(path)

        return dataset
