"""
Data Augmentation Module

Provides data augmentation utilities for training OCR models.
"""

import logging
from typing import Tuple

import cv2
import numpy as np


class DataAugmentation:
    """Data augmentation for OCR training images."""

    def __init__(self, config: dict = None):
        """
        Initialize data augmentation.

        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Augmentation parameters
        self.rotation_range = self.config.get("rotation_range", 10)
        self.width_shift_range = self.config.get("width_shift_range", 0.1)
        self.height_shift_range = self.config.get("height_shift_range", 0.1)
        self.shear_range = self.config.get("shear_range", 0.1)
        self.zoom_range = self.config.get("zoom_range", 0.1)
        self.brightness_range = self.config.get("brightness_range", (0.8, 1.2))
        self.noise_stddev = self.config.get("noise_stddev", 10)
        self.blur_kernel_range = self.config.get("blur_kernel_range", (1, 3))

    def rotate(self, image: np.ndarray, angle: float = None) -> np.ndarray:
        """
        Rotate image by a random or specified angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees (random if None)

        Returns:
            Rotated image
        """
        if angle is None:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def shift(
        self, image: np.ndarray, dx: float = None, dy: float = None
    ) -> np.ndarray:
        """
        Shift image horizontally and/or vertically.

        Args:
            image: Input image
            dx: Horizontal shift as fraction of width (random if None)
            dy: Vertical shift as fraction of height (random if None)

        Returns:
            Shifted image
        """
        h, w = image.shape[:2]

        if dx is None:
            dx = np.random.uniform(-self.width_shift_range, self.width_shift_range)
        if dy is None:
            dy = np.random.uniform(-self.height_shift_range, self.height_shift_range)

        shift_x = int(w * dx)
        shift_y = int(h * dy)

        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        return shifted

    def shear(self, image: np.ndarray, shear_factor: float = None) -> np.ndarray:
        """
        Apply shear transformation to image.

        Args:
            image: Input image
            shear_factor: Shear factor (random if None)

        Returns:
            Sheared image
        """
        if shear_factor is None:
            shear_factor = np.random.uniform(-self.shear_range, self.shear_range)

        h, w = image.shape[:2]

        # Create shear transformation matrix
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        pts2 = np.float32([
            [0, 0],
            [w + int(shear_factor * h), 0],
            [int(shear_factor * w), h]
        ])

        matrix = cv2.getAffineTransform(pts1, pts2)
        sheared = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        return sheared

    def zoom(self, image: np.ndarray, zoom_factor: float = None) -> np.ndarray:
        """
        Zoom in or out of image.

        Args:
            image: Input image
            zoom_factor: Zoom factor (random if None)

        Returns:
            Zoomed image
        """
        if zoom_factor is None:
            if isinstance(self.zoom_range, (tuple, list)):
                zoom_factor = np.random.uniform(
                    1 - self.zoom_range[0], 1 + self.zoom_range[1]
                )
            else:
                zoom_factor = np.random.uniform(
                    1 - self.zoom_range, 1 + self.zoom_range
                )

        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h))

        # Crop or pad to original size
        if zoom_factor > 1:
            # Crop center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            zoomed = resized[start_y:start_y + h, start_x:start_x + w]
        else:
            # Pad to original size
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            zoomed = cv2.copyMakeBorder(
                resized,
                pad_y, h - new_h - pad_y,
                pad_x, w - new_w - pad_x,
                cv2.BORDER_REPLICATE
            )

        return zoomed

    def adjust_brightness(
        self, image: np.ndarray, factor: float = None
    ) -> np.ndarray:
        """
        Adjust image brightness.

        Args:
            image: Input image
            factor: Brightness factor (random if None)

        Returns:
            Brightness-adjusted image
        """
        if factor is None:
            factor = np.random.uniform(
                self.brightness_range[0], self.brightness_range[1]
            )

        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted

    def add_noise(
        self, image: np.ndarray, stddev: float = None
    ) -> np.ndarray:
        """
        Add Gaussian noise to image.

        Args:
            image: Input image
            stddev: Noise standard deviation (uses config if None)

        Returns:
            Noisy image
        """
        if stddev is None:
            stddev = self.noise_stddev

        noise = np.random.normal(0, stddev, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def blur(self, image: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """
        Apply Gaussian blur to image.

        Args:
            image: Input image
            kernel_size: Blur kernel size (random if None)

        Returns:
            Blurred image
        """
        if kernel_size is None:
            kernel_size = np.random.randint(
                self.blur_kernel_range[0], self.blur_kernel_range[1] + 1
            )
            # Ensure kernel size is odd
            kernel_size = kernel_size * 2 + 1

        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred

    def erode_dilate(
        self, image: np.ndarray, operation: str = None, kernel_size: int = 2
    ) -> np.ndarray:
        """
        Apply erosion or dilation to image.

        Args:
            image: Input image
            operation: "erode" or "dilate" (random if None)
            kernel_size: Morphological kernel size

        Returns:
            Processed image
        """
        if operation is None:
            operation = np.random.choice(["erode", "dilate"])

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation == "erode":
            processed = cv2.erode(image, kernel, iterations=1)
        else:
            processed = cv2.dilate(image, kernel, iterations=1)

        return processed

    def augment(
        self,
        image: np.ndarray,
        augmentations: list = None
    ) -> np.ndarray:
        """
        Apply random augmentations to image.

        Args:
            image: Input image
            augmentations: List of augmentation names to apply
                          (random selection if None)

        Returns:
            Augmented image
        """
        if augmentations is None:
            # Randomly select augmentations
            all_augmentations = [
                "rotate", "shift", "shear", "zoom",
                "brightness", "noise", "blur"
            ]
            num_augmentations = np.random.randint(1, 4)
            augmentations = np.random.choice(
                all_augmentations, num_augmentations, replace=False
            )

        augmented = image.copy()

        for aug in augmentations:
            if aug == "rotate":
                augmented = self.rotate(augmented)
            elif aug == "shift":
                augmented = self.shift(augmented)
            elif aug == "shear":
                augmented = self.shear(augmented)
            elif aug == "zoom":
                augmented = self.zoom(augmented)
            elif aug == "brightness":
                augmented = self.adjust_brightness(augmented)
            elif aug == "noise":
                augmented = self.add_noise(augmented)
            elif aug == "blur":
                augmented = self.blur(augmented)
            elif aug == "erode":
                augmented = self.erode_dilate(augmented, "erode")
            elif aug == "dilate":
                augmented = self.erode_dilate(augmented, "dilate")

        return augmented

    def generate_batch(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        augment: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate augmented batch of images and labels.

        Args:
            images: Array of images
            labels: Array of corresponding labels
            batch_size: Number of samples in batch
            augment: Whether to apply augmentation

        Returns:
            Tuple of (batch_images, batch_labels)
        """
        indices = np.random.choice(len(images), batch_size, replace=True)
        batch_images = []
        batch_labels = []

        for idx in indices:
            img = images[idx].copy()
            label = labels[idx]

            if augment:
                img = self.augment(img)

            batch_images.append(img)
            batch_labels.append(label)

        return np.array(batch_images), np.array(batch_labels)
