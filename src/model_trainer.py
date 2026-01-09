"""
TensorFlow Model Training Utilities

This module provides utilities for training custom TensorFlow models for OCR.
"""

import logging
import os
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DataGenerator:
    """Data generator for OCR model training."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
    ):
        """
        Initialize data generator.

        Args:
            data_dir: Directory containing training data
            batch_size: Batch size for training
            image_size: Target image size (height, width)
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.logger = logging.getLogger(__name__)

    def load_dataset(
        self, split: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from directory.

        Args:
            split: Dataset split ('train', 'val', 'test')

        Returns:
            Tuple of (images, labels)
        """
        data_path = os.path.join(self.data_dir, split)

        if not os.path.exists(data_path):
            self.logger.error(f"Data path not found: {data_path}")
            return None, None

        images = []
        labels = []

        # Load images and labels from directory
        # This is a placeholder - actual implementation depends on data format
        for label_dir in os.listdir(data_path):
            label_path = os.path.join(data_path, label_dir)
            if not os.path.isdir(label_path):
                continue

            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img = cv2.resize(img, self.image_size)
                    img = img / 255.0  # Normalize

                    images.append(img)
                    labels.append(label_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to load image {img_path}: {e}")

        return np.array(images), np.array(labels)

    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to image.

        Args:
            image: Input image

        Returns:
            Augmented image
        """
        # Random brightness
        if np.random.random() > 0.5:
            image = tf.image.random_brightness(image, 0.2)

        # Random contrast
        if np.random.random() > 0.5:
            image = tf.image.random_contrast(image, 0.8, 1.2)

        # Random flip
        if np.random.random() > 0.5:
            image = tf.image.flip_left_right(image)

        # Random rotation (small angles for text)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            image = self._rotate_image(image, angle)

        return image

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated


class ModelBuilder:
    """Builder for creating OCR models."""

    def __init__(self, config: dict):
        """
        Initialize model builder.

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def build_cnn_model(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 10,
    ) -> keras.Model:
        """
        Build a CNN model for text recognition.

        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Input(shape=input_shape),

            # Convolutional blocks
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ])

        return model

    def build_crnn_model(
        self,
        input_shape: Tuple[int, int, int] = (32, 128, 1),
        num_classes: int = 37,  # 26 letters + 10 digits + blank
    ) -> keras.Model:
        """
        Build a CRNN (Convolutional Recurrent Neural Network) model for OCR.

        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes (characters)

        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=input_shape)

        # Convolutional layers
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)

        # Reshape for RNN
        new_shape = ((input_shape[0] // 4), (input_shape[1] // 4) * 256)
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(64, activation="relu")(x)

        # RNN layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

        # Output layer
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model


class ModelTrainer:
    """Trainer for OCR models."""

    def __init__(self, config: dict):
        """
        Initialize model trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def train(
        self,
        model: keras.Model,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        save_path: str = "./models/trained_model.h5",
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            model: Keras model to train
            train_data: Tuple of (train_images, train_labels)
            val_data: Tuple of (val_images, val_labels)
            epochs: Number of training epochs
            batch_size: Batch size
            save_path: Path to save trained model

        Returns:
            Training history
        """
        train_images, train_labels = train_data
        val_images, val_labels = val_data

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                save_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            keras.callbacks.TensorBoard(
                log_dir=f"./logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                histogram_freq=1,
            ),
        ]

        # Train model
        self.logger.info("Starting model training...")
        history = model.fit(
            train_images,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_images, val_labels),
            callbacks=callbacks,
            verbose=1,
        )

        self.logger.info(f"Training complete. Model saved to {save_path}")
        return history

    def evaluate(
        self,
        model: keras.Model,
        test_data: Tuple[np.ndarray, np.ndarray],
    ) -> dict:
        """
        Evaluate the model.

        Args:
            model: Trained Keras model
            test_data: Tuple of (test_images, test_labels)

        Returns:
            Dictionary of evaluation metrics
        """
        test_images, test_labels = test_data

        self.logger.info("Evaluating model...")
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)

        results = {
            "loss": loss,
            "accuracy": accuracy,
        }

        self.logger.info(f"Evaluation results: {results}")
        return results


def create_training_script():
    """
    Create a sample training script.
    This can be used as a template for custom training.
    """
    script_content = """#!/usr/bin/env python3
'''
Sample TensorFlow Model Training Script for Trakt OCR

This script demonstrates how to train a custom OCR model using the
training utilities provided by Trakt.
'''

from src.model_trainer import ModelBuilder, ModelTrainer, DataGenerator

# Configuration
config = {
    'data_dir': './training_data',
    'model_type': 'cnn',  # 'cnn' or 'crnn'
    'input_shape': (224, 224, 3),
    'num_classes': 36,  # 26 letters + 10 digits
    'batch_size': 32,
    'epochs': 50,
    'save_path': './models/custom_ocr_model.h5'
}

def main():
    # Initialize components
    data_gen = DataGenerator(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        image_size=config['input_shape'][:2]
    )

    # Load datasets
    print("Loading training data...")
    train_images, train_labels = data_gen.load_dataset('train')
    val_images, val_labels = data_gen.load_dataset('val')
    test_images, test_labels = data_gen.load_dataset('test')

    # Build model
    builder = ModelBuilder(config)
    if config['model_type'] == 'cnn':
        model = builder.build_cnn_model(
            input_shape=config['input_shape'],
            num_classes=config['num_classes']
        )
    else:
        model = builder.build_crnn_model(
            input_shape=config['input_shape'],
            num_classes=config['num_classes']
        )

    print(model.summary())

    # Train model
    trainer = ModelTrainer(config)
    history = trainer.train(
        model=model,
        train_data=(train_images, train_labels),
        val_data=(val_images, val_labels),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        save_path=config['save_path']
    )

    # Evaluate model
    results = trainer.evaluate(model, (test_images, test_labels))
    print(f"Final test accuracy: {results['accuracy']:.4f}")

if __name__ == '__main__':
    main()
"""
    return script_content
