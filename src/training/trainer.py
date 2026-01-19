"""
Trainer Module

Provides model training utilities for OCR models.
"""

import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Support both relative and direct imports
try:
    from .augmentation import DataAugmentation
    from .dataset import OCRDataset
    from .metrics import OCRMetrics
except ImportError:
    from augmentation import DataAugmentation
    from dataset import OCRDataset
    from metrics import OCRMetrics


class OCRTrainer:
    """Trainer class for OCR models."""

    def __init__(self, config: dict = None):
        """
        Initialize OCR trainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Training parameters
        self.epochs = self.config.get("epochs", 100)
        self.batch_size = self.config.get("batch_size", 32)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.early_stopping_patience = self.config.get("early_stopping_patience", 10)
        self.reduce_lr_patience = self.config.get("reduce_lr_patience", 5)
        self.reduce_lr_factor = self.config.get("reduce_lr_factor", 0.5)
        self.min_lr = self.config.get("min_lr", 1e-7)

        # Model configuration
        self.input_shape = tuple(self.config.get("input_shape", [224, 224, 3]))
        self.num_classes = self.config.get("num_classes", 63)  # charset + blank

        # Output configuration
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        self.log_dir = self.config.get("log_dir", "./logs")
        self.model_save_path = self.config.get("model_save_path", "./models/ocr_model.h5")

        # Components
        self.model = None
        self.history = None
        self.augmentation = None
        self.metrics = OCRMetrics()

        # Configure GPU
        self._configure_gpu()

    def _configure_gpu(self):
        """Configure GPU settings."""
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            try:
                allow_growth = self.config.get("allow_growth", True)
                gpu_memory_limit = self.config.get("gpu_memory_limit")

                for gpu in gpus:
                    if allow_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)

                    if gpu_memory_limit:
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=gpu_memory_limit
                            )]
                        )

                self.logger.info(f"Configured {len(gpus)} GPU(s) for training")
            except RuntimeError as e:
                self.logger.warning(f"GPU configuration failed: {e}")

    def build_model(
        self,
        architecture: str = "cnn",
        pretrained_path: str = None
    ) -> keras.Model:
        """
        Build or load OCR model.

        Args:
            architecture: Model architecture ("cnn", "crnn", or "custom")
            pretrained_path: Path to pretrained model weights

        Returns:
            Keras model
        """
        if pretrained_path and os.path.exists(pretrained_path):
            self.logger.info(f"Loading pretrained model from {pretrained_path}")
            self.model = keras.models.load_model(pretrained_path)
            return self.model

        self.logger.info(f"Building {architecture} model")

        if architecture == "cnn":
            self.model = self._build_cnn_model()
        elif architecture == "crnn":
            self.model = self._build_crnn_model()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        return self.model

    def _build_cnn_model(self) -> keras.Model:
        """
        Build a CNN-based OCR model.

        Returns:
            Keras model
        """
        inputs = keras.layers.Input(shape=self.input_shape)

        # Convolutional blocks
        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Dense layers
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(0.3)(x)

        outputs = keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_ocr")
        return model

    def _build_crnn_model(self) -> keras.Model:
        """
        Build a CRNN (CNN + RNN) model for sequence recognition.

        Returns:
            Keras model
        """
        inputs = keras.layers.Input(shape=self.input_shape)

        # CNN feature extraction
        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 1))(x)

        x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 1))(x)

        # Reshape for RNN
        # Calculate output shape after convolutions
        new_shape = self._calculate_rnn_input_shape(x)
        x = keras.layers.Reshape(target_shape=new_shape)(x)

        # Bidirectional LSTM layers
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
        )(x)
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
        )(x)

        # Output layer
        outputs = keras.layers.Dense(self.num_classes, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="crnn_ocr")
        return model

    def _calculate_rnn_input_shape(self, x):
        """Calculate the shape for RNN input after CNN layers."""
        shape = x.shape
        # Reshape to (time_steps, features)
        time_steps = shape[2]  # width
        features = shape[1] * shape[3]  # height * channels
        return (time_steps, features)

    def compile_model(
        self,
        optimizer: str = "adam",
        loss: str = "ctc",
        metrics: List[str] = None
    ):
        """
        Compile the model for training.

        Args:
            optimizer: Optimizer name or instance
            loss: Loss function name
            metrics: List of metric names
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        # Create optimizer
        if optimizer == "adam":
            opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif optimizer == "sgd":
            opt = keras.optimizers.SGD(
                learning_rate=self.learning_rate, momentum=0.9
            )
        elif optimizer == "rmsprop":
            opt = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            opt = optimizer

        # Create loss function
        if loss == "ctc":
            loss_fn = self._ctc_loss
        elif loss == "categorical_crossentropy":
            loss_fn = "categorical_crossentropy"
        else:
            loss_fn = loss

        # Default metrics
        if metrics is None:
            metrics = ["accuracy"]

        self.model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
        self.logger.info("Model compiled successfully")

    def _ctc_loss(self, y_true, y_pred):
        """
        CTC loss function for sequence-to-sequence learning.

        Args:
            y_true: True labels
            y_pred: Predicted outputs

        Returns:
            CTC loss value
        """
        batch_size = tf.shape(y_pred)[0]
        input_length = tf.shape(y_pred)[1]
        label_length = tf.shape(y_true)[1]

        input_lengths = tf.fill([batch_size], input_length)
        label_lengths = tf.fill([batch_size], label_length)

        return tf.keras.backend.ctc_batch_cost(
            y_true, y_pred, input_lengths, label_lengths
        )

    def train(
        self,
        train_dataset: OCRDataset,
        val_dataset: OCRDataset = None,
        augmentation: DataAugmentation = None,
        callbacks: List[keras.callbacks.Callback] = None
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            augmentation: Data augmentation instance
            callbacks: Additional callbacks

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")

        self.augmentation = augmentation

        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

        # Default callbacks
        default_callbacks = self._create_default_callbacks()

        if callbacks:
            default_callbacks.extend(callbacks)

        # Prepare datasets
        train_tf_dataset = train_dataset.to_tf_dataset(
            batch_size=self.batch_size,
            shuffle=True,
            augmentation_fn=self._augment_batch if augmentation else None
        )

        val_tf_dataset = None
        if val_dataset:
            val_tf_dataset = val_dataset.to_tf_dataset(
                batch_size=self.batch_size,
                shuffle=False
            )

        # Train model
        self.logger.info("Starting training...")
        start_time = time.time()

        self.history = self.model.fit(
            train_tf_dataset,
            validation_data=val_tf_dataset,
            epochs=self.epochs,
            callbacks=default_callbacks,
            verbose=1
        )

        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        # Save final model
        self.save_model(self.model_save_path)

        return self.history

    def _augment_batch(self, image):
        """Apply augmentation to a batch of images."""
        if self.augmentation is None:
            return image

        return tf.numpy_function(
            self.augmentation.augment,
            [image],
            tf.float32
        )

    def _create_default_callbacks(self) -> List[keras.callbacks.Callback]:
        """Create default training callbacks."""
        callbacks = []

        # Model checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, "model_{epoch:03d}_{val_loss:.4f}.h5"
        )
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))

        # Early stopping
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))

        # Learning rate reduction
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr,
            verbose=1
        ))

        # TensorBoard logging
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch"
        ))

        # CSV logger
        csv_path = os.path.join(self.log_dir, "training_log.csv")
        callbacks.append(keras.callbacks.CSVLogger(csv_path))

        return callbacks

    def evaluate(
        self,
        test_dataset: OCRDataset,
        decode_fn: Callable = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset.

        Args:
            test_dataset: Test dataset
            decode_fn: Function to decode model output to text

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        self.logger.info("Evaluating model...")

        # Get predictions
        predictions = []
        ground_truths = test_dataset.labels

        for i in range(0, len(test_dataset), self.batch_size):
            batch_end = min(i + self.batch_size, len(test_dataset))
            batch_images = np.array(test_dataset.images[i:batch_end])

            batch_preds = self.model.predict(batch_images, verbose=0)

            # Decode predictions
            for pred in batch_preds:
                if decode_fn:
                    decoded = decode_fn(pred)
                else:
                    decoded = test_dataset.decode_ctc(pred)
                predictions.append(decoded)

        # Calculate metrics
        metrics_result = self.metrics.evaluate(predictions, ground_truths)

        self.logger.info("\n" + self.metrics.format_metrics(metrics_result))

        return metrics_result

    def predict(
        self,
        images: np.ndarray,
        decode_fn: Callable = None
    ) -> List[str]:
        """
        Make predictions on images.

        Args:
            images: Array of images
            decode_fn: Function to decode model output to text

        Returns:
            List of predicted text strings
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        # Ensure batch dimension
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)

        predictions = self.model.predict(images, verbose=0)

        # Decode predictions
        decoded = []
        for pred in predictions:
            if decode_fn:
                text = decode_fn(pred)
            else:
                # Simple argmax decoding
                text = "".join([chr(int(np.argmax(p)) + 65) for p in pred])
            decoded.append(text)

        return decoded

    def save_model(self, path: str):
        """
        Save model to file.

        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        self.logger.info(f"Model saved to {path}")

        # Save training config
        config_path = path.replace(".h5", "_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def load_model(self, path: str):
        """
        Load model from file.

        Args:
            path: Path to saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        self.model = keras.models.load_model(path)
        self.logger.info(f"Model loaded from {path}")

        # Load training config if available
        config_path = path.replace(".h5", "_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)

    def export_model(
        self,
        export_path: str,
        format: str = "saved_model"
    ):
        """
        Export model in different formats.

        Args:
            export_path: Path to export model
            format: Export format ("saved_model", "tflite", "onnx")
        """
        if self.model is None:
            raise ValueError("No model to export.")

        os.makedirs(export_path, exist_ok=True)

        if format == "saved_model":
            self.model.save(export_path, save_format="tf")
            self.logger.info(f"Model exported as SavedModel to {export_path}")

        elif format == "tflite":
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_model = converter.convert()

            tflite_path = os.path.join(export_path, "model.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            self.logger.info(f"Model exported as TFLite to {tflite_path}")

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_training_summary(self) -> Dict:
        """
        Get training summary and statistics.

        Returns:
            Dictionary with training summary
        """
        if self.history is None:
            return {}

        history_dict = self.history.history

        summary = {
            "epochs_trained": len(history_dict.get("loss", [])),
            "final_loss": history_dict.get("loss", [None])[-1],
            "final_val_loss": history_dict.get("val_loss", [None])[-1],
            "best_val_loss": min(history_dict.get("val_loss", [float("inf")])),
            "final_accuracy": history_dict.get("accuracy", [None])[-1],
            "final_val_accuracy": history_dict.get("val_accuracy", [None])[-1],
            "config": self.config
        }

        return summary
