"""
Example: Training an OCR Model

This example demonstrates how to train a custom OCR model using the
Trakt training utilities.
"""

import argparse
import logging
import os
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import DataAugmentation, OCRDataset, OCRMetrics, OCRTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_dataset(output_dir: str, num_samples: int = 100):
    """
    Create a sample dataset for demonstration.

    Args:
        output_dir: Directory to save the sample dataset
        num_samples: Number of samples to generate
    """
    import json

    import cv2
    import numpy as np

    logger.info(f"Creating sample dataset with {num_samples} samples...")

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    labels = {}
    charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i in range(num_samples):
        # Generate random text (3-8 characters)
        text_len = np.random.randint(3, 9)
        text = "".join(np.random.choice(list(charset), text_len))

        # Create image with text
        img = np.ones((64, 200, 3), dtype=np.uint8) * 255

        # Add some noise
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = np.clip(img.astype(np.int32) - noise, 0, 255).astype(np.uint8)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Get text size for centering
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (200 - text_width) // 2
        y = (64 + text_height) // 2

        cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness)

        # Save image
        filename = f"sample_{i:05d}.png"
        cv2.imwrite(os.path.join(images_dir, filename), img)
        labels[filename] = text

    # Save labels
    labels_file = os.path.join(output_dir, "labels.json")
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)

    logger.info(f"Sample dataset created at {output_dir}")
    return output_dir


def main():
    """Run the training example."""
    parser = argparse.ArgumentParser(description="Train an OCR Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/ocr_dataset",
        help="Path to training dataset directory"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample dataset for demonstration"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["cnn", "crnn"],
        default="cnn",
        help="Model architecture"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/training",
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation"
    )
    args = parser.parse_args()

    # Create sample dataset if requested
    if args.create_sample:
        args.data_dir = create_sample_dataset(args.data_dir)

    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Dataset not found at {args.data_dir}")
        logger.info("Use --create-sample to create a sample dataset")
        return

    print("\n" + "=" * 60)
    print("Trakt OCR Model Training")
    print("=" * 60)

    # Dataset configuration
    dataset_config = {
        "image_size": [224, 224],
        "channels": 3,
        "normalize": True,
        "max_label_length": 32,
        "charset": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    }

    # Load dataset
    print("\n📂 Loading dataset...")
    dataset = OCRDataset(data_dir=args.data_dir, config=dataset_config)
    num_loaded = dataset.load_from_directory()
    print(f"   Loaded {num_loaded} samples")

    if num_loaded == 0:
        logger.error("No samples loaded. Check dataset format.")
        return

    # Split dataset
    print("\n📊 Splitting dataset...")
    train_ds, val_ds, test_ds = dataset.split(
        train_ratio=0.8,
        val_ratio=0.1,
        shuffle=True,
        seed=42
    )
    print(f"   Training:   {len(train_ds)} samples")
    print(f"   Validation: {len(val_ds)} samples")
    print(f"   Test:       {len(test_ds)} samples")

    # Training configuration
    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
        "input_shape": [224, 224, 3],
        "num_classes": dataset.num_classes,
        "checkpoint_dir": os.path.join(args.output_dir, "checkpoints"),
        "log_dir": os.path.join(args.output_dir, "logs"),
        "model_save_path": os.path.join(args.output_dir, "model.h5"),
    }

    # Initialize trainer
    print("\n🔧 Initializing trainer...")
    trainer = OCRTrainer(config=training_config)

    # Build model
    print(f"\n🏗️  Building {args.architecture.upper()} model...")
    trainer.build_model(architecture=args.architecture)
    trainer.compile_model(optimizer="adam", loss="categorical_crossentropy")
    trainer.model.summary()

    # Setup augmentation
    augmentation = None
    if args.augment:
        print("\n🎨 Enabling data augmentation...")
        augmentation_config = {
            "rotation_range": 10,
            "width_shift_range": 0.1,
            "height_shift_range": 0.1,
            "zoom_range": 0.1,
            "brightness_range": (0.8, 1.2),
            "noise_stddev": 10
        }
        augmentation = DataAugmentation(config=augmentation_config)

    # Train model
    print("\n🚀 Starting training...")
    print(f"   Epochs:        {args.epochs}")
    print(f"   Batch size:    {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Augmentation:  {'Enabled' if args.augment else 'Disabled'}")
    print()

    try:
        history = trainer.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            augmentation=augmentation
        )

        # Print training summary
        summary = trainer.get_training_summary()
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"   Epochs trained:      {summary.get('epochs_trained', 'N/A')}")
        print(f"   Final loss:          {summary.get('final_loss', 'N/A'):.4f}")
        print(f"   Final val loss:      {summary.get('final_val_loss', 'N/A'):.4f}")
        print(f"   Best val loss:       {summary.get('best_val_loss', 'N/A'):.4f}")

    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        trainer.save_model(os.path.join(args.output_dir, "model_interrupted.h5"))

    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    metrics_result = trainer.evaluate(test_ds)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics_result, f, indent=2)
    print(f"\n💾 Metrics saved to {metrics_path}")

    # Export model in different formats
    print("\n📦 Exporting model...")
    export_dir = os.path.join(args.output_dir, "exported")

    try:
        trainer.export_model(
            os.path.join(export_dir, "saved_model"),
            format="saved_model"
        )
    except Exception as e:
        logger.warning(f"Failed to export SavedModel: {e}")

    try:
        trainer.export_model(
            os.path.join(export_dir, "tflite"),
            format="tflite"
        )
    except Exception as e:
        logger.warning(f"Failed to export TFLite: {e}")

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print(f"\n📁 Output files:")
    print(f"   Model:       {training_config['model_save_path']}")
    print(f"   Checkpoints: {training_config['checkpoint_dir']}")
    print(f"   Logs:        {training_config['log_dir']}")
    print(f"   Metrics:     {metrics_path}")
    print()


if __name__ == "__main__":
    main()
