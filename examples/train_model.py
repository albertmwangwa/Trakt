#!/usr/bin/env python3
"""
Sample TensorFlow Model Training Script for Trakt OCR

This script demonstrates how to train a custom OCR model using the
training utilities provided by Trakt.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """Main training function."""
    print("=" * 60)
    print("Trakt OCR Model Training")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/5] Initializing data generator...")
    data_gen = DataGenerator(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        image_size=config['input_shape'][:2]
    )
    
    # Load datasets
    print("\n[2/5] Loading training data...")
    train_images, train_labels = data_gen.load_dataset('train')
    
    if train_images is None or len(train_images) == 0:
        print(f"Error: No training data found in {config['data_dir']}/train")
        print("\nExpected directory structure:")
        print("  training_data/")
        print("    train/")
        print("      class1/")
        print("        image1.jpg")
        print("        image2.jpg")
        print("      class2/")
        print("        image1.jpg")
        print("    val/")
        print("      class1/")
        print("      class2/")
        print("    test/")
        print("      class1/")
        print("      class2/")
        return
    
    print(f"Loaded {len(train_images)} training images")
    
    val_images, val_labels = data_gen.load_dataset('val')
    print(f"Loaded {len(val_images)} validation images")
    
    test_images, test_labels = data_gen.load_dataset('test')
    print(f"Loaded {len(test_images)} test images")
    
    # Build model
    print(f"\n[3/5] Building {config['model_type'].upper()} model...")
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
    
    print("\nModel architecture:")
    model.summary()
    
    # Train model
    print(f"\n[4/5] Training model for {config['epochs']} epochs...")
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
    print("\n[5/5] Evaluating model on test set...")
    results = trainer.evaluate(model, (test_images, test_labels))
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final test accuracy: {results['accuracy']:.4f}")
    print(f"Final test loss: {results['loss']:.4f}")
    print(f"Model saved to: {config['save_path']}")
    print("\nTo use this model, update your config.yaml:")
    print(f"  tensorflow:")
    print(f"    model_path: '{config['save_path']}'")
    print(f"    use_pretrained: true")


if __name__ == '__main__':
    main()
