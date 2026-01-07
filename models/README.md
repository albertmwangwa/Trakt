# Models Directory

This directory is for storing TensorFlow models and EasyOCR models.

## Directory Structure

```
models/
├── custom_ocr_model.h5      # Your custom TensorFlow model
├── easyocr/                  # EasyOCR models (auto-downloaded)
└── pretrained/               # Pre-trained models
```

## Using Custom TensorFlow Models

### Model Requirements

Your custom TensorFlow model should:
- Accept image input (configurable shape, default: 224x224x3)
- Output text predictions or character probabilities
- Be saved in `.h5` or SavedModel format

### Training Your Own Model

Example of creating a custom OCR model:

```python
import tensorflow as tf
from tensorflow import keras

# Define model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_data, train_labels, epochs=10)

# Save model
model.save('models/custom_ocr_model.h5')
```

### Using Your Model

Update `config.yaml`:

```yaml
tensorflow:
  model_path: "./models/custom_ocr_model.h5"
  use_pretrained: true
  input_shape: [224, 224, 3]  # Match your model's input shape
```

## Pre-trained Models

### Download Pre-trained Models

You can use pre-trained models from:

1. **TensorFlow Hub**: https://tfhub.dev/
2. **Keras Applications**: Built-in models
3. **Custom trained models**: Your own models

Example using TensorFlow Hub:

```python
import tensorflow_hub as hub

# Load pre-trained model
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
                   input_shape=(224, 224, 3))
])

model.save('models/pretrained/mobilenet_v2.h5')
```

## EasyOCR Models

EasyOCR models are automatically downloaded on first use to:
- Default: `~/.EasyOCR/model/`
- Custom: Set in `config.yaml` under `ocr.easyocr.model_storage_directory`

Available languages:
- English: 'en'
- French: 'fr'
- German: 'de'
- Spanish: 'es'
- And many more...

## Model Performance Tips

1. **Use GPU**: Enable GPU support for faster inference
2. **Model Size**: Smaller models = faster inference but lower accuracy
3. **Input Resolution**: Higher resolution = better accuracy but slower
4. **Quantization**: Consider model quantization for edge deployment

## Troubleshooting

### Model Loading Errors

If you encounter errors loading your model:
- Ensure model file exists at the specified path
- Verify TensorFlow version compatibility
- Check model format (.h5 or SavedModel)

### Out of Memory

If you run out of memory:
- Reduce batch size
- Use smaller input resolution
- Enable memory growth for GPU
- Use model quantization

## Resources

- [TensorFlow Model Garden](https://github.com/tensorflow/models)
- [Keras Applications](https://keras.io/api/applications/)
- [TensorFlow Hub](https://tfhub.dev/)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
