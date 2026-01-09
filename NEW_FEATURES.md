# New Features Guide

This document provides detailed information about the new features added to Trakt OCR.

## Table of Contents

1. [Multi-Camera Support](#multi-camera-support)
2. [Database Integration](#database-integration)
3. [Alert System](#alert-system)
4. [Model Training Utilities](#model-training-utilities)

---

## Multi-Camera Support

### Overview

Trakt now supports processing multiple ONVIF cameras simultaneously within a single application instance. Each camera runs in its own thread with independent OCR processing.

### Configuration

#### Enable Multi-Camera Mode

Edit `config.yaml`:

```yaml
app:
  multi_camera_mode: true
```

#### Define Cameras

Add a `cameras` section with multiple camera configurations:

```yaml
cameras:
  - name: "Front_Entrance"
    host: "192.168.1.100"
    port: 80
    username: "admin"
    password: "password"
    stream_profile: 0
    transport: "tcp"
    fps_limit: 5
    
  - name: "Back_Entrance"
    host: "192.168.1.101"
    port: 80
    username: "admin"
    password: "password"
    stream_profile: 0
    transport: "tcp"
    fps_limit: 5
```

### Features

- **Concurrent Processing**: All cameras run in parallel threads
- **Independent OCR**: Each camera has its own OCR engine instance
- **Per-Camera Settings**: Configure FPS, stream profile, and transport per camera
- **Database Integration**: All detections are tagged with camera ID
- **Status Monitoring**: Real-time status updates for all cameras

### Usage

```bash
python main.py
```

The application will:
1. Initialize all configured cameras
2. Start streaming from each camera
3. Process frames concurrently
4. Store results in the database with camera identification

### API Access

```python
from src.multi_camera_manager import MultiCameraManager

# Get manager status
status = manager.get_status()
print(f"Active cameras: {status['total_cameras']}")

# Get specific camera
camera = manager.get_camera("Front_Entrance")
camera_status = camera.get_status()
```

---

## Database Integration

### Overview

Trakt includes comprehensive database support using SQLAlchemy, allowing you to store and query OCR results, camera configurations, and alert history.

### Supported Databases

- **SQLite** (default, no setup required)
- **PostgreSQL** (for production deployments)
- **MySQL** (alternative production option)

### Configuration

Edit `config.yaml`:

```yaml
database:
  enabled: true
  url: "sqlite:///./output/trakt.db"
```

For PostgreSQL:
```yaml
database:
  url: "postgresql://username:password@localhost:5432/trakt"
```

### Database Schema

#### Cameras Table
- `id`: Primary key
- `name`: Unique camera name
- `host`: Camera IP address
- `port`: Camera port
- `username`: Camera username
- `stream_profile`: Stream profile index
- `active`: Whether camera is active
- `created_at`, `updated_at`: Timestamps

#### Detections Table
- `id`: Primary key
- `camera_id`: Foreign key to cameras
- `frame_number`: Frame number
- `text`: Detected text
- `confidence`: Detection confidence (0-1)
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`: Bounding box coordinates
- `matched_pattern`: Regex pattern that matched (if any)
- `timestamp`: Detection timestamp

#### Alerts Table
- `id`: Primary key
- `detection_id`: Foreign key to detections
- `alert_type`: Type of alert (email, webhook, log)
- `pattern`: Pattern that triggered alert
- `message`: Alert message/data
- `sent`: Whether alert was sent
- `sent_at`: When alert was sent
- `error_message`: Error if sending failed
- `created_at`: Alert creation timestamp

### API Usage

```python
from src.database import DatabaseManager

# Initialize
db = DatabaseManager("sqlite:///./output/trakt.db")

# Add camera
camera = db.add_camera(
    name="Front_Entrance",
    host="192.168.1.100",
    port=80,
    username="admin"
)

# Add detection
detection = db.add_detection(
    camera_id=camera.id,
    frame_number=100,
    text="ABC123",
    confidence=0.95,
    bbox=[10, 20, 100, 50],
    matched_pattern="[A-Z]{3}[0-9]{3}"
)

# Query recent detections
detections = db.get_recent_detections(camera_id=camera.id, limit=50)

# Get statistics
stats = db.get_detection_stats(camera_id=camera.id)
print(f"Total: {stats['total_detections']}")
print(f"Avg confidence: {stats['average_confidence']:.2f}")

# Get active cameras
cameras = db.get_active_cameras()
```

### Migration

To migrate from file-based storage:
1. Enable database in config
2. Run the application once to create schema
3. Historical file-based results will remain in JSON format
4. New detections will be stored in database

---

## Alert System

### Overview

The alert system monitors OCR detections for specific text patterns and triggers notifications through multiple channels.

### Alert Types

1. **Log Alerts**: Write to application log
2. **Email Alerts**: Send email notifications
3. **Webhook Alerts**: HTTP POST/GET to external endpoints

### Configuration

#### Basic Setup

```yaml
alerts:
  enabled: true
  
  # Patterns to monitor (regex)
  patterns:
    - pattern: "[A-Z]{3}[0-9]{3}"
      name: "license_plate"
    - pattern: "EMERGENCY|ALERT|WARNING"
      name: "emergency_text"
  
  # Alert methods
  alert_types:
    - "log"
    - "email"
    - "webhook"
  
  # Anti-spam cooldown (seconds)
  cooldown_seconds: 60
```

#### Email Configuration

```yaml
alerts:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "your-email@gmail.com"
    password: "your-app-password"
    from_email: "your-email@gmail.com"
    to_emails:
      - "recipient@example.com"
```

**Note**: For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833).

#### Webhook Configuration

```yaml
alerts:
  webhook:
    enabled: true
    url: "https://your-server.com/alerts"
    method: "POST"
    headers:
      Content-Type: "application/json"
      Authorization: "Bearer your-token"
    timeout: 10
```

### Webhook Payload

Webhooks receive JSON data:

```json
{
  "text": "ABC123",
  "pattern": "license_plate",
  "camera": "Front_Entrance",
  "timestamp": "2026-01-09T12:00:00.000000"
}
```

### Integration Examples

#### Slack Webhook

```yaml
alerts:
  webhook:
    enabled: true
    url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    method: "POST"
```

#### Discord Webhook

```yaml
alerts:
  webhook:
    enabled: true
    url: "https://discord.com/api/webhooks/YOUR/WEBHOOK"
    method: "POST"
```

#### Custom API

```python
# Your endpoint receives:
@app.route('/alerts', methods=['POST'])
def handle_alert():
    data = request.json
    text = data['text']
    pattern = data['pattern']
    camera = data['camera']
    
    # Process alert
    if pattern == 'license_plate':
        # Check database, send notification, etc.
        pass
    
    return {'status': 'ok'}
```

### Cooldown Behavior

- Prevents duplicate alerts for the same pattern/text/camera
- Configurable cooldown period (default: 60 seconds)
- Independent cooldowns per pattern/text/camera combination

### API Usage

```python
from src.alert_manager import AlertManager

# Initialize
config = {
    'enabled': True,
    'patterns': [
        {'pattern': '[A-Z]{3}[0-9]{3}', 'name': 'license_plate'}
    ],
    'alert_types': ['log', 'email']
}

alert_manager = AlertManager(config, database_manager=db)

# Check text manually
triggered = alert_manager.check_text(
    text="ABC123",
    camera_name="Front_Entrance",
    detection_id=123
)
```

---

## Model Training Utilities

### Overview

Train custom TensorFlow models for specialized OCR tasks using Trakt's built-in training utilities.

### Quick Start

1. **Prepare Data**

   Organize images in directory structure:
   ```
   training_data/
   ├── train/
   │   ├── class1/
   │   │   ├── img1.jpg
   │   │   └── img2.jpg
   │   └── class2/
   │       └── img1.jpg
   ├── val/
   │   └── ...
   └── test/
       └── ...
   ```

2. **Run Training Script**

   ```bash
   python examples/train_model.py
   ```

### Components

#### DataGenerator

Handles data loading and augmentation:

```python
from src.model_trainer import DataGenerator

data_gen = DataGenerator(
    data_dir='./training_data',
    batch_size=32,
    image_size=(224, 224),
    augment=True
)

train_images, train_labels = data_gen.load_dataset('train')
```

Data augmentation includes:
- Random brightness adjustment
- Random contrast adjustment
- Random horizontal flip
- Random rotation (±10 degrees)

#### ModelBuilder

Creates CNN or CRNN architectures:

```python
from src.model_trainer import ModelBuilder

builder = ModelBuilder(config)

# Standard CNN for classification
cnn_model = builder.build_cnn_model(
    input_shape=(224, 224, 3),
    num_classes=36
)

# CRNN for sequence recognition
crnn_model = builder.build_crnn_model(
    input_shape=(32, 128, 1),
    num_classes=37
)
```

#### ModelTrainer

Handles training with callbacks:

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer(config)

history = trainer.train(
    model=model,
    train_data=(train_images, train_labels),
    val_data=(val_images, val_labels),
    epochs=50,
    batch_size=32,
    save_path='./models/my_model.h5'
)
```

Includes automatic:
- Model checkpointing (save best model)
- Early stopping (prevent overfitting)
- Learning rate reduction
- TensorBoard logging

### Training Configuration

```python
config = {
    'data_dir': './training_data',
    'model_type': 'cnn',  # or 'crnn'
    'input_shape': (224, 224, 3),
    'num_classes': 36,
    'batch_size': 32,
    'epochs': 50,
    'save_path': './models/custom_model.h5'
}
```

### Model Architectures

#### CNN Model
- 4 convolutional blocks with batch normalization
- Max pooling for spatial reduction
- Dense layers with dropout for classification
- Suitable for: Character/digit recognition, classification tasks

#### CRNN Model
- Convolutional feature extraction
- Bidirectional LSTM for sequence learning
- Suitable for: Text line recognition, license plates, handwriting

### Using Trained Models

Update `config.yaml`:

```yaml
tensorflow:
  model_path: "./models/custom_model.h5"
  use_pretrained: true
  input_shape: [224, 224, 3]
  batch_size: 1
  gpu_memory_limit: 2048
  allow_growth: true
```

### Evaluation

```python
# Evaluate model
results = trainer.evaluate(
    model=model,
    test_data=(test_images, test_labels)
)

print(f"Test accuracy: {results['accuracy']:.4f}")
print(f"Test loss: {results['loss']:.4f}")
```

### TensorBoard Visualization

View training progress:

```bash
tensorboard --logdir=./logs
```

Access at http://localhost:6006

### Tips

1. **Data Augmentation**: Essential for small datasets
2. **Batch Size**: Larger = faster training, but needs more memory
3. **Early Stopping**: Prevents overfitting, monitors validation loss
4. **Learning Rate**: Reduced automatically when plateau is detected
5. **Input Shape**: Match your camera's typical text region size

---

## Complete Example

### Scenario: License Plate Detection System

1. **Configure Multi-Camera**

   ```yaml
   app:
     multi_camera_mode: true
   
   cameras:
     - name: "Entrance"
       host: "192.168.1.100"
       port: 80
       username: "admin"
       password: "password"
       fps_limit: 5
     - name: "Exit"
       host: "192.168.1.101"
       port: 80
       username: "admin"
       password: "password"
       fps_limit: 5
   ```

2. **Enable Database**

   ```yaml
   database:
     enabled: true
     url: "postgresql://user:pass@localhost/plates"
   ```

3. **Configure Alerts**

   ```yaml
   alerts:
     enabled: true
     patterns:
       - pattern: "[A-Z]{3}[0-9]{3}"
         name: "license_plate"
     alert_types:
       - "log"
       - "webhook"
     webhook:
       enabled: true
       url: "https://your-api.com/plate-detected"
       method: "POST"
   ```

4. **Run Application**

   ```bash
   python main.py
   ```

The system will:
- Monitor both entrance and exit cameras
- Detect license plates using OCR
- Store all detections in PostgreSQL database
- Send webhook alerts when plates are detected
- Maintain 60-second cooldown between duplicate detections

5. **Query Results**

   ```python
   from src.database import DatabaseManager
   
   db = DatabaseManager("postgresql://user:pass@localhost/plates")
   
   # Recent plates from entrance
   entrance_cam = db.get_camera_by_name("Entrance")
   plates = db.get_recent_detections(camera_id=entrance_cam.id, limit=100)
   
   for detection in plates:
       print(f"{detection.text} - {detection.timestamp}")
   ```

---

## Troubleshooting

### Multi-Camera Issues

**Problem**: Cameras fail to start

- Check camera IP addresses and credentials
- Verify ONVIF is enabled on cameras
- Ensure cameras are on same network
- Try reducing FPS limit

**Problem**: High CPU usage

- Reduce FPS for each camera
- Use sub-stream (stream_profile: 1) instead of main stream
- Disable frame saving
- Limit number of concurrent cameras

### Database Issues

**Problem**: Database connection errors

- Verify database URL is correct
- Ensure database server is running
- Check firewall settings
- Test connection with database client

**Problem**: Slow queries

- Add indexes if needed
- Use camera_id filters in queries
- Limit result sets
- Consider PostgreSQL for better performance

### Alert Issues

**Problem**: Email alerts not sending

- Use app-specific password (not account password)
- Enable "Less secure apps" or use OAuth2
- Check SMTP server and port
- Verify firewall allows SMTP traffic

**Problem**: Webhook alerts failing

- Verify URL is accessible
- Check SSL/TLS certificate validity
- Ensure endpoint accepts JSON
- Review webhook timeout setting

### Training Issues

**Problem**: Out of memory errors

- Reduce batch size
- Reduce input image size
- Use GPU if available
- Close other applications

**Problem**: Poor model accuracy

- Collect more training data
- Enable data augmentation
- Increase training epochs
- Try different model architecture

---

## Performance Tuning

### Multi-Camera Performance

```yaml
cameras:
  - name: "HighPriority"
    fps_limit: 10  # Higher FPS for critical cameras
  - name: "LowPriority"
    fps_limit: 2   # Lower FPS to save resources
    stream_profile: 1  # Use sub-stream
```

### Database Performance

```yaml
database:
  # Use connection pooling for better performance
  url: "postgresql://user:pass@localhost/trakt?pool_size=10"
```

### OCR Performance

```yaml
ocr:
  # Faster but less accurate
  engine: "tesseract"
  
  # Or: More accurate but slower
  engine: "easyocr"
  easyocr:
    gpu: true  # Enable GPU acceleration
```

---

## Security Considerations

1. **Credentials**: Store camera passwords securely, use environment variables
2. **Database**: Use strong passwords, enable SSL/TLS for remote databases
3. **Webhooks**: Use HTTPS, implement authentication tokens
4. **Email**: Use app-specific passwords, enable 2FA on email account
5. **Network**: Isolate camera network, use VPN for remote access

---

## Additional Resources

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [ONVIF Specification](https://www.onvif.org/)
- [Python Threading](https://docs.python.org/3/library/threading.html)
