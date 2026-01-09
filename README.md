# Trakt - TensorFlow OCR for ONVIF Cameras

A comprehensive TensorFlow-based application template for performing OCR (Optical Character Recognition) on video streams from ONVIF-compliant IP cameras.

## Features

- 🎥 **ONVIF Camera Support**: Connect to any ONVIF-compliant IP camera
- 🔍 **Multiple OCR Engines**: Choose between Tesseract and EasyOCR
- 🎯 **EAST Text Detection**: Optional text region detection for improved accuracy
- 🖼️ **Advanced Preprocessing**: De-skewing, normalization, and illumination correction
- 🤖 **TensorFlow Integration**: Support for custom TensorFlow models
- 📊 **Real-time Processing**: Process video streams in real-time
- 💾 **Configurable Output**: Save detected text and annotated frames
- 🐳 **Docker Support**: Easy deployment with Docker and Docker Compose
- 📝 **Comprehensive Logging**: Detailed logging with colored console output
- ⚙️ **Flexible Configuration**: YAML-based configuration for easy customization

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)

## Requirements

### System Requirements
- Python 3.8 or higher
- Tesseract OCR (for Tesseract engine)
- ONVIF-compliant IP camera

### Hardware Requirements
- Minimum 4GB RAM
- Optional: NVIDIA GPU for TensorFlow acceleration

## Installation

### Standard Installation

1. Clone the repository:
```bash
git clone https://github.com/albertmwangwa/Trakt.git
cd Trakt
```

2. Install Tesseract OCR (if using Tesseract engine):

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download and install from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### GPU Support (Optional)

For GPU acceleration, install TensorFlow with GPU support:

```bash
pip install tensorflow-gpu>=2.13.0
```

Ensure you have CUDA and cuDNN installed according to TensorFlow's requirements.

## Configuration

Edit `config.yaml` to configure your camera and OCR settings:

### Camera Configuration

```yaml
camera:
  host: "192.168.1.100"    # Your camera's IP address
  port: 80                  # ONVIF port
  username: "admin"         # Camera username
  password: "password"      # Camera password
  stream_profile: 0         # 0 = main stream, 1 = sub stream
  transport: "tcp"          # tcp or udp
  fps_limit: 5              # Process N frames per second
```

### OCR Configuration

```yaml
ocr:
  engine: "easyocr"         # Options: "tesseract" or "easyocr"
  confidence_threshold: 0.5 # Minimum confidence (0-1)
  min_text_length: 2        # Minimum characters to detect
```

### Output Configuration

```yaml
output:
  save_results: true        # Save detected text to JSON
  results_dir: "./output/results"
  save_frames: true         # Save annotated frames
  frames_dir: "./output/frames"
  save_interval: 30         # Save every N frames
  log_level: "INFO"         # DEBUG, INFO, WARNING, ERROR
```

## Usage

### Basic Usage

Run the application with default configuration:

```bash
python main.py
```

### Custom Configuration

Use a custom configuration file:

```bash
python main.py --config /path/to/custom_config.yaml
```

### Example Output

The application will:
1. Connect to your ONVIF camera
2. Stream video frames
3. Detect text in each frame
4. Log detected text to console and file
5. Save results and annotated frames (if enabled)

Sample console output:
```
2026-01-06 10:30:45 - INFO - Starting Trakt OCR Application
2026-01-06 10:30:45 - INFO - Connecting to ONVIF camera at 192.168.1.100:80
2026-01-06 10:30:46 - INFO - Successfully connected to ONVIF camera
2026-01-06 10:30:46 - INFO - Stream URI: rtsp://192.168.1.100:554/stream
2026-01-06 10:30:47 - INFO - Initializing EasyOCR with languages: ['en']
2026-01-06 10:30:50 - INFO - Frame 1: Detected 3 text regions
2026-01-06 10:30:50 - INFO -   Text: 'PARKING' (confidence: 0.95)
2026-01-06 10:30:50 - INFO -   Text: 'ABC123' (confidence: 0.89)
```

## Docker Deployment

### Using Docker Compose (Recommended)

1. Update `config.yaml` with your camera settings

2. Build and run:
```bash
docker-compose up -d
```

3. View logs:
```bash
docker-compose logs -f
```

4. Stop the application:
```bash
docker-compose down
```

### Using Docker Only

Build the image:
```bash
docker build -t trakt-ocr .
```

Run the container:
```bash
docker run -d \
  --name trakt-ocr \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/output:/app/output \
  trakt-ocr
```

## Project Structure

```
Trakt/
├── main.py                 # Main application entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── .gitignore              # Git ignore rules
├── .github/
│   └── workflows/          # GitHub Actions CI/CD workflows
│       ├── ci.yml          # Build and test workflow
│       └── code-quality.yml # Linting and security checks
├── src/
│   ├── camera_handler.py   # ONVIF camera connection handler
│   ├── ocr_engine.py       # OCR engine implementations
│   └── web_api.py          # REST API with CORS support
├── web/                    # Web interface files
│   ├── index.html          # Dashboard HTML
│   ├── styles.css          # Dashboard styles
│   └── app.js              # Dashboard JavaScript
├── tests/
│   ├── test_camera_handler.py
│   ├── test_ocr_engine.py
│   └── test_web_api.py     # Web API tests
├── models/                 # Directory for TensorFlow models
├── output/
│   ├── results/           # JSON results
│   ├── frames/            # Annotated frames
│   └── trakt.log          # Application log
└── README.md              # This file
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests/test_camera_handler.py

# Run with verbose output
python -m unittest discover tests -v
```

## Web Interface

Trakt includes a modern web interface for monitoring OCR activity and viewing results.

### Starting the Web Server

```bash
python -m src.web_api
```

The web interface will be available at `http://localhost:5000`.

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/api/health` | GET | Health check endpoint |
| `/api/status` | GET | Application status and statistics |
| `/api/detections` | GET | Recent OCR detections |
| `/api/camera/info` | GET | Camera information |
| `/api/results` | GET | Saved OCR results |

### CORS Support

The API supports Cross-Origin Resource Sharing (CORS), allowing integration with external web applications and services.

## Advanced Usage

### EAST Text Detection for Improved Accuracy

Enable EAST (Efficient and Accurate Scene Text) detector for better text recognition:

1. Download the EAST model:
```bash
wget https://github.com/oyyd/frozen-east-text-detection.pb/raw/master/frozen_east_text_detection.pb \
     -O models/frozen_east_text_detection.pb
```

2. Enable in `config.yaml`:
```yaml
ocr:
  use_text_detection: true
  text_detection:
    east_model_path: "./models/frozen_east_text_detection.pb"
    confidence_threshold: 0.5
    nms_threshold: 0.4
    input_width: 320
    input_height: 320
    region_padding: 5
  preprocessing:
    apply_deskewing: true
    apply_illumination_normalization: true
```

Benefits:
- Detects text regions before OCR
- Handles multiple text orientations
- Improves accuracy on complex scenes
- Reduces false positives

### Advanced Image Preprocessing

Configure preprocessing options for better OCR accuracy:

```yaml
ocr:
  preprocessing:
    apply_deskewing: true              # Correct rotation
    apply_contrast_enhancement: false  # Enhance contrast (optional)
    apply_noise_removal: false         # Remove noise (optional)
    apply_illumination_normalization: true  # Normalize lighting
```

### Custom TensorFlow Models

To use a custom TensorFlow model for OCR:

1. Train your model and save it in `.h5` format
2. Place the model in the `models/` directory
3. Update `config.yaml`:

```yaml
tensorflow:
  model_path: "./models/custom_ocr_model.h5"
  use_pretrained: true
  input_shape: [224, 224, 3]
```

### Text Pattern Filtering

Filter specific patterns (e.g., license plates, numbers):

```yaml
ocr:
  filter_patterns:
    - "[A-Z]{3}[0-9]{3}"     # License plate pattern
    - "[0-9]{3,}"            # Numbers with 3+ digits
    - "\\b[A-Z]{2,}\\b"      # Uppercase words
```

### Multiple Camera Support

To process multiple cameras, create separate configuration files and run multiple instances:

```bash
python main.py --config camera1_config.yaml &
python main.py --config camera2_config.yaml &
```

### Real-time Preview

Enable live preview window:

```yaml
app:
  show_preview: true
  preview_scale: 0.5  # Scale down for performance
```

Press 'q' to quit the preview window.

## Camera Compatibility

This application works with any ONVIF-compliant IP camera. Tested with:

- Axis cameras
- Hikvision cameras
- Dahua cameras
- Generic ONVIF cameras

### Finding Your Camera's ONVIF Settings

1. Check your camera's manual for ONVIF port (usually 80 or 8080)
2. Enable ONVIF in camera settings
3. Use ONVIF Device Manager to discover camera details
4. Note the IP address, port, username, and password

## Troubleshooting

### Camera Connection Issues

**Problem**: Cannot connect to camera
- Verify camera IP address and port
- Ensure ONVIF is enabled on the camera
- Check username and password
- Verify network connectivity

**Problem**: Stream fails to start
- Try changing transport from 'tcp' to 'udp' or vice versa
- Check firewall settings
- Verify RTSP port is accessible

### OCR Issues

**Problem**: Poor text detection accuracy
- Try different OCR engine (tesseract vs easyocr)
- Adjust confidence threshold
- Enable preprocessing
- Ensure adequate lighting in camera view

**Problem**: Out of memory errors
- Lower fps_limit in configuration
- Reduce camera resolution
- Disable frame saving
- Use GPU acceleration

## Performance Tips

1. **Reduce Processing Load**: Lower `fps_limit` to process fewer frames
2. **Use Sub-stream**: Set `stream_profile: 1` for lower resolution
3. **Disable Frame Saving**: Set `save_frames: false` if not needed
4. **GPU Acceleration**: Use GPU for both TensorFlow and EasyOCR
5. **Batch Processing**: Increase batch_size for multiple cameras

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/) - Machine learning framework
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR library
- [Tesseract](https://github.com/tesseract-ocr/tesseract) - OCR engine
- [python-onvif-zeep](https://github.com/FalkTannhaeuser/python-onvif-zeep) - ONVIF client

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## Roadmap

- [x] Web interface for configuration and monitoring
- [x] REST API for integration with other systems
- [ ] Support for multiple simultaneous cameras
- [ ] Database integration for results storage
- [ ] Alert system for specific text patterns
- [ ] Enhanced TensorFlow model training utilities
