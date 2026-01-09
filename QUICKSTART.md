# Quick Start Guide

Get up and running with Trakt OCR in 5 minutes!

## Prerequisites

- Python 3.8+
- ONVIF-compliant IP camera
- Camera network access

## Step 1: Install Dependencies

### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y tesseract-ocr python3-pip

# Install Python packages
pip install -r requirements.txt
```

### macOS
```bash
# Install system dependencies
brew install tesseract

# Install Python packages
pip install -r requirements.txt
```

### Windows
1. Download and install Tesseract from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Add Tesseract to PATH
3. Install Python packages:
```bash
pip install -r requirements.txt
```

## Step 2: Configure Your Camera

Edit `config.yaml`:

```yaml
camera:
  host: "192.168.1.100"  # Your camera IP
  port: 80               # ONVIF port
  username: "admin"      # Your username
  password: "password"   # Your password
```

## Step 3: (Optional) Enable Advanced Text Detection

For improved accuracy, download and enable the EAST text detector:

```bash
# Download EAST model (95MB)
wget https://github.com/oyyd/frozen-east-text-detection.pb/raw/master/frozen_east_text_detection.pb \
     -O models/frozen_east_text_detection.pb

# Or create models directory and download manually
mkdir -p models
# Then download from the URL above and place in models/
```

Enable in `config.yaml`:

```yaml
ocr:
  use_text_detection: true
  text_detection:
    east_model_path: "./models/frozen_east_text_detection.pb"
```

## Step 4: Run the Application

```bash
python main.py
```

That's it! The application will:
1. Connect to your camera
2. Start processing video frames
3. Detect and log text
4. Save results to `./output/`

## Step 4: View Results

Check the output directory:
```bash
ls -la output/results/  # JSON results
ls -la output/frames/   # Annotated images
tail -f output/trakt.log  # Live logs
```

## Docker Quick Start

Even faster with Docker:

```bash
# Edit config.yaml with your camera settings
nano config.yaml

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f
```

## Testing the Connection

Test your camera connection:

```bash
python -c "
from src.camera_handler import ONVIFCameraHandler
camera = ONVIFCameraHandler('192.168.1.100', 80, 'admin', 'password')
print('Connecting...')
if camera.connect():
    print('✓ Connected successfully!')
    info = camera.get_camera_info()
    print(f'Camera: {info}')
else:
    print('✗ Connection failed')
"
```

## Next Steps

1. **Customize OCR**: Edit `config.yaml` to adjust OCR settings
2. **Add Patterns**: Configure text pattern filters
3. **Try Examples**: Run examples in `examples/` directory:
   - `python examples/basic_ocr.py` - Basic OCR
   - `python examples/license_plate_detection.py` - License plate detection
   - `python examples/text_detection_pipeline.py` - Advanced text detection
4. **Build Custom Model**: See `models/README.md` for TensorFlow models

## Advanced Features

### Text Detection Pipeline

The text detection pipeline uses EAST detector to identify text regions before OCR:

1. **Text Region Detection**: EAST identifies bounding boxes
2. **Preprocessing**: Deskewing and normalization
3. **Region-based OCR**: Each region processed individually
4. **Result Aggregation**: Combined results with improved accuracy

### Preprocessing Options

Configure advanced preprocessing in `config.yaml`:

```yaml
ocr:
  preprocessing:
    apply_deskewing: true              # Correct rotation
    apply_contrast_enhancement: false  # Optional
    apply_noise_removal: false         # Optional
    apply_illumination_normalization: true  # Normalize lighting
```

## Troubleshooting

### Cannot connect to camera
- Verify IP address and port
- Check username/password
- Ensure ONVIF is enabled on camera
- Test with ONVIF Device Manager

### Poor OCR accuracy
- Try different OCR engine (tesseract vs easyocr)
- Adjust confidence threshold
- Enable preprocessing
- Ensure good lighting

### Out of memory
- Lower fps_limit
- Use camera sub-stream
- Disable frame saving
- Use GPU acceleration

## Getting Help

- Check the [README](README.md) for detailed documentation
- Review [examples](examples/) for code samples
- Open an issue on GitHub

## What's Next?

- Run examples: `python examples/basic_ocr.py`
- Customize patterns for license plates, signs, etc.
- Add your own TensorFlow models
- Integrate with other systems via the API

Happy text detecting! 🎉
