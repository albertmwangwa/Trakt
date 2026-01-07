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

## Step 3: Run the Application

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
3. **Try Examples**: Run examples in `examples/` directory
4. **Build Custom Model**: See `models/README.md` for TensorFlow models

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
