# Output Directory

This directory contains all output from the Trakt OCR application.

## Directory Structure

```
output/
├── results/        # JSON files with detected text
├── frames/         # Annotated image frames
└── trakt.log       # Application log file
```

## Results Format

JSON files in `results/` contain detected text with metadata:

```json
{
  "timestamp": "20260106_103045",
  "frame_number": 150,
  "detections": [
    {
      "text": "PARKING",
      "confidence": 0.95,
      "bbox": [100, 150, 250, 200],
      "matched_pattern": null
    },
    {
      "text": "ABC123",
      "confidence": 0.89,
      "bbox": [300, 400, 450, 450],
      "matched_pattern": "[A-Z]{3}[0-9]{3}"
    }
  ]
}
```

### Field Descriptions

- **timestamp**: When the detection occurred
- **frame_number**: Frame sequence number
- **detections**: Array of detected text regions
  - **text**: Detected text string
  - **confidence**: Confidence score (0-1)
  - **bbox**: Bounding box [x1, y1, x2, y2]
  - **matched_pattern**: Regex pattern matched (if any)

## Annotated Frames

Images in `frames/` show detected text with bounding boxes:

- Green boxes: High confidence (>0.7)
- Yellow boxes: Lower confidence (0.5-0.7)
- Labels show text and confidence score

File naming: `frame_{number}_{timestamp}.jpg`

## Log File

`trakt.log` contains application logs:

```
2026-01-06 10:30:45 - INFO - Starting Trakt OCR Application
2026-01-06 10:30:46 - INFO - Successfully connected to ONVIF camera
2026-01-06 10:30:50 - INFO - Frame 1: Detected 3 text regions
```

Log levels:
- **DEBUG**: Detailed debugging information
- **INFO**: General information
- **WARNING**: Warning messages
- **ERROR**: Error messages

## Storage Management

### Automatic Cleanup

Consider implementing cleanup for old files:

```bash
# Delete results older than 7 days
find output/results/ -name "*.json" -mtime +7 -delete

# Delete frames older than 7 days
find output/frames/ -name "*.jpg" -mtime +7 -delete
```

### Disk Space

Monitor disk usage:

```bash
# Check output directory size
du -sh output/

# Check available disk space
df -h .
```

### Archiving

Archive old results:

```bash
# Create archive of last month's data
tar -czf output_$(date +%Y%m).tar.gz output/

# Move to archive location
mv output_*.tar.gz /path/to/archives/
```

## Analysis

### Parsing Results

Python example to analyze results:

```python
import json
import glob

# Load all results
results = []
for file in glob.glob('output/results/*.json'):
    with open(file) as f:
        results.append(json.load(f))

# Count total detections
total = sum(len(r['detections']) for r in results)
print(f"Total detections: {total}")

# Find most common text
from collections import Counter
texts = []
for r in results:
    texts.extend([d['text'] for d in r['detections']])
common = Counter(texts).most_common(10)
print("Most common:", common)
```

### Exporting to CSV

Convert JSON results to CSV:

```python
import json
import csv

with open('output/detections.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Timestamp', 'Frame', 'Text', 'Confidence'])
    
    for file in glob.glob('output/results/*.json'):
        with open(file) as f:
            data = json.load(f)
            for detection in data['detections']:
                writer.writerow([
                    data['timestamp'],
                    data['frame_number'],
                    detection['text'],
                    detection['confidence']
                ])
```

## Best Practices

1. **Regular Monitoring**: Check logs regularly for errors
2. **Disk Management**: Monitor and clean old files
3. **Backup**: Backup important detections
4. **Analysis**: Periodically analyze results for insights
5. **Security**: Protect sensitive detected information

## Privacy Considerations

- Results may contain sensitive information
- Secure output directory appropriately
- Consider encryption for sensitive data
- Implement data retention policies
- Follow local privacy regulations

## Troubleshooting

### No Results Generated

Check:
- Output configuration in `config.yaml`
- Directory permissions
- Disk space availability
- Application logs for errors

### Missing Frames

Check:
- `save_frames` is enabled in config
- `save_interval` setting
- Disk space
- File permissions

### Large Log Files

Implement log rotation:

```yaml
# config.yaml
output:
  log_file: "./output/trakt.log"
  log_max_size: 10485760  # 10MB
  log_backup_count: 5
```

## Integration

Output can be integrated with:
- Databases (PostgreSQL, MongoDB)
- Analytics platforms (Elasticsearch, Grafana)
- Alert systems (email, SMS)
- Cloud storage (S3, Azure Blob)
- Business applications (custom APIs)
