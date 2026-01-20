# PPE Detection - Streamlit Assets

## Contents

| File | Description |
|------|-------------|
| `best.pt` | Trained YOLOv8 model weights |
| `metrics.json` | Model performance metrics |
| `training_results.png` | Training curves |
| `confusion_matrix.png` | Class confusion matrix |
| `samples/` | Sample detection images |

## Usage in Streamlit
```python
from ultralytics import YOLO
import json

# Load model
model = YOLO("best.pt")

# Load metrics
with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Run inference
results = model.predict(image, conf=0.25)
```

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | helmet | Worker wearing safety helmet |
| 1 | head | Worker NOT wearing helmet (violation) |
| 2 | vest | Worker wearing high-visibility vest |

## Recommended Settings

- **Confidence Threshold:** 0.25 (prioritizes recall for safety)
- **Image Size:** 640x640
