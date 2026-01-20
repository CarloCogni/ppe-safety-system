# 📦 Asset Setup Guide

This guide explains how to populate the `assets/` folder with files from your Colab notebooks.

## Quick Setup

### Step 1: Run the Colab Notebooks
Run both notebooks completely to generate the export archives:
- `MAICEN_1125_M3U3 - assignment-option A.ipynb` → generates `app_assets.zip`
- `MAICEN_1125_M3U3 - assignment-option B.ipynb` → generates `streamlit_assets.zip`

### Step 2: Download the Exports
From each Colab session, download:
- Option A: `app_assets.zip`
- Option B: `streamlit_assets.zip`

### Step 3: Extract to Assets Folder

```bash
# Extract Option A assets
unzip app_assets.zip -d assets/option_a/

# Extract Option B assets
unzip streamlit_assets.zip -d assets/option_b/
```

## Expected Files

### assets/option_a/
```
├── safety_model.h5           # Keras model weights (REQUIRED)
├── metrics.json              # Performance metrics
├── confusion_matrix.png      # Confusion matrix heatmap
├── training_curves.png       # Accuracy/Loss curves
├── confidence_histogram.png  # Confidence distribution
├── hard_negatives.png        # Failure case analysis
└── sample_images/            # Sample detection images
    ├── head_0.png
    ├── head_1.png
    ├── helmet_0.png
    └── helmet_1.png
```

### assets/option_b/
```
├── best.pt                   # YOLOv8 model weights (REQUIRED)
├── metrics.json              # Performance metrics
├── training_results.png      # YOLO training curves
├── confusion_matrix.png      # Class confusion matrix
├── confidence_histogram.png  # Confidence distribution
├── hard_negatives.png        # Failure case analysis
└── samples/                  # Sample detection images
    └── detection_samples.png
```

## Verification

After setup, verify with:

```bash
ls -la assets/option_a/
ls -la assets/option_b/

# Check model files exist
test -f assets/option_a/safety_model.h5 && echo "✅ Option A model OK"
test -f assets/option_b/best.pt && echo "✅ Option B model OK"
```

## Troubleshooting

### "Model not found" errors
- Verify the model file names match `config.py`
- Check file permissions: `chmod 644 assets/**/*`

### Images not displaying
- PNG file names must match those in `utils/visualization.py`
- If you renamed files, update the display functions accordingly

### Wrong metrics displayed
- Delete placeholder `metrics.json` files before extracting
- Ensure JSON syntax is valid: `python -m json.tool assets/option_a/metrics.json`
