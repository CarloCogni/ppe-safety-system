"""
Pytest fixtures for PPE Detection tests.
"""

import pytest
import numpy as np
from PIL import Image
import json
import tempfile
import os


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    # Create a 320x320 RGB image with random pixels
    img_array = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')


@pytest.fixture
def sample_rgba_image():
    """Create a sample RGBA image (with alpha channel) for testing."""
    img_array = np.random.randint(0, 255, (320, 320, 4), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGBA')


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    img_array = np.random.randint(0, 255, (320, 320), dtype=np.uint8)
    return Image.fromarray(img_array, mode='L')


@pytest.fixture
def option_a_metrics():
    """Sample metrics dictionary for Option A (classification)."""
    return {
        "accuracy": 0.864,
        "macro_avg_f1": 0.86,
        "total_samples": 2314,
        "head": {
            "precision": 0.83,
            "recall": 0.921,
            "f1_score": 0.87,
            "support": 1157
        },
        "helmet": {
            "precision": 0.911,
            "recall": 0.806,
            "f1_score": 0.86,
            "support": 1157
        },
        "confidence": {
            "mean_overall": 0.836,
            "mean_correct": 0.858,
            "mean_incorrect": 0.691,
            "calibration_gap": 0.167
        },
        "training": {
            "total_epochs": 9,
            "final_val_accuracy": 0.864,
            "final_val_loss": 0.41
        }
    }


@pytest.fixture
def option_b_metrics():
    """Sample metrics dictionary for Option B (object detection)."""
    return {
        "model": "YOLOv8-nano",
        "classes": ["helmet", "head", "vest"],
        "metrics": {
            "mAP50": 0.963,
            "mAP50_95": 0.715,
            "per_class": {
                "helmet": {"precision": 0.953, "recall": 0.881, "AP50": 0.957, "F1": 0.916},
                "head": {"precision": 0.933, "recall": 0.882, "AP50": 0.947, "F1": 0.906},
                "vest": {"precision": 0.959, "recall": 0.968, "AP50": 0.986, "F1": 0.963}
            }
        },
        "training_config": {
            "epochs": 50,
            "img_size": 640,
            "optimizer": "AdamW"
        }
    }


@pytest.fixture
def temp_metrics_file(option_a_metrics):
    """Create a temporary metrics.json file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(option_a_metrics, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def option_a_classes():
    """Class configuration for Option A."""
    return {
        0: {"name": "Head", "color": "#e74c3c", "description": "Worker WITHOUT helmet"},
        1: {"name": "Helmet", "color": "#2ecc71", "description": "Worker wearing helmet"},
    }


@pytest.fixture
def mock_option_a_predictions():
    """Mock prediction output from Option A model."""
    return np.array([[0.15, 0.85]])  # 85% helmet, 15% head


@pytest.fixture
def mock_option_b_detections():
    """Mock detection results from Option B model."""
    return {
        "detections": [
            {"class_name": "helmet", "class_id": 0, "confidence": 0.92, "bbox": [100, 100, 200, 200], "is_violation": False},
            {"class_name": "head", "class_id": 1, "confidence": 0.87, "bbox": [300, 150, 400, 250], "is_violation": True},
            {"class_name": "vest", "class_id": 2, "confidence": 0.95, "bbox": [100, 200, 250, 400], "is_violation": False},
        ],
        "num_detections": 3,
        "class_counts": {"helmet": 1, "head": 1, "vest": 1},
        "has_violations": True,
        "annotated_image": np.zeros((640, 640, 3), dtype=np.uint8)
    }
