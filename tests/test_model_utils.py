"""
Tests for utils/model_utils.py module.
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import (
    preprocess_image_option_a,
    get_safety_status,
)


class TestPreprocessImageOptionA:
    """Tests for Option A image preprocessing."""
    
    def test_output_shape_correct(self, sample_rgb_image):
        """Preprocessed image should have batch dimension and correct size."""
        result = preprocess_image_option_a(sample_rgb_image, target_size=(320, 320))
        
        assert result.shape == (1, 320, 320, 3)
    
    def test_output_normalized(self, sample_rgb_image):
        """Pixel values should be normalized to 0-1 range."""
        result = preprocess_image_option_a(sample_rgb_image)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_rgba_converted_to_rgb(self, sample_rgba_image):
        """RGBA images should be converted to RGB (3 channels)."""
        result = preprocess_image_option_a(sample_rgba_image)
        
        assert result.shape[-1] == 3  # Should be RGB, not RGBA
    
    def test_grayscale_converted_to_rgb(self, sample_grayscale_image):
        """Grayscale images should be converted to RGB."""
        result = preprocess_image_option_a(sample_grayscale_image)
        
        assert result.shape[-1] == 3
    
    def test_custom_target_size(self, sample_rgb_image):
        """Should respect custom target size."""
        result = preprocess_image_option_a(sample_rgb_image, target_size=(224, 224))
        
        assert result.shape == (1, 224, 224, 3)
    
    def test_output_dtype_is_float(self, sample_rgb_image):
        """Output should be float type for neural network input."""
        result = preprocess_image_option_a(sample_rgb_image)
        
        assert result.dtype in [np.float32, np.float64]


class TestGetSafetyStatus:
    """Tests for safety status determination."""
    
    def test_option_a_violation_detected(self):
        """Should return violation status when head detected."""
        results = {"is_violation": True, "predicted_class": "Head"}
        
        status_text, status_color, status_emoji = get_safety_status(results, 'A')
        
        assert "VIOLATION" in status_text.upper()
        assert status_color == "#e74c3c"  # Red/danger color
    
    def test_option_a_compliant(self):
        """Should return compliant status when helmet detected."""
        results = {"is_violation": False, "predicted_class": "Helmet"}
        
        status_text, status_color, status_emoji = get_safety_status(results, 'A')
        
        assert "COMPLIANT" in status_text.upper()
        assert status_color == "#2ecc71"  # Green/success color
    
    def test_option_b_violations_detected(self):
        """Should report violation count for Option B."""
        results = {
            "has_violations": True,
            "class_counts": {"helmet": 2, "head": 3, "vest": 1}
        }
        
        status_text, status_color, status_emoji = get_safety_status(results, 'B')
        
        assert "VIOLATION" in status_text.upper()
        assert "3" in status_text  # Should mention count
    
    def test_option_b_all_compliant(self):
        """Should report all compliant when no violations."""
        results = {
            "has_violations": False,
            "class_counts": {"helmet": 5, "head": 0, "vest": 3}
        }
        
        status_text, status_color, status_emoji = get_safety_status(results, 'B')
        
        assert "COMPLIANT" in status_text.upper()
        assert status_color == "#2ecc71"


class TestRunInferenceOptionA:
    """Tests for Option A inference (with mocked model)."""
    
    def test_inference_returns_expected_structure(self, sample_rgb_image, option_a_classes):
        """Inference should return dict with expected keys."""
        # Mock the model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.15, 0.85]])
        
        # Import and run with mock
        from utils.model_utils import run_inference_option_a
        
        results = run_inference_option_a(mock_model, sample_rgb_image, option_a_classes)
        
        assert "predicted_class" in results
        assert "confidence" in results
        assert "is_violation" in results
        assert "all_probabilities" in results
    
    def test_inference_confidence_valid_range(self, sample_rgb_image, option_a_classes):
        """Confidence should be between 0 and 1."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.3, 0.7]])
        
        from utils.model_utils import run_inference_option_a
        
        results = run_inference_option_a(mock_model, sample_rgb_image, option_a_classes)
        
        assert 0 <= results["confidence"] <= 1
    
    def test_inference_selects_highest_probability(self, sample_rgb_image, option_a_classes):
        """Should select class with highest probability."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.9, 0.1]])  # 90% Head
        
        from utils.model_utils import run_inference_option_a
        
        results = run_inference_option_a(mock_model, sample_rgb_image, option_a_classes)
        
        assert results["class_id"] == 0  # Head is class 0
        assert results["confidence"] == pytest.approx(0.9, rel=1e-5)
