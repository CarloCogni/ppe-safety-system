"""
Tests for config.py module.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    APP_TITLE,
    OPTION_A_CLASSES,
    OPTION_B_CLASSES,
    OPTION_A_IMG_SIZE,
    OPTION_B_IMG_SIZE,
    OPTION_A_CONFIDENCE_THRESHOLD,
    OPTION_B_CONFIDENCE_THRESHOLD,
    COLORS,
    ASSIGNMENT_TEXT,
)


class TestAppConfiguration:
    """Tests for application-level configuration."""
    
    def test_app_title_exists(self):
        """App title should be defined and non-empty."""
        assert APP_TITLE is not None
        assert len(APP_TITLE) > 0
    
    def test_assignment_text_exists(self):
        """Assignment text should be defined and contain key sections."""
        assert ASSIGNMENT_TEXT is not None
        assert "Option A" in ASSIGNMENT_TEXT
        assert "Option B" in ASSIGNMENT_TEXT


class TestOptionAConfig:
    """Tests for Option A (classification) configuration."""
    
    def test_option_a_has_two_classes(self):
        """Option A should have exactly 2 classes (binary classification)."""
        assert len(OPTION_A_CLASSES) == 2
    
    def test_option_a_class_structure(self):
        """Each class should have name, color, and description."""
        for class_id, class_info in OPTION_A_CLASSES.items():
            assert "name" in class_info
            assert "color" in class_info
            assert "description" in class_info
    
    def test_option_a_img_size_is_tuple(self):
        """Image size should be a tuple of two integers."""
        assert isinstance(OPTION_A_IMG_SIZE, tuple)
        assert len(OPTION_A_IMG_SIZE) == 2
        assert all(isinstance(dim, int) for dim in OPTION_A_IMG_SIZE)
    
    def test_option_a_confidence_threshold_valid(self):
        """Confidence threshold should be between 0 and 1."""
        assert 0 < OPTION_A_CONFIDENCE_THRESHOLD <= 1


class TestOptionBConfig:
    """Tests for Option B (object detection) configuration."""
    
    def test_option_b_has_three_classes(self):
        """Option B should have exactly 3 classes (helmet, head, vest)."""
        assert len(OPTION_B_CLASSES) == 3
    
    def test_option_b_includes_vest(self):
        """Option B should include vest class (the 'missing label' we solved)."""
        class_names = [info["name"].lower() for info in OPTION_B_CLASSES.values()]
        assert "vest" in class_names
    
    def test_option_b_confidence_lower_than_a(self):
        """Option B threshold should be lower (safety-first approach)."""
        assert OPTION_B_CONFIDENCE_THRESHOLD < OPTION_A_CONFIDENCE_THRESHOLD
    
    def test_option_b_img_size_larger(self):
        """Option B uses larger images (YOLO standard 640x640)."""
        assert OPTION_B_IMG_SIZE[0] >= OPTION_A_IMG_SIZE[0]


class TestColorPalette:
    """Tests for UI color configuration."""
    
    def test_colors_dict_exists(self):
        """Colors dictionary should be defined."""
        assert COLORS is not None
        assert isinstance(COLORS, dict)
    
    def test_essential_colors_defined(self):
        """Essential UI colors should be defined."""
        essential = ["primary", "accent", "success", "danger"]
        for color_name in essential:
            assert color_name in COLORS
    
    def test_colors_are_hex_format(self):
        """All colors should be valid hex format."""
        for color_name, color_value in COLORS.items():
            assert color_value.startswith("#"), f"{color_name} should start with #"
            assert len(color_value) == 7, f"{color_name} should be 7 chars (#RRGGBB)"
