"""
Tests for utils/visualization.py module.
"""

import pytest
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import load_metrics


class TestLoadMetrics:
    """Tests for metrics loading functionality."""
    
    def test_load_valid_metrics_file(self, temp_metrics_file):
        """Should successfully load valid metrics JSON."""
        metrics = load_metrics(temp_metrics_file)
        
        assert metrics is not None
        assert isinstance(metrics, dict)
    
    def test_load_nonexistent_file_returns_none(self):
        """Should return None for non-existent file."""
        metrics = load_metrics("/nonexistent/path/metrics.json")
        
        assert metrics is None
    
    def test_loaded_metrics_contain_expected_keys(self, temp_metrics_file):
        """Loaded metrics should contain expected structure."""
        metrics = load_metrics(temp_metrics_file)
        
        assert "accuracy" in metrics
        assert "head" in metrics
        assert "helmet" in metrics


class TestMetricsStructureOptionA:
    """Tests for Option A metrics structure validation."""
    
    def test_accuracy_in_valid_range(self, option_a_metrics):
        """Accuracy should be between 0 and 1."""
        assert 0 <= option_a_metrics["accuracy"] <= 1
    
    def test_class_metrics_complete(self, option_a_metrics):
        """Each class should have precision, recall, f1_score."""
        for class_name in ["head", "helmet"]:
            class_data = option_a_metrics[class_name]
            assert "precision" in class_data
            assert "recall" in class_data
            assert "f1_score" in class_data
    
    def test_confidence_metrics_present(self, option_a_metrics):
        """Confidence calibration metrics should be present."""
        conf = option_a_metrics["confidence"]
        
        assert "mean_correct" in conf
        assert "mean_incorrect" in conf
        assert "calibration_gap" in conf
    
    def test_calibration_gap_positive(self, option_a_metrics):
        """Calibration gap should be positive (correct > incorrect confidence)."""
        gap = option_a_metrics["confidence"]["calibration_gap"]
        
        assert gap > 0, "Model should be more confident on correct predictions"


class TestMetricsStructureOptionB:
    """Tests for Option B metrics structure validation."""
    
    def test_map_in_valid_range(self, option_b_metrics):
        """mAP should be between 0 and 1."""
        mAP = option_b_metrics["metrics"]["mAP50"]
        
        assert 0 <= mAP <= 1
    
    def test_all_three_classes_present(self, option_b_metrics):
        """Should have metrics for all three classes."""
        per_class = option_b_metrics["metrics"]["per_class"]
        
        assert "helmet" in per_class
        assert "head" in per_class
        assert "vest" in per_class
    
    def test_vest_class_has_high_performance(self, option_b_metrics):
        """Vest class should have high AP (our external dataset solution worked)."""
        vest_ap = option_b_metrics["metrics"]["per_class"]["vest"]["AP50"]
        
        assert vest_ap > 0.9, "Vest detection should be strong after external data integration"
    
    def test_training_config_present(self, option_b_metrics):
        """Training configuration should be documented."""
        config = option_b_metrics["training_config"]
        
        assert "epochs" in config
        assert "img_size" in config
        assert "optimizer" in config


class TestSafetyMetricsValidation:
    """Tests validating safety-critical metrics."""
    
    def test_head_recall_above_threshold(self, option_a_metrics):
        """Head recall should be high (safety-critical: catch violations)."""
        head_recall = option_a_metrics["head"]["recall"]
        
        # We require >85% recall for safety applications
        assert head_recall > 0.85, f"Head recall {head_recall} too low for safety"
    
    def test_false_negative_rate_acceptable(self, option_a_metrics):
        """False negative rate should be below safety threshold."""
        head_recall = option_a_metrics["head"]["recall"]
        fnr = 1 - head_recall
        
        # FNR should be <15% for safety applications
        assert fnr < 0.15, f"False negative rate {fnr:.1%} too high"
    
    def test_safety_ratio_acceptable(self, option_a_metrics):
        """Safety ratio (false alarms per missed violation) should favor caution."""
        head_recall = option_a_metrics["head"]["recall"]
        helmet_recall = option_a_metrics["helmet"]["recall"]
        
        # False alarm rate (helmets misclassified as heads)
        false_alarm_rate = 1 - helmet_recall
        # Miss rate (heads misclassified as helmets)
        miss_rate = 1 - head_recall
        
        if miss_rate > 0:
            safety_ratio = false_alarm_rate / miss_rate
            # We prefer more false alarms than missed violations
            assert safety_ratio > 1, "Should have more false alarms than missed violations"
