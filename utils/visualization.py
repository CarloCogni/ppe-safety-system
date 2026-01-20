"""
Visualization utilities for creating charts and displaying metrics.
"""

import json
import os
from typing import Optional
import streamlit as st
from PIL import Image


def load_metrics(metrics_path: str) -> Optional[dict]:
    """
    Load metrics from a JSON file.
    
    Args:
        metrics_path: Path to metrics.json file
    
    Returns:
        Dictionary with metrics or None if file not found
    """
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None


def display_image_with_caption(image_path: str, caption: str = "", use_column_width: bool = True):
    """
    Display an image if it exists, with optional caption.
    
    Args:
        image_path: Path to image file
        caption: Caption text to display
        use_column_width: Whether to scale image to column width
    """
    if os.path.exists(image_path):
        img = Image.open(image_path)
        st.image(img, caption=caption, use_container_width=use_column_width)
    else:
        st.warning(f"Image not found: {os.path.basename(image_path)}")


def render_metrics_cards_option_a(metrics: dict):
    """
    Render metrics cards for Option A (classification metrics).
    """
    st.markdown("### 📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Accuracy",
            value=f"{metrics.get('accuracy', 0) * 100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Macro F1 Score",
            value=f"{metrics.get('macro_avg_f1', 0) * 100:.1f}%"
        )
    
    head_metrics = metrics.get('head', {})
    with col3:
        st.metric(
            label="Head Recall",
            value=f"{head_metrics.get('recall', 0) * 100:.1f}%",
            help="Critical for safety: catches helmet violations"
        )
    
    helmet_metrics = metrics.get('helmet', {})
    with col4:
        st.metric(
            label="Helmet Precision",
            value=f"{helmet_metrics.get('precision', 0) * 100:.1f}%"
        )


def render_metrics_cards_option_b(metrics: dict):
    """
    Render metrics cards for Option B (object detection metrics).
    """
    st.markdown("### 📊 Model Performance")
    
    metrics_data = metrics.get('metrics', {})
    per_class = metrics_data.get('per_class', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="mAP@0.5",
            value=f"{metrics_data.get('mAP50', 0) * 100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="mAP@0.5:0.95",
            value=f"{metrics_data.get('mAP50_95', 0) * 100:.1f}%"
        )
    
    with col3:
        vest_data = per_class.get('vest', {})
        st.metric(
            label="Vest AP@0.5",
            value=f"{vest_data.get('AP50', 0) * 100:.1f}%",
            help="New class detected through external dataset"
        )
    
    # Per-class breakdown
    st.markdown("#### Per-Class Performance")
    
    cols = st.columns(3)
    class_order = ['helmet', 'head', 'vest']
    class_emojis = {'helmet': '🪖', 'head': '👤', 'vest': '🦺'}
    
    for i, class_name in enumerate(class_order):
        class_data = per_class.get(class_name, {})
        with cols[i]:
            st.markdown(f"**{class_emojis.get(class_name, '')} {class_name.title()}**")
            st.write(f"Precision: {class_data.get('precision', 0) * 100:.1f}%")
            st.write(f"Recall: {class_data.get('recall', 0) * 100:.1f}%")
            st.write(f"F1: {class_data.get('F1', 0) * 100:.1f}%")


def render_confidence_analysis(metrics: dict):
    """
    Render confidence calibration analysis for Option A.
    """
    confidence = metrics.get('confidence', {})
    
    if not confidence:
        return
    
    st.markdown("### 🎯 Confidence Calibration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Mean Confidence (Correct)",
            value=f"{confidence.get('mean_correct', 0) * 100:.1f}%",
            delta="Model is certain when right"
        )
    
    with col2:
        st.metric(
            label="Mean Confidence (Incorrect)",
            value=f"{confidence.get('mean_incorrect', 0) * 100:.1f}%",
            delta="Model hesitates when unsure"
        )
    
    with col3:
        gap = confidence.get('calibration_gap', 0) * 100
        st.metric(
            label="Calibration Gap",
            value=f"{gap:.1f}%",
            help="Healthy separation = good uncertainty estimation"
        )


def render_training_summary(metrics: dict):
    """
    Render training configuration summary.
    """
    training = metrics.get('training', {}) or metrics.get('training_config', {})
    
    if not training:
        return
    
    st.markdown("### ⚙️ Training Configuration")
    
    # Handle different metric structures
    if 'total_epochs' in training:
        # Option A format
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Epochs:** {training.get('total_epochs', 'N/A')}")
        with col2:
            st.write(f"**Final Val Acc:** {training.get('final_val_accuracy', 0) * 100:.1f}%")
        with col3:
            st.write(f"**Final Val Loss:** {training.get('final_val_loss', 0):.4f}")
    else:
        # Option B format
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Epochs:** {training.get('epochs', 'N/A')}")
        with col2:
            st.write(f"**Image Size:** {training.get('img_size', 'N/A')}")
        with col3:
            st.write(f"**Optimizer:** {training.get('optimizer', 'N/A')}")


def display_results_grid(assets_path: str, image_names: list, columns: int = 2):
    """
    Display a grid of result images.
    
    Args:
        assets_path: Base path to assets folder
        image_names: List of image filenames to display
        columns: Number of columns in grid
    """
    cols = st.columns(columns)
    
    for idx, img_name in enumerate(image_names):
        img_path = os.path.join(assets_path, img_name)
        if os.path.exists(img_path):
            with cols[idx % columns]:
                img = Image.open(img_path)
                st.image(img, caption=img_name.replace('.png', '').replace('_', ' ').title())
