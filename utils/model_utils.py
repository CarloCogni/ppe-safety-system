"""
Model utilities for loading and running inference on PPE detection models.
"""

import os
import numpy as np
from PIL import Image
import streamlit as st

# Lazy imports to avoid loading heavy libraries on startup
_tf_model = None
_yolo_model = None


def load_option_a_model(model_path: str):
    """
    Load the TensorFlow/Keras model for Option A (binary classification).
    Uses caching to avoid reloading on every interaction.
    """
    global _tf_model
    
    if _tf_model is not None:
        return _tf_model
    
    try:
        import tensorflow as tf
        _tf_model = tf.keras.models.load_model(model_path)
        return _tf_model
    except Exception as e:
        st.error(f"Failed to load Option A model: {e}")
        return None


def load_option_b_model(model_path: str):
    """
    Load the YOLOv8 model for Option B (object detection).
    Uses caching to avoid reloading on every interaction.
    """
    global _yolo_model
    
    if _yolo_model is not None:
        return _yolo_model
    
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(model_path)
        return _yolo_model
    except Exception as e:
        st.error(f"Failed to load Option B model: {e}")
        return None


def preprocess_image_option_a(image: Image.Image, target_size: tuple = (320, 320)) -> np.ndarray:
    """
    Preprocess an image for Option A model (MobileNetV2 classifier).
    
    Args:
        image: PIL Image object
        target_size: Target dimensions (height, width)
    
    Returns:
        Preprocessed numpy array ready for model input
    """
    # Resize
    img_resized = image.resize(target_size)
    
    # Convert to RGB if necessary
    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')
    
    # Convert to numpy and normalize to 0-1
    img_array = np.array(img_resized) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def run_inference_option_a(model, image: Image.Image, class_names: dict) -> dict:
    """
    Run inference using Option A model (binary classification).
    
    Args:
        model: Loaded Keras model
        image: PIL Image to classify
        class_names: Dictionary mapping class IDs to names
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess
    img_array = preprocess_image_option_a(image)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get predicted class and confidence
    pred_class_idx = int(np.argmax(predictions))
    confidence = float(predictions[pred_class_idx])
    
    # Build results
    results = {
        "predicted_class": class_names[pred_class_idx]["name"],
        "confidence": confidence,
        "class_id": pred_class_idx,
        "is_violation": pred_class_idx == 0,  # Head = violation
        "all_probabilities": {
            class_names[i]["name"]: float(predictions[i]) 
            for i in range(len(predictions))
        }
    }
    
    return results


def run_inference_option_b(model, image: Image.Image, confidence_threshold: float = 0.25) -> dict:
    """
    Run inference using Option B model (YOLOv8 object detection).
    
    Args:
        model: Loaded YOLO model
        image: PIL Image to detect objects in
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        Dictionary with detection results
    """
    # Run YOLO inference
    results = model.predict(image, conf=confidence_threshold, verbose=False)
    result = results[0]
    
    # Parse detections
    detections = []
    class_counts = {"helmet": 0, "head": 0, "vest": 0}
    class_names_map = {0: "helmet", 1: "head", 2: "vest"}
    
    if result.boxes is not None:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_id = int(cls)
            class_name = class_names_map.get(class_id, f"class_{class_id}")
            
            detection = {
                "class_name": class_name,
                "class_id": class_id,
                "confidence": float(conf),
                "bbox": [float(x) for x in box.tolist()],
                "is_violation": class_name == "head"
            }
            detections.append(detection)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Get annotated image
    annotated_img = result.plot()
    annotated_img = annotated_img[:, :, ::-1]  # BGR to RGB
    
    return {
        "detections": detections,
        "num_detections": len(detections),
        "class_counts": class_counts,
        "has_violations": any(d["is_violation"] for d in detections),
        "annotated_image": annotated_img
    }


def get_safety_status(results: dict, option: str) -> tuple:
    """
    Determine overall safety status from inference results.
    
    Args:
        results: Dictionary from run_inference_option_a or run_inference_option_b
        option: 'A' or 'B'
    
    Returns:
        Tuple of (status_text, status_color, status_emoji)
    """
    if option == 'A':
        if results["is_violation"]:
            return ("⚠️ SAFETY VIOLATION", "#e74c3c", "🚨")
        else:
            return ("✅ COMPLIANT", "#2ecc71", "✅")
    else:  # Option B
        if results["has_violations"]:
            violation_count = results["class_counts"].get("head", 0)
            return (f"⚠️ {violation_count} VIOLATION(S) DETECTED", "#e74c3c", "🚨")
        else:
            return ("✅ ALL WORKERS COMPLIANT", "#2ecc71", "✅")
