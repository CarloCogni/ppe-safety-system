"""
Configuration settings for the PPE Detection Streamlit App.
"""

# App metadata
APP_TITLE = "🏗️ Intelligent Safety Compliance System"
APP_SUBTITLE = "AI-Powered PPE Detection for Construction Sites"
APP_AUTHOR = "MSc in AI for Construction"

# Class configurations
OPTION_A_CLASSES = {
    0: {"name": "Head", "color": "#e74c3c", "description": "Worker WITHOUT helmet (violation)"},
    1: {"name": "Helmet", "color": "#2ecc71", "description": "Worker wearing safety helmet"},
}

OPTION_B_CLASSES = {
    0: {"name": "Helmet", "color": "#2ecc71", "description": "Worker wearing safety helmet"},
    1: {"name": "Head", "color": "#e74c3c", "description": "Worker WITHOUT helmet (violation)"},
    2: {"name": "Vest", "color": "#f39c12", "description": "Worker wearing high-visibility vest"},
}

# Model settings
OPTION_A_MODEL_PATH = "assets/option_a/safety_model.h5"
OPTION_B_MODEL_PATH = "assets/option_b/best.pt"

OPTION_A_IMG_SIZE = (320, 320)
OPTION_B_IMG_SIZE = (640, 640)

# Inference settings
OPTION_A_CONFIDENCE_THRESHOLD = 0.5
OPTION_B_CONFIDENCE_THRESHOLD = 0.25  # Lower for safety-first approach

# Color palette for the app
COLORS = {
    "primary": "#1a1a2e",
    "secondary": "#16213e",
    "accent": "#e94560",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "text": "#eaeaea",
    "muted": "#8892b0",
}

# Assignment text (without grading criteria)
ASSIGNMENT_TEXT = """
## Scenario

You are working with a dataset of construction site images. As either the **AI Innovation Team (Group)** 
or a **Specialized Consultant (Individual)** for a construction conglomerate, you are tasked with 
**automating safety compliance**.

The current dataset is focused on helmets and includes labels like "Person", "Head", "Helmet".

### Dataset Overview

| Metric | Value |
|--------|-------|
| Total Images | 5,000 |
| Total Objects | 25,502 |
| Helmet Labels | 18,966 |
| Head Labels | 5,785 |
| Person Labels | 751 |

---

## Option A: Individual Track (Helmet Compliance Focus)

As a solo consultant, your mandate is laser-focused: **Helmet Compliance**. The client needs a 
robust model that can distinguish between a worker wearing a safety helmet (`helmet`) and a 
worker without one (`head`).

**Task:** Clean the data (handling the generic person class) and train a model to detect 
helmet vs head with high precision.

**Constraint:** Address the class imbalance (18,966 helmets vs 5,785 heads) to ensure the 
model does not become biased toward the majority class.

---

## Option B: Group Track (Multi-PPE Expansion)

The client requires a more comprehensive safety solution. They have noticed that while the 
dataset labels helmet and head, it fails to annotate other critical PPE (e.g., Safety Vests 
or Goggles), which are visible in the images but not labelled.

**Task:** In addition to helmet detection, identify at least one additional PPE that is 
not currently annotated in the dataset.

**Strategy:** Devise and implement a strategy to solve the missing label problem 
(e.g., manually annotating a "Golden Set", using a pre-trained model for pseudo-labeling, 
or applying transfer learning).

**Constraint:** The final system must demonstrate the ability to detect both Helmets and 
the new un-annotated PPE class.
"""
