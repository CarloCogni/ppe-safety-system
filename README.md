# 🏗️ Intelligent Safety Compliance System
### AI-Powered PPE Detection for Construction Sites

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?style=for-the-badge&logo=tensorflow)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge&logo=yolo)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-GPLv3-green?style=for-the-badge)

> *"In construction safety, a False Negative can cause death. A False Positive causes a 2-minute verification. We optimize for the former."*

---

## 🧠 The Project

**Intelligent Safety Compliance System** is a Deep Learning solution that automates Personal Protective Equipment (PPE) detection on construction sites.

This project implements **two distinct approaches** to demonstrate the trade-offs between classification and object detection architectures:

| | Option A | Option B |
|---|---|---|
| **Approach** | Binary Classification | Multi-Class Object Detection |
| **Architecture** | MobileNetV2 (Transfer Learning) | YOLOv8-nano |
| **Classes** | Helmet ✅ / Head ❌ | Helmet ✅ / Head ❌ / Vest 🦺 |
| **Use Case** | Pre-cropped worker images | Full CCTV frame analysis |
| **Key Innovation** | Safety-weighted class balancing | External dataset integration |

---

## 🎯 Key Performance Metrics

### Option A — Classification
| Metric | Value | Interpretation |
|:-------|:-----:|:---------------|
| **Overall Accuracy** | 86.4% | Reliable binary decisions |
| **Head Recall** | 92.1% | Catches 9/10 violations 🛡️ |
| **False Negative Rate** | 7.9% | Only 1 in 12 violations missed |
| **Safety Ratio** | 2.5:1 | 2.5 false alarms per missed violation |

### Option B — Object Detection
| Class | Precision | Recall | AP@0.5 | Status |
|:------|:---------:|:------:|:------:|:-------|
| 🪖 **Helmet** | 95.3% | 88.1% | 95.7% | ✅ Production Ready |
| 👤 **Head** | 93.3% | 88.2% | 94.7% | 🛡️ High Safety Sensitivity |
| 🦺 **Vest** | 95.9% | 96.8% | 98.6% | 🏆 Outstanding |

**Overall mAP@0.5: 96.3%** | **Inference: 2.4ms/image**

---

## 🛠️ Architecture & Innovation

### Option A: Safety-First Classification

1. **Data Cleaning** — Filtered 751 ambiguous "Person" labels
2. **Class Balancing** — Undersampled helmets from 18,966 → 5,785 (1:1 ratio)
3. **Transfer Learning** — MobileNetV2 @ 320×320 (high-res for small objects)
4. **Safety Weighting** — 2× penalty for missed "Head" detections

### Option B: Solving the Missing Label Problem

The original dataset had **zero vest annotations**. Our multi-stage approach:

1. **Attempted:** HSV color-based pseudo-labeling for high-vis detection
2. **Diagnosed:** Too noisy — detected construction equipment, not just vests
3. **Pivoted:** Integrated external Roboflow dataset (2,007 images, 4,601 vest labels)
4. **Balanced:** 3× oversampling of vest images

> This demonstrates real-world ML problem-solving: try → fail → diagnose → adapt.

---

## ⚠️ Limitations & Real-World Testing

While validation metrics are strong, field testing revealed **domain shift** issues:

| Training Data | Real-World Data | Result |
|---------------|-----------------|--------|
| Sleeveless mesh vests | Full high-vis jackets | ❌ Often missed |
| Lime-green dominant | Orange/yellow variations | ⚠️ Lower confidence |
| Clean backgrounds | Busy construction sites | ⚠️ Equipment confusion |

**Key Insight:** High validation metrics ≠ deployment success. Always test on truly independent real-world data.

---

## ⚡ Quick Start

### Option A: Using `uv` (Fastest)

```bash
git clone https://github.com/your-username/ppe-safety-system.git
cd ppe-safety-system

# Initialize virtual environment & install dependencies
uv sync

# Run the app
uv run streamlit run app.py
```

### Option B: Using `pip` (Standard)

```bash
git clone https://github.com/your-username/ppe-safety-system.git
cd ppe-safety-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Usage

1. **Open the App** — Navigate to `http://localhost:8501`
2. **Read the Brief** — Expand the assignment overview
3. **Choose an Approach** — Click Option A or Option B
4. **Explore Results** — View metrics, training curves, confusion matrices
5. **Test Live** — Upload your own construction site images

---

## 📂 Project Structure

```text
ppe_detection_app/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration & constants
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata (for uv)
│
├── utils/
│   ├── __init__.py             # Package exports
│   ├── model_utils.py          # Model loading & inference
│   ├── visualization.py        # Metrics & chart rendering
│   └── ui_components.py        # Reusable styled components
│
└── assets/
    ├── option_a/
    │   ├── safety_model.h5     # Keras classification model
    │   ├── metrics.json        # Performance metrics
    │   ├── confusion_matrix.png
    │   ├── training_curves.png
    │   ├── confidence_histogram.png
    │   └── hard_negatives.png
    │
    └── option_b/
        ├── best.pt             # YOLOv8 detection model
        ├── metrics.json        # Performance metrics
        ├── confusion_matrix.png
        ├── training_results.png
        ├── confidence_histogram.png
        └── hard_negatives.png
```

---

## 🔧 Configuration

Key settings in `config.py`:

```python
# Model paths
OPTION_A_MODEL_PATH = "assets/option_a/safety_model.h5"
OPTION_B_MODEL_PATH = "assets/option_b/best.pt"

# Confidence thresholds
OPTION_A_CONFIDENCE_THRESHOLD = 0.5
OPTION_B_CONFIDENCE_THRESHOLD = 0.25  # Lower = safety-first (prioritize recall)
```

---

## 📊 When to Use Which?

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| Turnstile gate entry | Option A | Single person, fast binary check |
| CCTV monitoring | Option B | Multiple workers, need localization |
| Drone footage | Option B | Wide scenes, many objects |
| Access control | Option A | Binary decision, low latency |
| Audit reporting | Option B | Full detection counts needed |

---

## 🎓 Academic Context

This project was developed as part of  **Master's in Artificial Intelligence for Architecture & Construction** program 
by Zigurat Global Institute of Technology.

**Module:** M3U3 — Machine Learning with Python  
**Assignment:** Intelligent Safety Compliance System  
**Approach:** Both Individual Track (Option A) and Group Track (Option B) implemented

---

## ⚖️ License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**.

You are free to use, modify, and distribute this software, but **all derivative works must remain open source** under the same license.

> ⚠️ **Note:** Commercial closed-source use is **prohibited** without a separate commercial license agreement.

See [LICENSE](LICENSE) for full terms.

---

## 🙏 Acknowledgments

- **Dataset:** Safety Helmet Detection — Original helmet/head annotations
- **External Data:** [Roboflow Construction Safety](https://universe.roboflow.com/) — Vest annotations
- **Frameworks:** TensorFlow, Ultralytics YOLOv8, Streamlit

---

<p align="center">
  <i>Made with ☕ and Python by <b>Carlo Cogni</b></i>
  <br><br>
  <a href="https://github.com/CarloCogni">
    <img src="https://img.shields.io/badge/GitHub-Profile-black?style=flat-square&logo=github" alt="GitHub">
  </a>
</p>
