"""
Reusable UI components for the PPE Detection Streamlit App.
"""

import streamlit as st
from PIL import Image
import io


def render_header():
    """Render the app header with title and description."""
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #e94560; margin-bottom: 0.5rem;">🏗️ Intelligent Safety Compliance System</h1>
        <p style="color: #8892b0; font-size: 1.1rem;">AI-Powered PPE Detection for Construction Sites</p>
        <hr style="border-color: #16213e; margin: 1.5rem 0;">
    </div>
    """, unsafe_allow_html=True)


def render_option_selector():
    """
    Render the option selector (A vs B).
    
    Returns:
        Selected option ('A' or 'B') or None if not selected
    """
    st.markdown("### Choose an Approach")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #2ecc71;
            border-radius: 12px;
            padding: 1.5rem;
            height: 100%;
        ">
            <h3 style="color: #2ecc71; margin-top: 0;">🎯 Option A</h3>
            <p style="color: #eaeaea; font-weight: bold;">Individual Track</p>
            <p style="color: #8892b0; font-size: 0.9rem;">
                Binary Classification<br>
                Helmet vs. Head<br>
                MobileNetV2 Transfer Learning
            </p>
        </div>
        """, unsafe_allow_html=True)
        option_a = st.button("Explore Option A", key="btn_option_a", width="content")
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 2px solid #f39c12;
            border-radius: 12px;
            padding: 1.5rem;
            height: 100%;
        ">
            <h3 style="color: #f39c12; margin-top: 0;">🔍 Option B</h3>
            <p style="color: #eaeaea; font-weight: bold;">Group Track</p>
            <p style="color: #8892b0; font-size: 0.9rem;">
                Object Detection<br>
                Helmet + Head + Vest<br>
                YOLOv8 with External Dataset
            </p>
        </div>
        """, unsafe_allow_html=True)
        option_b = st.button("Explore Option B", key="btn_option_b", width="content")
    
    if option_a:
        return 'A'
    elif option_b:
        return 'B'
    return None


def render_back_button():
    """Render a back to home button."""
    if st.button("← Back to Home", key="back_btn"):
        st.session_state.selected_option = None
        st.rerun()


def render_rationale_card(option: str):
    """
    Render the rationale card for the selected option.
    
    Args:
        option: 'A' or 'B'
    """
    if option == 'A':
        st.markdown("""
        ### 📋 Approach Rationale
        
        **Problem:** The dataset has severe class imbalance (18,966 helmets vs 5,785 heads) 
        and ambiguous "Person" labels.
        
        **Solution Strategy:**
        1. **Filter ambiguous labels** — Remove 751 generic "Person" annotations
        2. **Undersample majority class** — Balance to 5,785 images per class
        3. **Transfer Learning** — Use MobileNetV2 pre-trained on ImageNet
        4. **Safety-first weighting** — Apply 2x class weight to "Head" (violations)
        
        **Key Design Decision:**
        > *"A False Negative (missing a helmetless worker) can cause death. 
        > A False Positive (flagging a helmeted worker) causes a 2-minute verification."*
        
        We intentionally tune for **high recall** on the "Head" class at the cost of some precision.
        """)
    else:
        st.markdown("""
        ### 📋 Approach Rationale
        
        **Problem:** The dataset lacks annotations for Safety Vests — a critical PPE class 
        visible in images but not labeled.
        
        **Solution Strategy:**
        1. **Attempted color-based pseudo-labeling** — HSV detection for high-vis colors
           - Result: Too noisy (detected construction equipment, not just vests)
        2. **External dataset integration** — Merged 2,007 images with 4,601 vest annotations
        3. **YOLOv8 object detection** — Detect multiple objects per image
        4. **Class balancing** — 3x oversampling of vest images
        
        **Key Innovation:**
        > This demonstrates real-world problem-solving: we tried pseudo-labeling, 
        > diagnosed the failure, and pivoted to transfer learning with external data.
        
        Lower confidence threshold (0.25) prioritizes recall for safety applications.
        """)


def render_image_uploader(option: str):
    """
    Render the image uploader component.
    
    Args:
        option: 'A' or 'B'
    
    Returns:
        PIL Image or None
    """
    st.markdown("### 🖼️ Try It Yourself")
    st.markdown("Upload a construction site image to test the model.")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        key=f"uploader_{option}",
        help="Upload a single image to see the model's predictions"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    return None


def render_prediction_result(results: dict, option: str, original_image: Image.Image):
    """
    Render the prediction results.
    
    Args:
        results: Inference results dictionary
        option: 'A' or 'B'
        original_image: Original uploaded image
    """
    from utils.model_utils import get_safety_status
    
    status_text, status_color, status_emoji = get_safety_status(results, option)
    
    # Status banner
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {status_color}22 0%, {status_color}11 100%);
        border: 2px solid {status_color};
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    ">
        <h2 style="color: {status_color}; margin: 0;">{status_text}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if option == 'A':
        # Classification result
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image, caption="Input Image", width="content")
        
        with col2:
            st.markdown("#### Prediction Details")
            st.write(f"**Predicted Class:** {results['predicted_class']}")
            st.write(f"**Confidence:** {results['confidence'] * 100:.1f}%")
            
            st.markdown("#### Class Probabilities")
            for class_name, prob in results['all_probabilities'].items():
                st.progress(prob, text=f"{class_name}: {prob * 100:.1f}%")
    
    else:
        # Object detection result
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image, caption="Original Image", width="content")
        
        with col2:
            st.image(results['annotated_image'], caption="Detections", width="content")
        
        # Detection summary
        st.markdown("#### Detection Summary")
        counts = results['class_counts']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🪖 Helmets", counts.get('helmet', 0))
        with col2:
            st.metric("👤 Heads (Violations)", counts.get('head', 0))
        with col3:
            st.metric("🦺 Vests", counts.get('vest', 0))
            # Show all detections with confidence scores
        if results['detections']:
            st.markdown("#### Detection Details")
            for i, det in enumerate(results['detections']):
                conf_color = "#2ecc71" if det['confidence'] > 0.5 else "#f39c12"
                st.markdown(
                    f"**{i + 1}.** {det['class_name'].title()} — `{det['confidence'] * 100:.1f}%` confidence")


def render_safety_tradeoff():
    """Render the safety trade-off explanation."""
    st.markdown("""
    ---
    ### ⚖️ The Safety Trade-off
    
    In construction safety, **not all errors are equal**:
    
    | Error Type | Example | Consequence |
    |------------|---------|-------------|
    | **False Negative** | Miss a worker without helmet | 🚨 **Potential injury/death** |
    | **False Positive** | Flag a compliant worker | ⚠️ Minor inconvenience (2-min check) |
    
    > *"It is better to stop 100 workers unnecessarily than to let 1 worker enter a danger zone without protection."*
    """)


def apply_custom_css():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
        /* Dark theme base */
        .stApp {
            background-color: #0f0f1a;
        }
        
        /* Metric cards */
        [data-testid="stMetricValue"] {
            color: #e94560;
            font-weight: bold;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #eaeaea !important;
        }
        
        /* Text */
        p, li {
            color: #b8b8d0;
        }
        
        /* Code blocks */
        code {
            background-color: #1a1a2e;
            color: #e94560;
        }
        
        /* Tables */
        table {
            border-collapse: collapse;
            width: 100%;
        }
        
        th {
            background-color: #16213e;
            color: #eaeaea;
            padding: 0.75rem;
        }
        
        td {
            background-color: #1a1a2e;
            color: #b8b8d0;
            padding: 0.75rem;
            border-bottom: 1px solid #2a2a4a;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #e94560 0%, #c73a54 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: bold;
            transition: transform 0.2s;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background-color: #16213e;
            border: 2px dashed #2a2a4a;
            border-radius: 12px;
            padding: 1rem;
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: #e94560;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #16213e;
            color: #eaeaea;
        }
        
        /* Dividers */
        hr {
            border-color: #2a2a4a;
        }
    </style>
    """, unsafe_allow_html=True)
