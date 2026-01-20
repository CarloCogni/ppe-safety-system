"""
PPE Detection Streamlit App - Main Entry Point

A clean, elegant interface for demonstrating two approaches to 
construction site safety compliance using AI.

Author: MSc in AI for Construction
"""

import streamlit as st
import os

# Configure page (must be first Streamlit command)
st.set_page_config(
    page_title="PPE Safety Detection",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import configuration and utilities
from config import (
    APP_TITLE,
    ASSIGNMENT_TEXT,
    OPTION_A_CLASSES,
    OPTION_B_CLASSES,
    OPTION_A_MODEL_PATH,
    OPTION_B_MODEL_PATH,
    OPTION_A_CONFIDENCE_THRESHOLD,
    OPTION_B_CONFIDENCE_THRESHOLD,
)

from utils import (
    # UI Components
    apply_custom_css,
    render_header,
    render_option_selector,
    render_back_button,
    render_rationale_card,
    render_image_uploader,
    render_prediction_result,
    render_safety_tradeoff,
    # Visualization
    load_metrics,
    display_image_with_caption,
    render_metrics_cards_option_a,
    render_metrics_cards_option_b,
    render_confidence_analysis,
    render_training_summary,
    # Model utilities
    load_option_a_model,
    load_option_b_model,
    run_inference_option_a,
    run_inference_option_b,
)


def render_home_page():
    """Render the home page with assignment overview and option selection."""
    render_header()
    
    # Assignment overview in expander
    with st.expander("📜 View Assignment Brief", expanded=False):
        st.markdown(ASSIGNMENT_TEXT)
    
    st.markdown("---")
    
    # Option selection
    selected = render_option_selector()
    
    if selected:
        st.session_state.selected_option = selected
        st.rerun()
    
    # Footer with dual-approach explanation
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #8892b0;">
        <p><strong>Why Two Approaches?</strong></p>
        <p style="max-width: 600px; margin: 0 auto;">
            Option A demonstrates <em>classification</em> — perfect for binary compliance checking.<br>
            Option B showcases <em>object detection</em> — necessary for multi-PPE and localization.<br>
            Both are valid production strategies depending on deployment requirements.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_option_a_page():
    """Render the Option A (Individual Track) page."""
    render_back_button()
    
    st.markdown("# 🎯 Option A: Individual Track")
    st.markdown("### Binary Classification — Helmet vs. Head")
    
    # Rationale
    render_rationale_card('A')
    
    st.markdown("---")
    
    # Load and display metrics
    metrics_path = "assets/option_a/metrics.json"
    metrics = load_metrics(metrics_path)
    
    if metrics:
        render_metrics_cards_option_a(metrics)
        render_confidence_analysis(metrics)
        render_training_summary(metrics)
    else:
        st.warning("Metrics file not found. Please ensure assets/option_a/metrics.json exists.")
    
    st.markdown("---")
    
    # Results visualizations
    st.markdown("### 📈 Training Results")
    
    col1, col2 = st.columns(2)
    with col1:
        display_image_with_caption(
            "assets/option_a/training_curves.png",
            "Training & Validation Curves"
        )
    with col2:
        display_image_with_caption(
            "assets/option_a/confusion_matrix.png",
            "Confusion Matrix"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        display_image_with_caption(
            "assets/option_a/confidence_histogram.png",
            "Confidence Distribution"
        )
    with col2:
        display_image_with_caption(
            "assets/option_a/hard_negatives.png",
            "Hard Negatives Analysis"
        )
    
    st.markdown("---")
    
    # Interactive demo
    image = render_image_uploader('A')
    
    if image is not None:
        with st.spinner("Running inference..."):
            model = load_option_a_model(OPTION_A_MODEL_PATH)
            
            if model is not None:
                results = run_inference_option_a(model, image, OPTION_A_CLASSES)
                render_prediction_result(results, 'A', image)
            else:
                st.error("Model could not be loaded. Please check the model file exists.")
    
    # Safety trade-off discussion
    render_safety_tradeoff()


def render_option_b_page():
    """Render the Option B (Group Track) page."""
    render_back_button()
    
    st.markdown("# 🔍 Option B: Group Track")
    st.markdown("### Object Detection — Helmet + Head + Vest")
    
    # Rationale
    render_rationale_card('B')
    
    st.markdown("---")
    
    # Load and display metrics
    metrics_path = "assets/option_b/metrics.json"
    metrics = load_metrics(metrics_path)
    
    if metrics:
        render_metrics_cards_option_b(metrics)
        render_training_summary(metrics)
    else:
        st.warning("Metrics file not found. Please ensure assets/option_b/metrics.json exists.")
    
    st.markdown("---")
    
    # Results visualizations
    st.markdown("### 📈 Training Results")
    
    col1, col2 = st.columns(2)
    with col1:
        display_image_with_caption(
            "assets/option_b/training_results.png",
            "Training Progress"
        )
    with col2:
        display_image_with_caption(
            "assets/option_b/confusion_matrix.png",
            "Confusion Matrix"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        display_image_with_caption(
            "assets/option_b/confidence_histogram.png",
            "Confidence Distribution"
        )
    with col2:
        display_image_with_caption(
            "assets/option_b/hard_negatives.png",
            "Hard Negatives Analysis"
        )
    
    st.markdown("---")
    
    # Interactive demo
    image = render_image_uploader('B')
    
    if image is not None:
        with st.spinner("Running inference..."):
            model = load_option_b_model(OPTION_B_MODEL_PATH)
            
            if model is not None:
                results = run_inference_option_b(
                    model, 
                    image, 
                    confidence_threshold=OPTION_B_CONFIDENCE_THRESHOLD
                )
                render_prediction_result(results, 'B', image)
            else:
                st.error("Model could not be loaded. Please check the model file exists.")
    
    # Safety trade-off discussion
    render_safety_tradeoff()

    # Limitations section
    with st.expander("⚠️ Limitations & Real-World Testing", expanded=False):
        st.markdown("""
            While validation metrics showed 98.6% AP for vests, real-world testing revealed **domain shift** issues:

            | Training Data | Real-World | Result |
            |---------------|------------|--------|
            | Sleeveless mesh vests | Full high-vis jackets | ❌ Missed |
            | Lime-green dominant | Orange variations | ⚠️ Lower confidence |

            **Why?** The external dataset contained one dominant vest style. Real sites have diverse PPE.

            **Takeaway:** High validation metrics ≠ deployment success. Always test on independent real-world data.
            """)


def main():
    """Main application entry point."""
    # Apply custom styling
    apply_custom_css()
    
    # Initialize session state
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None
    
    # Route to appropriate page
    if st.session_state.selected_option is None:
        render_home_page()
    elif st.session_state.selected_option == 'A':
        render_option_a_page()
    elif st.session_state.selected_option == 'B':
        render_option_b_page()


if __name__ == "__main__":
    main()
