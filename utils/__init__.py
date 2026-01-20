"""
Utility modules for the PPE Detection Streamlit App.
"""

from utils.model_utils import (
    load_option_a_model,
    load_option_b_model,
    run_inference_option_a,
    run_inference_option_b,
    get_safety_status,
)

from utils.visualization import (
    load_metrics,
    display_image_with_caption,
    render_metrics_cards_option_a,
    render_metrics_cards_option_b,
    render_confidence_analysis,
    render_training_summary,
    display_results_grid,
)

from utils.ui_components import (
    render_header,
    render_option_selector,
    render_back_button,
    render_rationale_card,
    render_image_uploader,
    render_prediction_result,
    render_safety_tradeoff,
    apply_custom_css,
)

__all__ = [
    # Model utils
    "load_option_a_model",
    "load_option_b_model",
    "run_inference_option_a",
    "run_inference_option_b",
    "get_safety_status",
    # Visualization
    "load_metrics",
    "display_image_with_caption",
    "render_metrics_cards_option_a",
    "render_metrics_cards_option_b",
    "render_confidence_analysis",
    "render_training_summary",
    "display_results_grid",
    # UI Components
    "render_header",
    "render_option_selector",
    "render_back_button",
    "render_rationale_card",
    "render_image_uploader",
    "render_prediction_result",
    "render_safety_tradeoff",
    "apply_custom_css",
]
