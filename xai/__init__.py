"""
Explainable AI (XAI) package for the Adaptive Drift-Aware IDS.

Primary entry points:
  - ExplanationService: orchestrates SHAP explanations and plots
  - generate_explanation: single-sample explanation dict for analysts
"""

from xai.explanation_service import ExplanationService, generate_explanation

__all__ = ["ExplanationService", "generate_explanation", "close_all_figures"]

from xai.visualization import close_all_figures  # noqa: E402
