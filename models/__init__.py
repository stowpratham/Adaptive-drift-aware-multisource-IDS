"""Model wrappers for inference and explainability."""

from models.ensemble_wrapper import EnsemblePredictor, get_explainable_model

__all__ = ["EnsemblePredictor", "get_explainable_model"]
