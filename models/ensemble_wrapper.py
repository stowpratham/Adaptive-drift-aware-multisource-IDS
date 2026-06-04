"""
Wrappers around the AdaptiveEnsemble for prediction and SHAP explainability.

SHAP TreeExplainer is applied to RandomForest (primary tree-based learner).
The full ensemble is still used for production confidence scores.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# KDD convention used throughout ids_main.py
CLASS_LABELS = {
    0: "attack",
    1: "benign",
}

CLASS_DISPLAY = {
    0: "Malicious / Attack",
    1: "Benign / Normal",
}


def get_explainable_model(ensemble: Any) -> Any:
    """
    Return the sklearn model used for SHAP TreeExplainer.

    Random Forest is chosen because it is fast, stable, and matches the
    user's primary explainable model in the AdaptiveEnsemble.
    """
    if not hasattr(ensemble, "batch_models"):
        raise TypeError("Expected AdaptiveEnsemble with batch_models attribute.")
    if "RandomForest" not in ensemble.batch_models:
        raise KeyError("RandomForest not found in ensemble.batch_models.")
    return ensemble.batch_models["RandomForest"]


class EnsemblePredictor:
    """Unified prediction API for the IDS ensemble."""

    def __init__(self, ensemble: Any) -> None:
        self.ensemble = ensemble
        self.explainable_model = get_explainable_model(ensemble)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ensemble.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.ensemble.predict_proba_all(X)

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        probas = self.predict_proba(X)
        preds = np.argmax(probas, axis=1)
        confidence = probas.max(axis=1)
        return preds, confidence

    def label_name(self, class_id: int) -> str:
        return CLASS_LABELS.get(int(class_id), str(class_id))

    def label_display(self, class_id: int) -> str:
        return CLASS_DISPLAY.get(int(class_id), str(class_id))
