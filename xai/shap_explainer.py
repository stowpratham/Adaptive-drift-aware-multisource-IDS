"""
SHAP explainer for tree-based IDS models (Random Forest in AdaptiveEnsemble).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import shap

logger = logging.getLogger(__name__)

ShapValues = Union[np.ndarray, List[np.ndarray]]


class ShapExplainer:
    """
    Wraps SHAP TreeExplainer with background sampling and version-safe value extraction.
    """

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: Sequence[str],
        max_background_samples: int = 200,
    ) -> None:
        if background_data.ndim != 2:
            raise ValueError("background_data must be a 2D array (n_samples, n_features).")

        self.model = model
        self.feature_names = list(feature_names)
        self.n_features = background_data.shape[1]

        if len(self.feature_names) != self.n_features:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) "
                f"!= n_features ({self.n_features})."
            )

        self.background = self._subsample_background(
            background_data, max_background_samples
        )
        self.explainer = self._build_explainer()

        logger.info(
            "ShapExplainer ready | background=%s | features=%d",
            self.background.shape,
            self.n_features,
        )

    @staticmethod
    def _subsample_background(X: np.ndarray, max_samples: int) -> np.ndarray:
        if len(X) <= max_samples:
            return X
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        return X[idx]

    def _build_explainer(self) -> shap.TreeExplainer:
        try:
            return shap.TreeExplainer(
                self.model,
                data=self.background,
                feature_perturbation="interventional",
            )
        except Exception as exc:
            logger.warning("Interventional TreeExplainer failed (%s); using model only.", exc)
            return shap.TreeExplainer(self.model)

    def compute_shap_values(self, X: np.ndarray) -> ShapValues:
        """Compute SHAP values for one or more samples."""
        X = self._ensure_2d(X)
        try:
            return self.explainer.shap_values(X)
        except Exception as exc:
            logger.error("SHAP computation failed: %s", exc)
            raise

    def shap_values_for_class(
        self, shap_values: ShapValues, class_index: int
    ) -> np.ndarray:
        """
        Return SHAP matrix (n_samples, n_features) for a target class index.

        Handles SHAP API differences across versions (list vs ndarray).
        """
        if isinstance(shap_values, list):
            if class_index >= len(shap_values):
                raise IndexError(f"class_index {class_index} out of range.")
            return np.asarray(shap_values[class_index])

        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            if class_index >= arr.shape[-1]:
                raise IndexError(f"class_index {class_index} out of range.")
            return arr[..., class_index]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Unexpected SHAP array shape: {arr.shape}")

    def expected_value_for_class(
        self, class_index: int
    ) -> float:
        """Base value (expected model output) for the given class."""
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and len(np.asarray(ev).flatten()) > 1:
            flat = np.asarray(ev).flatten()
            return float(flat[class_index])
        return float(ev)

    def local_contributions(
        self,
        sample: np.ndarray,
        class_index: int,
    ) -> Tuple[np.ndarray, float]:
        """
        SHAP contributions for a single sample and predicted class.

        Returns:
            contributions: shape (n_features,)
            base_value: expected value for the class
        """
        X = self._ensure_2d(sample)
        shap_values = self.compute_shap_values(X)
        matrix = self.shap_values_for_class(shap_values, class_index)
        contributions = matrix[0]
        base_value = self.expected_value_for_class(class_index)
        return contributions, base_value

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            if X.shape[0] != self.n_features:
                raise ValueError(
                    f"Sample has {X.shape[0]} features; expected {self.n_features}."
                )
            return X.reshape(1, -1)
        return X
