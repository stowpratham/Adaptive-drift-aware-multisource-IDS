"""
SHAP visualization helpers — saves plots to results/explainability/.

Headless-only rendering: Agg backend, no GUI, figures saved to disk.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, List, Optional, Sequence, Union

import matplotlib

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import shap

logger = logging.getLogger(__name__)

ShapValues = Union[np.ndarray, List[np.ndarray]]

# Non-interactive backend; never open GUI windows.
plt.ioff()

_plot_lock = threading.Lock()


def close_all_figures() -> None:
    """Thread-safe cleanup of lingering matplotlib figures."""
    with _plot_lock:
        plt.close("all")


@contextmanager
def _headless_figure(
    *args, **kwargs
) -> Generator[plt.Figure, None, None]:
    """Create a figure under lock and always close it on exit."""
    with _plot_lock:
        fig = plt.figure(*args, **kwargs)
        try:
            yield fig
        finally:
            plt.close(fig)


class ExplanationVisualizer:
    """Generate and persist global/local SHAP plots."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save_figure(self, fig: plt.Figure, path: str) -> None:
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        finally:
            plt.close(fig)
            plt.close("all")

    def save_global_importance(
        self,
        shap_values: ShapValues,
        feature_names: Sequence[str],
        class_index: int = 0,
        filename: Optional[str] = None,
    ) -> str:
        """Bar plot of mean |SHAP| across background / evaluation set."""
        matrix = self._matrix_for_class(shap_values, class_index)
        mean_abs = np.abs(matrix).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:20]

        path = os.path.join(
            self.output_dir,
            filename or f"global_importance_class{class_index}_{self._timestamp()}.png",
        )

        with _headless_figure(figsize=(10, 8)) as fig:
            ax = fig.add_subplot(111)
            names = [feature_names[i] for i in order]
            ax.barh(range(len(order)), mean_abs[order][::-1], color="#2563eb")
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(names[::-1])
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"Global Feature Importance (class {class_index})")
            fig.tight_layout()
            self._save_figure(fig, path)

        logger.info("Saved global importance plot: %s", path)
        return path

    def save_summary_plot(
        self,
        shap_values: ShapValues,
        X: np.ndarray,
        feature_names: Sequence[str],
        class_index: int = 0,
        max_display: int = 20,
        filename: Optional[str] = None,
    ) -> str:
        """SHAP beeswarm summary plot."""
        matrix = self._matrix_for_class(shap_values, class_index)
        path = os.path.join(
            self.output_dir,
            filename or f"shap_summary_class{class_index}_{self._timestamp()}.png",
        )

        with _headless_figure(figsize=(10, 8)) as fig:
            shap.summary_plot(
                matrix,
                X,
                feature_names=list(feature_names),
                show=False,
                max_display=max_display,
            )
            active = plt.gcf()
            active.tight_layout()
            self._save_figure(active, path)

        logger.info("Saved SHAP summary plot: %s", path)
        return path

    def save_waterfall_plot(
        self,
        contributions: np.ndarray,
        base_value: float,
        feature_names: Sequence[str],
        sample_values: np.ndarray,
        prediction_label: str,
        filename: Optional[str] = None,
    ) -> str:
        """Waterfall plot for a single prediction."""
        path = os.path.join(
            self.output_dir,
            filename or f"waterfall_{prediction_label}_{self._timestamp()}.png",
        )

        explanation = shap.Explanation(
            values=contributions,
            base_values=base_value,
            data=sample_values,
            feature_names=list(feature_names),
        )

        with _headless_figure(figsize=(10, 8)):
            shap.plots.waterfall(explanation, show=False, max_display=15)
            fig = plt.gcf()
            self._save_figure(fig, path)

        logger.info("Saved waterfall plot: %s", path)
        return path

    def save_force_plot_html(
        self,
        contributions: np.ndarray,
        base_value: float,
        feature_names: Sequence[str],
        sample_values: np.ndarray,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Interactive SHAP force plot saved as HTML (when supported).
        """
        path = os.path.join(
            self.output_dir,
            filename or f"force_plot_{self._timestamp()}.html",
        )
        try:
            shap.initjs()
            force = shap.force_plot(
                base_value,
                contributions,
                sample_values,
                feature_names=list(feature_names),
            )
            shap.save_html(path, force)
            logger.info("Saved force plot HTML: %s", path)
            return path
        except Exception as exc:
            logger.warning("Force plot HTML skipped: %s", exc)
            return None

    @staticmethod
    def _matrix_for_class(shap_values: ShapValues, class_index: int) -> np.ndarray:
        if isinstance(shap_values, list):
            return np.asarray(shap_values[class_index])
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            return arr[..., class_index]
        return arr
