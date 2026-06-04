"""
High-level explanation service for security analysts.

Exposes generate_explanation(sample) returning structured JSON-ready dicts
and human-readable narratives.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from models.ensemble_wrapper import EnsemblePredictor
from preprocessing.feature_names import display_name, resolve_feature_names
from xai.shap_explainer import ShapExplainer
from xai.visualization import ExplanationVisualizer, close_all_figures

logger = logging.getLogger(__name__)

DEFAULT_EXPLAIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "explainability",
)

MAX_VISUALIZATION_FILES = 20
GLOBAL_VIS_FILES = 2
TOP_ATTACK_SAMPLES = 5
TOP_BENIGN_SAMPLES = 5

PREDICTION_EXPLANATIONS_JSON = "prediction_explanations.json"
GLOBAL_IMPORTANCE_FILENAME = "global_feature_importance.png"
GLOBAL_SUMMARY_FILENAME = "shap_summary.png"


class ExplanationService:
    """
    Orchestrates predictions, SHAP values, plots, and analyst-facing text.

    Connect after ensemble.fit() on fused latent features (32-d).
    """

    def __init__(
        self,
        ensemble: Any,
        background_data: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        output_dir: str = DEFAULT_EXPLAIN_DIR,
        top_k_features: int = 5,
        save_plots_on_explain: bool = False,
        max_visualization_files: int = MAX_VISUALIZATION_FILES,
    ) -> None:
        self.predictor = EnsemblePredictor(ensemble)
        self.background_data = np.asarray(background_data, dtype=np.float64)
        n_features = self.background_data.shape[1]
        self.feature_names = resolve_feature_names(n_features, feature_names)
        self.top_k = top_k_features
        self.save_plots = save_plots_on_explain
        self.output_dir = output_dir
        self.max_visualization_files = max_visualization_files
        self._vis_files_saved = 0
        self._visualization_paths: List[str] = []

        explainable = self.predictor.explainable_model
        self.shap_explainer = ShapExplainer(
            model=explainable,
            background_data=self.background_data,
            feature_names=self.feature_names,
        )
        self.visualizer = ExplanationVisualizer(output_dir=self.output_dir)
        self._global_plots_generated = False

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("ExplanationService initialized | out=%s", self.output_dir)

    def _can_save_plots(self, count: int = 1) -> bool:
        return self._vis_files_saved + count <= self.max_visualization_files

    def _register_plot(self, path: str) -> None:
        self._vis_files_saved += 1
        self._visualization_paths.append(path)

    def _cleanup_stale_artifacts(self) -> None:
        """Remove prior explainability artifacts before a fresh run."""
        for name in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, name)
            if os.path.isfile(path) and name.endswith((".png", ".html", ".jsonl", ".json")):
                os.remove(path)

    def generate_explanation(
        self,
        sample: Union[np.ndarray, Sequence[float]],
        sample_id: Optional[str] = None,
        attack_type_label: Optional[str] = None,
        save_plots: Optional[bool] = None,
        selection_reason: Optional[str] = None,
        plot_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a full explanation for one fused feature vector.
        """
        X = np.asarray(sample, dtype=np.float64).reshape(1, -1)
        pred, confidence = self.predictor.predict_with_confidence(X)
        class_id = int(pred[0])
        conf = float(confidence[0])

        contributions, base_value = self.shap_explainer.local_contributions(
            X[0], class_index=class_id
        )
        return self._build_explanation_record(
            sample=X[0],
            sample_id=sample_id,
            class_id=class_id,
            confidence=conf,
            contributions=contributions,
            base_value=base_value,
            attack_type_label=attack_type_label,
            selection_reason=selection_reason,
            save_plots=self.save_plots if save_plots is None else save_plots,
            plot_types=plot_types,
        )

    def generate_global_explanations(
        self,
        evaluation_data: Optional[np.ndarray] = None,
        max_samples: int = 500,
        class_index: int = 0,
    ) -> Dict[str, str]:
        """Generate one global summary plot and one global importance plot."""
        self._cleanup_stale_artifacts()
        self._vis_files_saved = 0
        self._visualization_paths = []

        X = evaluation_data if evaluation_data is not None else self.background_data
        X = np.asarray(X, dtype=np.float64)
        if len(X) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=max_samples, replace=False)
            X = X[idx]

        shap_values = self.shap_explainer.compute_shap_values(X)
        paths: Dict[str, str] = {}

        importance_path = self.visualizer.save_global_importance(
            shap_values,
            self.feature_names,
            class_index=class_index,
            filename=GLOBAL_IMPORTANCE_FILENAME,
        )
        paths["global_feature_importance"] = importance_path
        self._register_plot(importance_path)

        summary_path = self.visualizer.save_summary_plot(
            shap_values,
            X,
            self.feature_names,
            class_index=class_index,
            filename=GLOBAL_SUMMARY_FILENAME,
        )
        paths["shap_summary"] = summary_path
        self._register_plot(summary_path)

        self._global_plots_generated = True
        close_all_figures()
        logger.info(
            "Global explainability plots saved (%d/%d files used)",
            self._vis_files_saved,
            self.max_visualization_files,
        )
        return paths

    def generate_stream_explanations(
        self,
        X_stream: np.ndarray,
        drift_chunk_ids: Sequence[int],
        chunk_size: int,
        top_attack: int = 2,
        top_benign: int = 2,
    ) -> Dict[str, Any]:
        """
        Explain a LIMITED subset of stream samples (max 10 total) sequentially.
        
        WINDOWS CRASH FIX:
        - Process at most 10 samples (not all selected)
        - Compute SHAP values one sample at a time (not in batch)
        - Force sequential processing (no parallelism)
        - Clean memory between samples
        - Preserve prediction_explanations.json output
        
        Local plots are limited by the remaining visualization budget (max 20 total).
        """
        import gc
        
        X_stream = np.asarray(X_stream, dtype=np.float64)
        preds, confidences = self.predictor.predict_with_confidence(X_stream)

        selected = self._select_stream_samples(
            predictions=preds,
            confidences=confidences,
            drift_chunk_ids=drift_chunk_ids,
            chunk_size=chunk_size,
            top_attack=min(top_attack, 2),  # Limit to 2 attack samples max
            top_benign=min(top_benign, 2),   # Limit to 2 benign samples max
            n_samples=len(X_stream),
        )
        
        # CRITICAL: Limit to max 10 samples to prevent memory exhaustion
        max_samples_to_explain = 10
        selected = selected[:max_samples_to_explain]
        
        plot_plan = self._build_plot_plan(selected)
        indices = [idx for idx, _ in selected]
        explanations: List[Dict[str, Any]] = []

        # Process each sample INDIVIDUALLY to prevent memory issues
        if indices:
            for sample_row, (global_idx, reason) in enumerate(selected):
                try:
                    # Compute SHAP for single sample (not batch)
                    sample_X = X_stream[global_idx:global_idx+1]  # Shape (1, n_features)
                    shap_values_single = self.shap_explainer.compute_shap_values(sample_X)
                    
                    class_id = int(preds[global_idx])
                    
                    # Extract contributions for this sample from single-sample SHAP result
                    shap_arr = np.asarray(shap_values_single)
                    if isinstance(shap_values_single, list):
                        # Multiple classes: shap_values is list of arrays
                        contributions = np.asarray(shap_values_single[class_id])[0]
                    elif shap_arr.ndim == 3:
                        # 3D array: extract class_id slice, then first sample
                        contributions = shap_arr[0, :, class_id]
                    elif shap_arr.ndim == 2:
                        # 2D array: use directly (first sample)
                        contributions = shap_arr[0]
                    else:
                        # 1D: already contributions
                        contributions = shap_arr
                    
                    base_value = self.shap_explainer.expected_value_for_class(class_id)
                    sample_plots = plot_plan.get(global_idx, [])
                    
                    record = self._build_explanation_record(
                        sample=X_stream[global_idx],
                        sample_id=self._sample_id(global_idx, chunk_size),
                        class_id=class_id,
                        confidence=float(confidences[global_idx]),
                        contributions=contributions,
                        base_value=base_value,
                        selection_reason=reason,
                        save_plots=bool(sample_plots),
                        plot_types=sample_plots,
                        global_index=global_idx,
                        chunk_index=(global_idx // chunk_size) + 1,
                        chunk_local_index=global_idx % chunk_size,
                    )
                    explanations.append(record)
                    
                    # Clean up memory after each sample
                    del shap_values_single, sample_X
                    gc.collect()
                    
                except Exception as exc:
                    logger.warning("Failed to explain sample %d: %s", global_idx, exc)
                    continue

        payload = {
            "metadata": {
                "stream_samples": len(X_stream),
                "explained_samples": len(explanations),
                "selection_policy": {
                    "top_attack": min(top_attack, 2),
                    "top_benign": min(top_benign, 2),
                    "drift_boundaries": True,
                    "max_samples_explained": max_samples_to_explain,
                },
                "drift_chunks": list(drift_chunk_ids),
                "visualization_files": list(self._visualization_paths),
                "visualization_file_count": self._vis_files_saved,
                "visualization_file_limit": self.max_visualization_files,
            },
            "explanations": explanations,
        }

        json_path = os.path.join(self.output_dir, PREDICTION_EXPLANATIONS_JSON)
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        close_all_figures()
        logger.info(
            "Stream explanations saved: %s (%d samples, %d/%d visualization files)",
            json_path,
            len(explanations),
            self._vis_files_saved,
            self.max_visualization_files,
        )
        return payload

    def _build_explanation_record(
        self,
        sample: np.ndarray,
        sample_id: Optional[str],
        class_id: int,
        confidence: float,
        contributions: np.ndarray,
        base_value: float,
        attack_type_label: Optional[str] = None,
        selection_reason: Optional[str] = None,
        save_plots: bool = False,
        plot_types: Optional[List[str]] = None,
        global_index: Optional[int] = None,
        chunk_index: Optional[int] = None,
        chunk_local_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        top_features = self._top_features(sample, contributions)
        feat_contrib = {
            name: float(val) for name, val in zip(self.feature_names, contributions)
        }

        label = self.predictor.label_name(class_id)
        display = self.predictor.label_display(class_id)
        narrative_label = attack_type_label or label

        human = self._build_human_readable(
            predicted_label=label,
            display_label=display,
            narrative_label=narrative_label,
            confidence=confidence,
            top_features=top_features,
            is_malicious=(class_id == 0),
        )

        plot_paths: Dict[str, str] = {}
        safe_id = (sample_id or "sample").replace(" ", "_")

        if save_plots and plot_types:
            for plot_type in plot_types:
                if plot_type == "waterfall" and self._can_save_plots():
                    path = self.visualizer.save_waterfall_plot(
                        contributions=contributions,
                        base_value=base_value,
                        feature_names=self.feature_names,
                        sample_values=sample,
                        prediction_label=label,
                        filename=f"waterfall_{safe_id}.png",
                    )
                    plot_paths["waterfall"] = path
                    self._register_plot(path)
                elif plot_type == "force" and self._can_save_plots():
                    path = self.visualizer.save_force_plot_html(
                        contributions=contributions,
                        base_value=base_value,
                        feature_names=self.feature_names,
                        sample_values=sample,
                        filename=f"force_{safe_id}.html",
                    )
                    if path:
                        plot_paths["force"] = path
                        self._register_plot(path)

        result: Dict[str, Any] = {
            "sample_id": sample_id,
            "global_index": global_index,
            "chunk_index": chunk_index,
            "chunk_local_index": chunk_local_index,
            "selection_reason": selection_reason,
            "prediction": label,
            "prediction_display": display,
            "prediction_class_id": class_id,
            "confidence": round(confidence, 4),
            "top_features": top_features,
            "feature_contributions": feat_contrib,
            "shap_values": dict(feat_contrib),
            "shap_base_value": float(base_value),
            "human_readable_explanation": human,
            "model_explained": "RandomForest",
            "feature_space": "fused_latent_32d",
            "plots": plot_paths,
        }
        return result

    @staticmethod
    def _sample_id(global_index: int, chunk_size: int) -> str:
        chunk = (global_index // chunk_size) + 1
        local_idx = global_index % chunk_size
        return f"chunk_{chunk}_idx_{local_idx}"

    @staticmethod
    def _select_stream_samples(
        predictions: np.ndarray,
        confidences: np.ndarray,
        drift_chunk_ids: Sequence[int],
        chunk_size: int,
        top_attack: int,
        top_benign: int,
        n_samples: int,
    ) -> List[Tuple[int, str]]:
        """Return ordered (global_index, selection_reason) pairs without duplicates."""
        selected: List[Tuple[int, str]] = []
        seen: set[int] = set()

        def add(index: int, reason: str) -> None:
            if index in seen or index < 0 or index >= n_samples:
                return
            seen.add(index)
            selected.append((index, reason))

        attack_indices = np.where(predictions == 0)[0]
        if len(attack_indices):
            order = attack_indices[np.argsort(confidences[attack_indices])[::-1]]
            for idx in order[:top_attack]:
                add(int(idx), "top_attack")

        benign_indices = np.where(predictions == 1)[0]
        if len(benign_indices):
            order = benign_indices[np.argsort(confidences[benign_indices])[::-1]]
            for idx in order[:top_benign]:
                add(int(idx), "top_benign")

        for chunk_num in drift_chunk_ids:
            after_idx = (int(chunk_num) - 1) * chunk_size
            before_idx = after_idx - 1
            add(before_idx, "pre_drift")
            add(after_idx, "post_drift")

        return selected

    def _build_plot_plan(
        self, selected: List[Tuple[int, str]]
    ) -> Dict[int, List[str]]:
        """
        Allocate waterfall/force plots within the remaining visualization budget.

        Waterfalls are prioritized for all selected samples, then force plots.
        """
        remaining = self.max_visualization_files - self._vis_files_saved
        plan: Dict[int, List[str]] = {idx: [] for idx, _ in selected}

        for idx, _ in selected:
            if remaining <= 0:
                break
            plan[idx].append("waterfall")
            remaining -= 1

        for idx, _ in selected:
            if remaining <= 0:
                break
            plan[idx].append("force")
            remaining -= 1

        return plan

    def _top_features(
        self, sample: np.ndarray, contributions: np.ndarray
    ) -> List[Dict[str, Any]]:
        order = np.argsort(np.abs(contributions))[::-1][: self.top_k]
        items: List[Dict[str, Any]] = []
        for idx in order:
            name = self.feature_names[idx]
            shap_val = float(contributions[idx])
            raw_val = float(sample[idx])
            direction = "increased" if shap_val > 0 else "decreased"
            items.append(
                {
                    "feature": name,
                    "display_name": display_name(name),
                    "feature_value": raw_val,
                    "shap_value": round(shap_val, 6),
                    "direction": direction,
                    "impact": "supports predicted class" if shap_val > 0 else "opposes predicted class",
                }
            )
        return items

    def _build_human_readable(
        self,
        predicted_label: str,
        display_label: str,
        narrative_label: str,
        confidence: float,
        top_features: List[Dict[str, Any]],
        is_malicious: bool,
    ) -> str:
        """Build analyst-facing narrative from top SHAP contributors."""
        if not top_features:
            return (
                f"Traffic was classified as {display_label} "
                f"(confidence {confidence:.1%}), but no feature contributions were available."
            )

        parts: List[str] = []
        for item in top_features[:3]:
            fname = item["display_name"]
            shap_val = item["shap_value"]
            fval = item["feature_value"]

            if shap_val > 0:
                if fval > 0:
                    parts.append(f"{fname} was elevated (value={fval:.3f})")
                else:
                    parts.append(f"{fname} strongly pushed toward {predicted_label}")
            else:
                if fval < 0 or "rate" in fname.lower():
                    parts.append(f"{fname} was unusually low")
                else:
                    parts.append(f"{fname} was unusually high")

        feature_sentence = ", ".join(parts[:-1]) + (
            f", and {parts[-1]}" if len(parts) > 1 else parts[0]
        )

        if is_malicious:
            return (
                f"Traffic was classified as {display_label} ({narrative_label}) with "
                f"{confidence:.1%} confidence because {feature_sentence}, "
                f"which together indicate patterns inconsistent with normal baseline behavior."
            )
        return (
            f"Traffic was classified as {display_label} with {confidence:.1%} confidence "
            f"because {feature_sentence}, aligning with expected normal traffic profiles "
            f"in the fused multi-source representation."
        )


_service: Optional[ExplanationService] = None


def configure_explanation_service(service: ExplanationService) -> None:
    """Register the active ExplanationService for generate_explanation()."""
    global _service
    _service = service
    logger.info("Global ExplanationService configured.")


def generate_explanation(
    sample: Union[np.ndarray, Sequence[float]],
    sample_id: Optional[str] = None,
    attack_type_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public API: explain a single sample using the configured ExplanationService.
    """
    if _service is None:
        raise RuntimeError(
            "ExplanationService not configured. "
            "Call configure_explanation_service() after training the ensemble, "
            "or instantiate ExplanationService directly."
        )
    return _service.generate_explanation(
        sample=sample,
        sample_id=sample_id,
        attack_type_label=attack_type_label,
    )
