"""Build latent feature mappings from trained encoder weights.

This module exposes a single function for converting a trained
FeatureEncoder into human-friendly latent labels and top original
feature mappings suitable for SHAP explainability.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn

LATENT_MAPPING_FILENAME = "latent_feature_mapping.json"


def _encoder_weight_importance(encoder: nn.Module) -> np.ndarray:
    if not isinstance(encoder.encoder[0], nn.Linear) or not isinstance(
        encoder.encoder[2], nn.Linear
    ):
        raise TypeError("Encoder must have Linear layers at positions 0 and 2.")

    w1 = encoder.encoder[0].weight.detach().cpu().numpy()
    w2 = encoder.encoder[2].weight.detach().cpu().numpy()
    return np.abs(w2) @ np.abs(w1)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    total = float(np.sum(scores))
    if total <= 0:
        return scores
    return scores / total


def _display_label_for_latent(index: int, top_features: Sequence[str]) -> str:
    hint = ", ".join(top_features)
    return f"Latent {index:02d} ({hint})"


def build_latent_feature_mapping(
    encoder: nn.Module,
    original_feature_names: Sequence[str],
    top_k: int = 3,
) -> List[dict]:
    """Build a latent-to-original-feature mapping from a trained encoder.

    Args:
        encoder: Trained FeatureEncoder or autoencoder encoder module.
        original_feature_names: List of raw input feature names.
        top_k: Number of top contributing original features per latent dimension.

    Returns:
        List of latent mapping records.
    """
    if len(original_feature_names) == 0:
        raise ValueError("original_feature_names must be non-empty.")

    importances = _encoder_weight_importance(encoder)
    print(importances.shape)

    for i in range(5):
     idx = np.argsort(importances[i])[::-1][:5]
     print(
        f"Latent {i}:",
        [original_feature_names[j] for j in idx]
    )
    latent_dim, input_dim = importances.shape

    if input_dim != len(original_feature_names):
        raise ValueError(
            f"Original feature names length ({len(original_feature_names)}) "
            f"does not match encoder input dim ({input_dim})."
        )

    mapping: List[dict] = []
    for latent_idx in range(latent_dim):
        scores = importances[latent_idx]
        normalized = _normalize_scores(scores)
        ranked = np.argsort(normalized)[::-1][:top_k]
        top_features = [original_feature_names[i] for i in ranked]
        top_scores = [float(normalized[i]) for i in ranked]

        mapping.append(
            {
                "latent_feature": f"Latent {latent_idx:02d}",
                "display_label": _display_label_for_latent(latent_idx, top_features),
                "top_original_features": [
                    {
                        "feature": feature,
                        "importance": round(score, 4),
                    }
                    for feature, score in zip(top_features, top_scores)
                ],
            }
        )
    return mapping


def save_latent_feature_mapping(mapping: List[dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, LATENT_MAPPING_FILENAME)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(mapping, fp, ensure_ascii=False, indent=2)
    return path


__all__ = ["build_latent_feature_mapping", "save_latent_feature_mapping"]
