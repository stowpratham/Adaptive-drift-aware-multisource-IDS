from __future__ import annotations

import importlib
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
SUPPORTED_MODEL_EXTENSIONS = [".pkl", ".joblib", ".sav", ".pt", ".pth"]


@st.cache_resource
def find_saved_model_paths() -> Dict[str, Path]:
    if not MODEL_DIR.exists():
        return {}
    paths: Dict[str, Path] = {}
    for ext in SUPPORTED_MODEL_EXTENSIONS:
        for path in MODEL_DIR.glob(f"*{ext}"):
            key = path.stem
            paths[key] = path
    return paths


@st.cache_resource
def load_model(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    if path.suffix in [".pkl", ".joblib", ".sav"]:
        return joblib.load(path)
    if path.suffix in [".pt", ".pth"]:
        try:
            import torch
            return torch.load(path, map_location="cpu")
        except Exception:
            return None
    return None


def has_saved_models() -> bool:
    return bool(find_saved_model_paths())


def get_saved_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for name, path in find_saved_model_paths().items():
        loaded = load_model(path)
        if loaded is not None:
            models[name] = loaded
    return models


def get_model_status() -> str:
    if has_saved_models():
        return "Saved model artifacts found"
    return "No serialized models available"
