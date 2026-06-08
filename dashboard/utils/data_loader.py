from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
EXPLAIN_DIR = RESULTS_DIR / "explainability"
LOG_PATHS = [BASE_DIR / "output_fixed.log", BASE_DIR / "output.log"]
METRICS_PATHS = [MODEL_DIR / "stream_metrics.csv", BASE_DIR / "stream_metrics.csv"]
SHAP_FILES = {
    "summary": EXPLAIN_DIR / "shap_summary.png",
    "importance": EXPLAIN_DIR / "global_feature_importance.png",
}


def _safe_read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    for encoding in ["utf-16-le", "utf-8", "cp1252"]:
        try:
            return path.read_text(encoding=encoding, errors="replace")
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def load_stream_metrics() -> Optional[pd.DataFrame]:
    for path in METRICS_PATHS:
        if path.exists():
            return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_prediction_explanations() -> Optional[Dict[str, Any]]:
    path = EXPLAIN_DIR / "prediction_explanations.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


@st.cache_data(show_spinner=False)
def load_shap_assets() -> Dict[str, Optional[str]]:
    return {
        "summary": str(SHAP_FILES["summary"]) if SHAP_FILES["summary"].exists() else None,
        "importance": str(SHAP_FILES["importance"]) if SHAP_FILES["importance"].exists() else None,
    }


@st.cache_data(show_spinner=False)
def load_waterfall_files() -> List[str]:
    if not EXPLAIN_DIR.exists():
        return []
    return sorted(
        str(path)
        for ext in ["*.png", "*.html"]
        for path in EXPLAIN_DIR.glob(ext)
    )


@st.cache_data(show_spinner=False)
def load_log_text() -> Optional[str]:
    for path in LOG_PATHS:
        text = _safe_read_text(path)
        if text:
            return text
    return None


def _parse_int(match: Optional[re.Match]) -> Optional[int]:
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def _parse_float(match: Optional[re.Match]) -> Optional[float]:
    if match:
        return float(match.group(1))
    return None


@st.cache_data(show_spinner=False)
def parse_summary_metrics() -> Dict[str, Optional[Any]]:
    text = load_log_text()
    if not text:
        return {}

    metrics: Dict[str, Optional[Any]] = {}
    metrics["accuracy"] = _parse_float(re.search(r"Final Accuracy\s*:\s*([0-9.]+)", text))
    metrics["precision"] = _parse_float(re.search(r"Final Precision\s*:\s*([0-9.]+)", text))
    metrics["recall"] = _parse_float(re.search(r"Final Recall\s*:\s*([0-9.]+)", text))
    metrics["f1"] = _parse_float(re.search(r"Final F1-Score\s*:\s*([0-9.]+)", text))
    metrics["kappa"] = _parse_float(re.search(r"Kappa Score\s*:\s*([0-9.]+)", text))
    metrics["drift_events"] = _parse_int(re.search(r"Drift Events\s*:\s*([0-9]+)", text))
    metrics["retrain_count"] = _parse_int(re.search(r"Retrain count\s*:\s*([0-9]+)", text))
    metrics["total_chunks"] = _parse_int(re.search(r"Total chunks\s*:\s*([0-9]+)", text))
    metrics["stream_metrics_path"] = str(MODEL_DIR / "stream_metrics.csv") if (MODEL_DIR / "stream_metrics.csv").exists() else None
    metrics["stream_samples"] = _parse_int(re.search(r"Stream samples\s*:\s*([0-9,]+)", text))
    return metrics


@st.cache_data(show_spinner=False)
def parse_phase_distribution() -> Dict[str, Any]:
    text = load_log_text()
    if not text:
        return {}

    phase_pattern = re.compile(
        r"Phase\s+\d+\s*:\s*([0-9,]+)\s*samples\s*\(normal=([0-9,]+).*?attack=([0-9,]+).*?%\)",
        re.IGNORECASE,
    )
    phases: List[Dict[str, int]] = []
    total_attack = 0
    total_benign = 0
    total_samples = 0
    for match in phase_pattern.finditer(text):
        samples = int(match.group(1).replace(",", ""))
        normal = int(match.group(2).replace(",", ""))
        attack = int(match.group(3).replace(",", ""))
        phases.append({"samples": samples, "normal": normal, "attack": attack})
        total_attack += attack
        total_benign += normal
        total_samples += samples

    return {
        "phases": phases,
        "attack_samples": total_attack,
        "benign_samples": total_benign,
        "total_samples": total_samples,
    }


def check_required_outputs() -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if not load_stream_metrics() is not None:
        missing.append("stream_metrics.csv")
    if not (EXPLAIN_DIR / "prediction_explanations.json").exists():
        missing.append("results/explainability/prediction_explanations.json")
    if not (EXPLAIN_DIR / "shap_summary.png").exists():
        missing.append("results/explainability/shap_summary.png")
    if not (EXPLAIN_DIR / "global_feature_importance.png").exists():
        missing.append("results/explainability/global_feature_importance.png")
    return (len(missing) == 0, missing)


def get_drift_chunks() -> List[int]:
    explanations = load_prediction_explanations()
    if explanations and "metadata" in explanations:
        return explanations["metadata"].get("drift_chunks", [])
    text = load_log_text()
    if not text:
        return []
    matches = re.findall(r"DRIFT @ chunk\s*([0-9]+)", text)
    return [int(value) for value in matches]
