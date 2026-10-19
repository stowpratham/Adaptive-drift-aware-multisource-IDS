from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
BASELINES_DIR = RESULTS_DIR / "baselines"
EXPLAIN_DIR = RESULTS_DIR / "explainability"
LOG_PATHS = [PROJECT_ROOT / "output_fixed.log", PROJECT_ROOT / "output.log"]
METRICS_PATHS = [PROJECT_ROOT / "models" / "stream_metrics.csv", PROJECT_ROOT / "stream_metrics.csv"]

def _safe_read_text(path: Path) -> Optional[str]:
    """Read text files robustly, including UTF-16 Windows logs."""
    encodings = (
        "utf-16",
        "utf-8-sig",
        "utf-8",
        "cp1252",
    )

    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
            if text:
                return text
        except (UnicodeDecodeError, OSError):
            continue

    return None


@st.cache_data(show_spinner=False)
def load_stream_metrics() -> Optional[pd.DataFrame]:
    for path in METRICS_PATHS:
        if path.exists():
            try:
                df = pd.read_csv(path)
                required = {"chunk", "accuracy", "precision", "recall", "f1"}
                if required.issubset(df.columns):
                    return df.sort_values("chunk").reset_index(drop=True)
            except Exception:
                continue
    return None


@st.cache_data(show_spinner=False)
def load_comparison_results() -> Optional[pd.DataFrame]:
    path = BASELINES_DIR / "final_comparison.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_prediction_explanations() -> Optional[Dict[str, Any]]:
    path = EXPLAIN_DIR / "prediction_explanations.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_latent_mapping() -> Optional[List[Dict[str, Any]]]:
    path = EXPLAIN_DIR / "latent_feature_mapping.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("mapping", [])
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_shap_assets() -> Dict[str, Optional[str]]:
    files = {
        "summary": EXPLAIN_DIR / "shap_summary.png",
        "importance": EXPLAIN_DIR / "global_feature_importance.png",
    }
    return {k: str(v) if v.exists() else None for k, v in files.items()}


@st.cache_data(show_spinner=False)
def load_waterfall_files() -> List[str]:
    if not EXPLAIN_DIR.exists():
        return []
    return sorted(str(p) for p in EXPLAIN_DIR.glob("waterfall_*.png"))


@st.cache_data(show_spinner=False)
def load_log_text() -> Optional[str]:
    for path in LOG_PATHS:
        text = _safe_read_text(path)
        if text:
            return text
    return None


def _number(text: str, patterns: List[str], integer: bool = False) -> Optional[float]:
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                return int(float(raw)) if integer else float(raw)
            except ValueError:
                pass
    return None


@st.cache_data(show_spinner=False)
def parse_summary_metrics() -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    # First preference: persisted baseline evaluation for the Proposed IDS.
    df = load_comparison_results()
    if df is not None and not df.empty and "Model" in df.columns:
        proposed = df[df["Model"].astype(str).str.strip().str.lower() == "proposed ids"]
        if not proposed.empty:
            row = proposed.iloc[0]
            for key in ("Accuracy", "Precision", "Recall", "F1", "Kappa", "FAR", "DetectionRate"):
                if key in row and pd.notna(row[key]):
                    metrics[key.lower()] = float(row[key])

    # Stream metrics are also reliable for stream/chunk counts.
    stream = load_stream_metrics()
    if stream is not None:
        metrics["total_chunks"] = int(stream["chunk"].nunique())
        metrics["last_chunk"] = int(stream["chunk"].max())

    text = load_log_text() or ""

    metrics["drift_events"] = _number(
        text,
        [r"Drift Events\s*:\s*([0-9]+)", r"Total Drift Events\s*:\s*([0-9]+)"],
        integer=True,
    )
    metrics["retrain_count"] = _number(
        text,
        [r"Retrain count\s*:\s*([0-9]+)", r"Retraining Count\s*:\s*([0-9]+)"],
        integer=True,
    )
    metrics["stream_samples"] = _number(
        text,
        [r"Stream samples\s*:\s*([0-9,]+)", r"Total stream samples\s*:\s*([0-9,]+)"],
        integer=True,
    )

    explanations = load_prediction_explanations()
    if explanations:
        meta = explanations.get("metadata", {})
        metrics.setdefault("stream_samples", meta.get("stream_samples"))
        drift_chunks = meta.get("drift_chunks") or []
        if metrics.get("drift_events") is None and drift_chunks:
            metrics["drift_events"] = len(drift_chunks)
        metrics["explained_samples"] = meta.get("explained_samples")

    return metrics


@st.cache_data(show_spinner=False)
def parse_phase_distribution() -> Dict[str, Any]:
    text = load_log_text() or ""
    phase_pattern = re.compile(
        r"Phase\s+\d+\s*:\s*([0-9,]+)\s*samples\s*\(normal=([0-9,]+).*?attack=([0-9,]+)",
        re.IGNORECASE,
    )
    phases = []
    total_attack = total_benign = total_samples = 0
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
        "attack_samples": total_attack or None,
        "benign_samples": total_benign or None,
        "total_samples": total_samples or None,
    }


def check_required_outputs() -> Tuple[bool, List[str]]:
    required = [
        (load_stream_metrics() is not None, "stream_metrics.csv"),
        ((EXPLAIN_DIR / "prediction_explanations.json").exists(), "prediction_explanations.json"),
        ((EXPLAIN_DIR / "shap_summary.png").exists(), "shap_summary.png"),
        ((EXPLAIN_DIR / "global_feature_importance.png").exists(), "global_feature_importance.png"),
    ]
    missing = [name for ok, name in required if not ok]
    return not missing, missing


def get_drift_chunks() -> List[int]:
    explanations = load_prediction_explanations()
    if explanations:
        chunks = explanations.get("metadata", {}).get("drift_chunks", [])
        if chunks:
            return [int(x) for x in chunks]
    text = load_log_text() or ""
    return [int(x) for x in re.findall(r"DRIFT\s*@\s*chunk\s*([0-9]+)", text, re.IGNORECASE)]
