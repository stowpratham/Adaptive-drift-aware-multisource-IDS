from __future__ import annotations

from typing import Dict, Optional, Tuple

import streamlit as st


def app_style() -> None:
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #0f172a;
            color: #e2e8f0;
        }
        .stApp {
            background-color: #0f172a;
            color: #e2e8f0;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
        }
        .stMetricValue {
            color: #ffffff;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #f8fafc;
        }
        .css-1d391kg {
            background-color: #020617;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_percentage(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%" if value <= 1 else f"{value:.2f}"


def format_int(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    return f"{value:,}"


def system_health_status(accuracy: Optional[float], drift_events: Optional[int]) -> Tuple[str, str]:
    if accuracy is None:
        return "Unknown", "gray"
    if accuracy >= 0.85 and (drift_events is None or drift_events <= 50):
        return "Healthy", "green"
    if accuracy >= 0.75 or (drift_events is not None and drift_events <= 130):
        return "Warning", "yellow"
    return "Critical", "red"


def build_status_badge(status: str, color: str) -> str:
    return f"**Status:** <span style='color:{color}; font-weight:bold'>{status}</span>"


def summary_to_display(metrics: Dict[str, Optional[object]]) -> Dict[str, str]:
    return {
        "Accuracy": format_percentage(metrics.get("accuracy") if isinstance(metrics.get("accuracy"), float) else None),
        "Precision": format_percentage(metrics.get("precision") if isinstance(metrics.get("precision"), float) else None),
        "Recall": format_percentage(metrics.get("recall") if isinstance(metrics.get("recall"), float) else None),
        "F1 Score": format_percentage(metrics.get("f1") if isinstance(metrics.get("f1"), float) else None),
        "Kappa": format_percentage(metrics.get("kappa") if isinstance(metrics.get("kappa"), float) else None),
    }
