from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None or v == "":
        return "N/A"
    try:
        x = float(v)
        if pct:
            return f"{x * 100:.2f}%" if abs(x) <= 1.5 else f"{x:.2f}%"
        return f"{int(x):,}" if x.is_integer() else f"{x:.2f}"
    except Exception:
        return str(v)


def render_metric_cards(
    metrics: Dict[str, Any],
    status: str,
    status_color: str,
    attack_samples: Optional[int],
    benign_samples: Optional[int],
    total_samples: Optional[int],
) -> None:
    st.markdown("<div class='section-label'>SYSTEM PERFORMANCE</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    cards = [
        ("Accuracy", metrics.get("accuracy"), True),
        ("Precision", metrics.get("precision"), True),
        ("Recall", metrics.get("recall"), True),
        ("F1 Score", metrics.get("f1"), True),
        ("Kappa", metrics.get("kappa"), True),
    ]
    for col, (label, value, pct) in zip(cols, cards):
        with col:
            st.metric(label, _fmt(value, pct))

    st.markdown("<div class='section-label'>STREAM & ADAPTATION</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    items = [
        ("Stream Samples", total_samples, False),
        ("Attack Records", attack_samples, False),
        ("Benign Records", benign_samples, False),
        ("Drift Events", metrics.get("drift_events"), False),
        ("Chunks", metrics.get("total_chunks"), False),
    ]
    for col, (label, value, pct) in zip(cols, items):
        with col:
            st.metric(label, _fmt(value, pct))

    st.markdown(
        f"<div style='margin-top:12px'><span class='status-pill' style='color:{status_color};border-color:{status_color}55'>● {status}</span></div>",
        unsafe_allow_html=True,
    )
