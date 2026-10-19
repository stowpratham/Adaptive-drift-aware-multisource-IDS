from __future__ import annotations

from typing import Dict, Optional

import streamlit as st


def render_metric_cards(metrics: Dict[str, Optional[object]], status: str, status_color: str, attack_samples: Optional[int], benign_samples: Optional[int], total_samples: Optional[int]) -> None:
    st.markdown(f"### System KPIs — {status}")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Accuracy", metrics.get("Accuracy", "N/A"))
        st.metric("Precision", metrics.get("Precision", "N/A"))
    with cols[1]:
        st.metric("Recall", metrics.get("Recall", "N/A"))
        st.metric("F1 Score", metrics.get("F1 Score", "N/A"))
    with cols[2]:
        st.metric("Kappa Score", metrics.get("Kappa", "N/A"))
        st.metric("Drift Events", metrics.get("drift_events", "N/A"))
    with cols[3]:
        st.metric("Retraining Count", metrics.get("retrain_count", "N/A"))
        st.metric("Total Chunks", metrics.get("total_chunks", "N/A"))

    counts = st.columns(3)
    with counts[0]:
        st.metric("Total Samples", f"{total_samples:,}" if total_samples is not None else "N/A")
    with counts[1]:
        st.metric("Attack Records", f"{attack_samples:,}" if attack_samples is not None else "N/A")
    with counts[2]:
        st.metric("Benign Records", f"{benign_samples:,}" if benign_samples is not None else "N/A")

    st.markdown(f"<div style='margin-top: 16px; font-size: 1rem; color:{status_color}; font-weight:600'>System status: {status}</div>", unsafe_allow_html=True)
