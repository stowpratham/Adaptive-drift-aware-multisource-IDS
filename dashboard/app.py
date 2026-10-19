from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.components.charts import model_metric_chart
from dashboard.pages import (
    csv_upload,
    drift_analytics,
    explainability,
    model_comparison,
    overview,
    soc_dashboard,
    stream_monitor,
)
from dashboard.utils.helpers import app_style

st.set_page_config(
    page_title="Adaptive Drift-Aware IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

app_style()

PAGES = {
    "System Overview": overview,
    "Stream Monitoring": stream_monitor,
    "Model Comparison": model_comparison,
    "Explainable AI": explainability,
    "Drift Analytics": drift_analytics,
    "SOC Dashboard": soc_dashboard,
    "CSV Upload": csv_upload,
}

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <div>
                <div class="brand-title">ADAPTIVE IDS</div>
                <div class="brand-subtitle">Drift-Aware Security</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio("NAVIGATION", list(PAGES.keys()), index=0)
    st.markdown("---")
    st.caption("Dashboard reads generated IDS artifacts. No model training is performed here.")

PAGES[page].render()
