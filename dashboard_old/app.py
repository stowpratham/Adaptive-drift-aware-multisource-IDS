import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dashboard.pages import (
    csv_upload,
    drift_analytics,
    explainability,
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
    "Drift Analytics": drift_analytics,
    "Explainable AI": explainability,
    "CSV Upload": csv_upload,
    "SOC Dashboard": soc_dashboard,
}

with st.sidebar:
    st.title("Adaptive IDS Portal")
    st.markdown(
        "Built on the existing IDS pipeline outputs, this dashboard loads generated results, explainability artifacts, and stream metrics without retraining models."
    )
    page = st.radio("Navigate to", list(PAGES.keys()), index=0)
    st.divider()
    st.caption("Streamlit command: `streamlit run dashboard/app.py`")

PAGES[page].render()
