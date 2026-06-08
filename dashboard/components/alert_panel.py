from __future__ import annotations

from typing import Dict

import streamlit as st


def render_alert_panel(alerts: Dict[str, int], drift_status: str, health_status: str) -> None:
    st.subheader("SOC Alert Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Critical Alerts", alerts.get("critical", 0))
    c2.metric("High Alerts", alerts.get("high", 0))
    c3.metric("Medium Alerts", alerts.get("medium", 0))
    c4.metric("Low Alerts", alerts.get("low", 0))

    st.markdown("---")
    st.markdown(f"**Drift Status:** {drift_status}")
    st.markdown(f"**System Health:** {health_status}")
    st.markdown(f"**Latest Detection Events:** {alerts.get('latest_events', 'N/A')} events")
