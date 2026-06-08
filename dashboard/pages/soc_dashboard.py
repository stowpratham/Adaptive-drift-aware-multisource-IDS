from __future__ import annotations

import streamlit as st

from dashboard.components.alert_panel import render_alert_panel
from dashboard.utils.data_loader import get_drift_chunks, load_stream_metrics, parse_phase_distribution, parse_summary_metrics
from dashboard.utils.helpers import system_health_status


def render() -> None:
    st.title("SOC Dashboard")
    stream_df = load_stream_metrics()
    summary_metrics = parse_summary_metrics()
    drift_chunks = get_drift_chunks()
    phase_stats = parse_phase_distribution()
    status, _ = system_health_status(summary_metrics.get("accuracy"), summary_metrics.get("drift_events"))

    if stream_df is None:
        st.warning("Stream metrics are missing. Run `ids_main.py` first to generate the dashboard inputs.")
        return

    total_chunks = len(stream_df)
    critical = sum(1 for chunk in drift_chunks if chunk > total_chunks * 0.75)
    high = sum(1 for chunk in drift_chunks if total_chunks * 0.5 < chunk <= total_chunks * 0.75)
    medium = sum(1 for chunk in drift_chunks if total_chunks * 0.25 < chunk <= total_chunks * 0.5)
    low = sum(1 for chunk in drift_chunks if chunk <= total_chunks * 0.25)

    alerts = {
        "critical": critical,
        "high": high,
        "medium": medium,
        "low": low,
        "latest_events": ", ".join(str(chunk) for chunk in drift_chunks[-5:]) if drift_chunks else 0,
    }

    render_alert_panel(alerts, status, status)

    st.markdown("---")
    st.markdown("### Operational Metrics")
    st.metric("Final Accuracy", f"{summary_metrics.get('accuracy', 0):.4f}" if summary_metrics.get("accuracy") is not None else "N/A")
    st.metric("Drift Events", summary_metrics.get("drift_events", "N/A"))
    st.metric("Retraining Events", summary_metrics.get("retrain_count", "N/A"))
    st.metric("Stream Chunks", total_chunks)

    st.markdown("---")
    st.subheader("Recent Detection Events")
    if drift_chunks:
        st.write(
            "Latest drift detection chunks: "
            + ", ".join(str(chunk) for chunk in drift_chunks[-10:])
        )
    else:
        st.write("No drift detection events are available in the current outputs.")

    if phase_stats.get("total_samples"):
        st.markdown(
            f"**Current stream composition:** {phase_stats['attack_samples']:,} attack records, "
            f"{phase_stats['benign_samples']:,} benign records."
        )
