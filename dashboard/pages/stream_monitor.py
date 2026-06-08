from __future__ import annotations

import streamlit as st

from dashboard.components.charts import (
    accuracy_vs_chunk,
    drift_detection_timeline,
    prediction_distribution_chart,
    retraining_events_timeline,
    f1_vs_chunk,
)
from dashboard.utils.data_loader import (
    get_drift_chunks,
    load_stream_metrics,
    parse_phase_distribution,
)


def render() -> None:
    st.title("Stream Monitoring")
    stream_df = load_stream_metrics()
    if stream_df is None:
        st.warning("Stream metrics are missing. Run `ids_main.py` first to generate `stream_metrics.csv`.")
        return

    drift_chunks = get_drift_chunks()
    phase_stats = parse_phase_distribution()
    dist = {
        "Attack": phase_stats.get("attack_samples", 0) or 0,
        "Benign": phase_stats.get("benign_samples", 0) or 0,
    }

    st.markdown("### Real-Time Stream Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(accuracy_vs_chunk(stream_df), use_container_width=True)
    with col2:
        st.plotly_chart(f1_vs_chunk(stream_df), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(drift_detection_timeline(stream_df, drift_chunks), use_container_width=True)
    st.markdown("---")
    st.plotly_chart(prediction_distribution_chart(dist), use_container_width=True)
    st.markdown("---")
    st.plotly_chart(retraining_events_timeline(len(stream_df), drift_chunks), use_container_width=True)
    st.markdown("---")
    st.write(
        "Use the interactive Plotly controls to zoom into specific stream chunks and inspect drift-driven retraining events in the actual IDS output data."
    )
