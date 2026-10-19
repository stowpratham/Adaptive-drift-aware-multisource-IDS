from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import drift_density_graph
from dashboard.utils.data_loader import get_drift_chunks, load_stream_metrics, parse_phase_distribution, parse_summary_metrics
from dashboard.utils.helpers import system_health_status


def render() -> None:
    st.title("Drift Analytics")
    stream_df = load_stream_metrics()
    if stream_df is None:
        st.warning("Stream metrics are missing. Run `ids_main.py` first to generate `stream_metrics.csv`.")
        return

    summary_metrics = parse_summary_metrics()
    drift_chunks = get_drift_chunks()
    phase_stats = parse_phase_distribution()
    status, _ = system_health_status(summary_metrics.get("accuracy"), summary_metrics.get("drift_events"))

    st.markdown("### Drift Event Summary")
    st.metric("Total Drift Events", summary_metrics.get("drift_events", "N/A"))
    st.metric("Retraining Frequency", f"{(len(drift_chunks) / len(stream_df) * 100):.1f}%" if len(stream_df) else "N/A")
    st.metric("Drift Status", status)

    st.markdown("---")
    if drift_chunks:
        drift_table = pd.DataFrame({"Drift Chunk": drift_chunks})
        st.dataframe(drift_table, hide_index=True)
    else:
        st.info("No drift chunks were found in the available outputs.")

    st.markdown("---")
    st.plotly_chart(drift_density_graph(drift_chunks, len(stream_df)), use_container_width=True)

    st.markdown("---")
    st.subheader("What is Concept Drift?")
    st.markdown(
        "Concept drift occurs when the statistical properties of input data change over time. "
        "In an IDS, drift can happen when normal traffic patterns shift or new attack techniques appear. "
        "Adaptive retraining allows the system to update its decision boundary when drift is detected so the model remains effective against evolving threats."
    )

    st.markdown("### Drift Trend Analysis")
    if phase_stats.get("phases"):
        for index, phase in enumerate(phase_stats["phases"], start=1):
            st.write(
                f"Phase {index}: {phase['samples']:,} records — "
                f"{phase['attack']:,} attack, {phase['normal']:,} benign"
            )
