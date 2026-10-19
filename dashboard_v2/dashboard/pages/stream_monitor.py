from __future__ import annotations

import streamlit as st

from dashboard.components.charts import (
    all_stream_metrics,
    drift_detection_timeline,
    drift_density_graph,
    retraining_events_timeline,
    stream_performance,
)
from dashboard.utils.data_loader import get_drift_chunks, load_stream_metrics


def render() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Streaming Evaluation</div>
            <div class="hero-title">Stream Monitoring</div>
            <p class="hero-text">Chunk-by-chunk performance from the persisted stream_metrics.csv artifact.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_stream_metrics()
    if df is None:
        st.error("stream_metrics.csv could not be loaded.")
        return

    drift_chunks = get_drift_chunks()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", int(df["chunk"].nunique()))
    c2.metric("Latest Accuracy", f"{df.iloc[-1]['accuracy']:.2%}")
    c3.metric("Latest F1", f"{df.iloc[-1]['f1']:.2%}")
    c4.metric("Reported Drift Events", len(drift_chunks))

    st.plotly_chart(all_stream_metrics(df), use_container_width=True)

    tabs = st.tabs(["Accuracy", "Precision", "Recall", "F1", "Drift"])
    metrics = ["accuracy", "precision", "recall", "f1"]
    for tab, metric in zip(tabs[:4], metrics):
        with tab:
            st.plotly_chart(
                stream_performance(df, metric, f"{metric.title()} by Stream Chunk"),
                use_container_width=True,
            )
    with tabs[4]:
        st.plotly_chart(drift_detection_timeline(df, drift_chunks), use_container_width=True)
        st.plotly_chart(
            retraining_events_timeline(int(df["chunk"].max()), drift_chunks),
            use_container_width=True,
        )
        st.plotly_chart(
            drift_density_graph(drift_chunks, int(df["chunk"].max())),
            use_container_width=True,
        )

    st.markdown("### Raw Stream Metrics")
    st.dataframe(
        df.style.format({
            "accuracy": "{:.2%}",
            "precision": "{:.2%}",
            "recall": "{:.2%}",
            "f1": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
