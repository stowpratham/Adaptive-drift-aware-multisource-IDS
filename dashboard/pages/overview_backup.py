from __future__ import annotations

import streamlit as st

from dashboard.components.charts import all_stream_metrics, model_metric_chart
from dashboard.components.metric_cards import render_metric_cards
from dashboard.utils.data_loader import (
    get_drift_chunks,
    load_comparison_results,
    load_prediction_explanations,
    load_stream_metrics,
    parse_phase_distribution,
    parse_summary_metrics,
)
from dashboard.utils.helpers import format_percentage, system_health_status


def render():
    metrics = parse_summary_metrics()
    stream_df = load_stream_metrics()
    comparison = load_comparison_results()
    phases = parse_phase_distribution()

    status, status_color = system_health_status(metrics.get("accuracy"), metrics.get("drift_events"))

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">Adaptive Drift-Aware Intrusion Detection</div>
            <div class="hero-title">Security Operations Overview</div>
            <p class="hero-text">Live view of persisted IDS evaluation, stream performance, drift events and explainability artifacts.</p>
            <div style="margin-top:12px"><span class="status-pill" style="color:{status_color};border-color:{status_color}55">● {status}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_metric_cards(
        metrics,
        status,
        status_color,
        phases.get("attack_samples"),
        phases.get("benign_samples"),
        metrics.get("stream_samples"),
    )

    if stream_df is not None:
        st.markdown("<div class='section-label'>STREAM PERFORMANCE</div>", unsafe_allow_html=True)
        st.plotly_chart(all_stream_metrics(stream_df), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.metric("Latest chunk accuracy", format_percentage(float(stream_df.iloc[-1]["accuracy"])))
        with right:
            st.metric("Latest chunk F1", format_percentage(float(stream_df.iloc[-1]["f1"])))

    st.markdown("<div class='section-label'>MODEL SNAPSHOT</div>", unsafe_allow_html=True)
    if comparison is not None and not comparison.empty:
        st.dataframe(
            comparison[["Model", "Accuracy", "Precision", "Recall", "F1", "Kappa"]]
            .style.format({c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "F1", "Kappa"]}),
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(model_metric_chart(comparison, "Accuracy"), use_container_width=True)

    explanation_json = load_prediction_explanations()
    if explanation_json:
        meta = explanation_json.get("metadata", {})
        st.markdown("<div class='section-label'>EXPLAINABILITY STATUS</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Stream samples", f"{int(meta.get('stream_samples', 0)):,}")
        c2.metric("Explained samples", f"{int(meta.get('explained_samples', 0)):,}")
        c3.metric("Reported drift events", f"{len(meta.get('drift_chunks', [])):,}")


if __name__ == "__main__":
    render()
