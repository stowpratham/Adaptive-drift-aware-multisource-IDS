from __future__ import annotations

import streamlit as st

from dashboard.components.charts import all_stream_metrics, model_metric_chart
from dashboard.utils.data_loader import (
    load_comparison_results,
    load_prediction_explanations,
    load_stream_metrics,
    parse_summary_metrics,
)
from dashboard.utils.helpers import format_percentage, system_health_status


def render():
    metrics = parse_summary_metrics()
    stream_df = load_stream_metrics()
    comparison = load_comparison_results()
    explanation_json = load_prediction_explanations()

    status, status_color = system_health_status(
        metrics.get("accuracy"),
        metrics.get("drift_events"),
    )

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">ADAPTIVE DRIFT-AWARE INTRUSION DETECTION</div>
            <div class="hero-title">Security Operations Overview</div>
            <p class="hero-text">
                Overall IDS performance and drift-aware streaming evaluation
                from the persisted experiment artifacts.
            </p>
            <div style="margin-top:12px">
                <span class="status-pill"
                      style="color:{status_color};
                             border-color:{status_color}55">
                    ● {status}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # OVERALL IDS PERFORMANCE
    # =========================================================
    st.markdown(
        "<div class='section-label'>OVERALL IDS PERFORMANCE</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        format_percentage(metrics.get("accuracy"))
        if metrics.get("accuracy") is not None else "N/A",
    )

    c2.metric(
        "Precision",
        format_percentage(metrics.get("precision"))
        if metrics.get("precision") is not None else "N/A",
    )

    c3.metric(
        "Recall",
        format_percentage(metrics.get("recall"))
        if metrics.get("recall") is not None else "N/A",
    )

    c4.metric(
        "F1 Score",
        format_percentage(metrics.get("f1"))
        if metrics.get("f1") is not None else "N/A",
    )

    c5.metric(
        "Kappa",
        format_percentage(metrics.get("kappa"))
        if metrics.get("kappa") is not None else "N/A",
    )

    # =========================================================
    # DRIFT-AWARE STREAM EVALUATION
    # =========================================================
    st.markdown(
        "<div class='section-label'>DRIFT-AWARE STREAM EVALUATION</div>",
        unsafe_allow_html=True,
    )

    if stream_df is not None and not stream_df.empty:

        total_chunks = len(stream_df)
        latest_chunk = int(stream_df["chunk"].max())

        drift_events = metrics.get("drift_events")
        retrain_count = metrics.get("retrain_count")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Persisted Stream Chunks",
            f"{total_chunks:,}",
        )

        s2.metric(
            "Detected Drift Events",
            f"{int(drift_events):,}"
            if drift_events is not None
            else "N/A",
        )

        s3.metric(
            "Retraining Events",
            f"{int(retrain_count):,}"
            if retrain_count is not None
            else "N/A",
        )

        s4.metric(
            "Latest Chunk",
            f"{latest_chunk:,}",
        )

        st.plotly_chart(
            all_stream_metrics(stream_df),
            use_container_width=True,
        )

        st.markdown(
            "<div class='section-label'>LATEST STREAM STATE</div>",
            unsafe_allow_html=True,
        )

        latest = stream_df.iloc[-1]

        chunk_id = int(latest["chunk"])
        total_samples = int(latest.get("total_samples", 0))
        normal_samples = int(latest.get("normal_samples", 0))
        attack_samples = int(latest.get("attack_samples", 0))

        accuracy = float(latest["accuracy"])
        precision = float(latest["precision"])
        recall = float(latest["recall"])
        f1 = float(latest["f1"])

        # Precision, Recall and F1 are attack-class metrics.
        # If the chunk contains no attack samples, these metrics
        # are undefined and should be displayed as N/A.
        attack_metrics_defined = attack_samples > 0

        precision_display = (
            format_percentage(precision)
            if attack_metrics_defined else "N/A"
        )

        recall_display = (
            format_percentage(recall)
            if attack_metrics_defined else "N/A"
        )

        f1_display = (
            format_percentage(f1)
            if attack_metrics_defined else "N/A"
        )

        # Stream composition
        st.markdown(
            "<div class='section-label'>STREAM COMPOSITION</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Chunk", f"{chunk_id:,}")
        c2.metric("Total Samples", f"{total_samples:,}")
        c3.metric("Normal Samples", f"{normal_samples:,}")
        c4.metric("Attack Samples", f"{attack_samples:,}")

        # Classification metrics
        st.markdown(
            "<div class='section-label'>LATEST CHUNK METRICS</div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Accuracy", format_percentage(accuracy))
        m2.metric("Precision", precision_display)
        m3.metric("Recall", recall_display)
        m4.metric("F1 Score", f1_display)

        if attack_samples == 0:
            st.info(
                "No attack samples are present in this stream chunk. "
                "Attack-specific Precision, Recall and F1 are therefore "
                "undefined for this chunk and are shown as N/A."
            )
        else:
            st.caption(
                "Attack-specific metrics are calculated from the attack "
                "samples present in this stream chunk."
            )

    else:
        st.warning("No persisted stream metrics were found.")
    # =========================================================
    # MODEL COMPARISON
    # =========================================================
    st.markdown(
        "<div class='section-label'>MODEL COMPARISON</div>",
        unsafe_allow_html=True,
    )

    if comparison is not None and not comparison.empty:

        columns = [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Kappa",
        ]

        columns = [c for c in columns if c in comparison.columns]

        st.dataframe(
            comparison[columns].style.format(
                {
                    c: "{:.2%}"
                    for c in columns
                    if c != "Model"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(
            model_metric_chart(comparison, "Accuracy"),
            use_container_width=True,
        )

    # =========================================================
    # EXPLAINABILITY
    # =========================================================
    if explanation_json:

        meta = explanation_json.get("metadata", {})

        st.markdown(
            "<div class='section-label'>EXPLAINABILITY STATUS</div>",
            unsafe_allow_html=True,
        )

        x1, x2, x3 = st.columns(3)

        stream_samples = meta.get("stream_samples")
        explained_samples = meta.get("explained_samples")
        drift_chunks = meta.get("drift_chunks") or []

        x1.metric(
            "XAI Stream Samples",
            f"{int(stream_samples):,}"
            if stream_samples is not None else "N/A",
        )

        x2.metric(
            "Explained Samples",
            f"{int(explained_samples):,}"
            if explained_samples is not None else "N/A",
        )

        x3.metric(
            "XAI Drift Chunks",
            f"{len(drift_chunks):,}",
        )


if __name__ == "__main__":
    render()
