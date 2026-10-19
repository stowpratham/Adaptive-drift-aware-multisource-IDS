from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.components.charts import model_metric_chart
from dashboard.utils.data_loader import load_comparison_results


# Training times from the latest successful IDS experiment run.
TRAINING_TIMES = {
    "Random Forest": 8.35,
    "Gradient Boosting": 19.03,
    "Decision Tree": 0.88,
    "KNN": 0.00,
    "SGD Hinge": 0.01,
    "SGD Log": 0.00,
    "Perceptron": 0.01,
    "Gaussian NB": 0.00,
}


def render():

    # =========================================================
    # HERO
    # =========================================================
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Experimental Comparison</div>
            <div class="hero-title">Model Comparison</div>
            <p class="hero-text">
                Comparison of persisted IDS evaluation results and
                training-time records from the eight-model ensemble.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # PERSISTED EVALUATION
    # =========================================================
    df = load_comparison_results()

    if df is None or df.empty:
        st.error(
            "results/baselines/final_comparison.csv "
            "was not found."
        )
        return

    metric_cols = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Kappa",
        "FAR",
        "DetectionRate",
    ]

    available = [
        c for c in metric_cols
        if c in df.columns
    ]

    display = df[["Model"] + available].copy()

    for column in available:
        display[column] = pd.to_numeric(
            display[column],
            errors="coerce",
        )

    st.markdown(
        "<div class='section-label'>PERSISTED EVALUATION RESULTS</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "These are evaluation metrics persisted by the IDS experiment. "
        "Only models present in final_comparison.csv are shown here."
    )

    formatted = display.style.format(
        {
            column: "{:.2%}"
            for column in available
        }
    )

    st.dataframe(
        formatted,
        use_container_width=True,
        hide_index=True,
    )

    # =========================================================
    # METRIC TABS
    # =========================================================
    st.markdown(
        "<div class='section-label'>METRIC-WISE COMPARISON</div>",
        unsafe_allow_html=True,
    )

    chart_metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Kappa",
        "FAR",
    ]

    tabs = st.tabs(chart_metrics)

    for tab, metric in zip(tabs, chart_metrics):
        with tab:
            if metric in display.columns:
                st.plotly_chart(
                    model_metric_chart(
                        display,
                        metric,
                    ),
                    use_container_width=True,
                )
            else:
                st.info(
                    f"{metric} is not available "
                    "in the persisted comparison results."
                )

    # =========================================================
    # EIGHT MODEL ENSEMBLE
    # =========================================================
    st.markdown(
        "<div class='section-label'>EIGHT-MODEL ENSEMBLE</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "The IDS ensemble contains eight models: "
        "Random Forest, Gradient Boosting, Decision Tree, KNN, "
        "SGD Hinge, SGD Log, Perceptron and Gaussian NB."
    )

    timing_df = pd.DataFrame(
        [
            {
                "Model": model,
                "Training Time (s)": seconds,
            }
            for model, seconds in TRAINING_TIMES.items()
        ]
    )

    timing_df = timing_df.sort_values(
        "Training Time (s)",
        ascending=False,
    )

    # =========================================================
    # TRAINING TIME CARDS
    # =========================================================
    st.markdown(
        "<div class='section-label'>TRAINING TIME</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    cards = timing_df.head(4).to_dict("records")

    for column, record in zip(
        [c1, c2, c3, c4],
        cards,
    ):
        with column:
            st.metric(
                record["Model"],
                f"{record['Training Time (s)']:.2f}s",
            )

    # =========================================================
    # TRAINING TIME CHART
    # =========================================================
    st.markdown(
        "<div class='section-label'>ALL EIGHT MODELS</div>",
        unsafe_allow_html=True,
    )

    chart_df = timing_df.sort_values(
        "Training Time (s)",
        ascending=True,
    )

    st.bar_chart(
        chart_df.set_index("Model")[
            "Training Time (s)"
        ]
    )

    st.dataframe(
        timing_df.style.format(
            {
                "Training Time (s)": "{:.2f}s"
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # =========================================================
    # IMPORTANT METHODOLOGY NOTE
    # =========================================================
    st.markdown(
        "<div class='section-label'>INTERPRETATION</div>",
        unsafe_allow_html=True,
    )

    st.info(
        "The persisted evaluation CSV currently contains "
        "evaluation metrics for Proposed IDS, Random Forest and "
        "XGBoost. Training times are available for all eight "
        "ensemble models. Evaluation scores for the remaining "
        "ensemble models are not displayed because they were not "
        "persisted by the current experiment."
    )


if __name__ == "__main__":
    render()