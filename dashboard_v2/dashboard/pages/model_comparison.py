from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.components.charts import model_metric_chart
from dashboard.utils.data_loader import load_comparison_results

# Training times recorded from the IDS experiment output supplied by the project.
# They are displayed separately because the current persisted comparison CSV does
# not contain evaluation metrics for all eight streaming/batch models.
TRAINING_TIMES = {
    "RandomForest": 13.71,
    "GradientBoosting": 41.51,
    "DecisionTree": 1.47,
    "KNN": 0.00,
    "SGD_Hinge": 0.01,
    "SGD_Log": 0.01,
    "Perceptron": 0.00,
    "GaussianNB": 0.01,
}


def render():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Experimental Comparison</div>
            <div class="hero-title">Model Comparison</div>
            <p class="hero-text">Persisted evaluation metrics are shown for models currently saved by the experiment. Training-time records for all eight auxiliary models are shown separately.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_comparison_results()
    if df is None or df.empty:
        st.error("results/baselines/final_comparison.csv was not found.")
        return

    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "Kappa", "FAR", "DetectionRate"]
    available = [c for c in metric_cols if c in df.columns]

    display = df[["Model"] + available].copy()
    for c in available:
        display[c] = pd.to_numeric(display[c], errors="coerce")

    st.markdown("### Persisted Evaluation Results")
    st.dataframe(
        display.style.format({c: "{:.2%}" for c in available}),
        use_container_width=True,
        hide_index=True,
    )

    tabs = st.tabs(["Accuracy", "Precision", "Recall", "F1", "Kappa", "FAR"])
    for tab, metric in zip(tabs, ["Accuracy", "Precision", "Recall", "F1", "Kappa", "FAR"]):
        with tab:
            if metric in display.columns:
                st.plotly_chart(model_metric_chart(display, metric), use_container_width=True)

    st.markdown("### Eight-Model Training Time")
    timing_df = pd.DataFrame(
        [{"Model": name, "Training Time (s)": seconds} for name, seconds in TRAINING_TIMES.items()]
    ).sort_values("Training Time (s)", ascending=False)
    st.bar_chart(timing_df.set_index("Model"))
    st.dataframe(timing_df, use_container_width=True, hide_index=True)

    st.info(
        "The current final_comparison.csv persists evaluation metrics for Proposed IDS, Random Forest and XGBoost. "
        "The eight-model section therefore reports the recorded training times, not fabricated evaluation scores. "
        "If you want Accuracy/Precision/Recall/F1/Kappa for all eight models, the IDS pipeline should be updated to persist those metrics during evaluation."
    )
