from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.components.shap_viewer import render_sample_explanation, render_shap_panel
from dashboard.utils.data_loader import load_prediction_explanations, load_shap_assets, load_waterfall_files


def render() -> None:
    st.title("Explainable AI")
    shap_assets = load_shap_assets()
    render_shap_panel(shap_assets.get("summary"), shap_assets.get("importance"))

    explanation_json = load_prediction_explanations()
    if explanation_json is None:
        st.warning("No explanation JSON was found in results/explainability. Run ids_main.py first.")
        return

    explanations = explanation_json.get("explanations", [])
    if not explanations:
        st.info("No individual sample explanations were produced yet.")
        return

    sample_ids = [item.get("sample_id", f"sample_{i}") for i, item in enumerate(explanations)]
    selected = st.selectbox("Select a sample to review", sample_ids)
    sample_record = next((item for item in explanations if item.get("sample_id") == selected), explanations[0])

    waterfall_files = [path for path in load_waterfall_files() if path.endswith(".png")]
    force_files = [path for path in load_waterfall_files() if path.endswith(".html")]

    render_sample_explanation(sample_record, waterfall_files, force_files)

    if force_files:
        st.markdown("---")
        st.subheader("Interactive Force Plots")
        for file_path in force_files:
            st.markdown(f"- [Force plot]({Path(file_path).as_uri()})")
