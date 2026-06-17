from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st


def render_shap_panel(summary_path: Optional[str], importance_path: Optional[str]) -> None:
    st.subheader("Global Explainability")
    if summary_path:
        st.image(summary_path, caption="SHAP Summary", width="stretch")
    else:
        st.warning("SHAP summary plot not found.")

    if importance_path:
        st.image(importance_path, caption="Global Feature Importance", use_column_width=True)
    else:
        st.warning("Global feature importance plot not found.")


def render_sample_explanation(record: Dict[str, any], waterfall_files: List[str], force_files: List[str]) -> None:
    st.markdown("### Selected Sample Explanation")
    st.markdown(f"**Prediction:** {record.get('prediction_display', 'N/A')}")
    st.markdown(f"**Confidence:** {record.get('confidence', 'N/A'):.2f}")
    st.markdown(f"**Sample ID:** {record.get('sample_id', 'N/A')}")
    st.markdown(f"**Selection Reason:** {record.get('selection_reason', 'N/A')}")

    st.markdown("#### Top Features")
    top_features = record.get("top_features", [])
    if top_features:
        for feature in top_features:
            st.write(f"- {feature['display_name']} ({feature['feature']}) : {feature['shap_value']:+.4f} — {feature['impact']}")
    else:
        st.write("No top features available.")

    st.markdown("#### SHAP Feature Contributions")
    contributions = record.get("feature_contributions", {})
    if contributions:
        contributions_table = {
            "Feature": [],
            "SHAP Value": [],
        }
        for name, value in sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:10]:
            contributions_table["Feature"].append(name)
            contributions_table["SHAP Value"].append(value)
        st.table(contributions_table)
    else:
        st.write("No feature contributions available.")

    plot_files = record.get("plots", {})
    if plot_files:
        st.markdown("#### Associated Visualizations")
        for plot_type, path in plot_files.items():
            if Path(path).exists():
                if path.endswith(".png"):
                    st.image(path, caption=f"{plot_type.title()} Plot", use_column_width="stretch")
                elif path.endswith(".html"):
                    st.markdown(f"[{plot_type.title()} plot]({Path(path).as_uri()})")
    elif waterfall_files or force_files:
        st.markdown("#### Available XAI Artifacts")
        for path in waterfall_files[:3]:
            st.markdown(f"- [Waterfall plot]({Path(path).as_uri()})")
        for path in force_files[:3]:
            st.markdown(f"- [Force plot]({Path(path).as_uri()})")
    else:
        st.info("No local SHAP plot artifacts were found for this sample.")
