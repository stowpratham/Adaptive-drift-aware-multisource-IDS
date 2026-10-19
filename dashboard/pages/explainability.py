from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.utils.data_loader import (
    load_latent_mapping,
    load_prediction_explanations,
    load_shap_assets,
    load_waterfall_files,
)


def _contribution_chart(record):
    rows = []
    for name, value in sorted(
        (record.get("feature_contributions") or {}).items(),
        key=lambda x: abs(float(x[1])),
        reverse=True,
    )[:12]:
        rows.append({"Feature": name, "SHAP": float(value)})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("SHAP")
    return df.set_index("Feature")


def render() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Explainable AI</div>
            <div class="hero-title">Why did the IDS make this decision?</div>
            <p class="hero-text">SHAP explanations are projected from latent features back to the original feature space through the saved latent mapping.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    assets = load_shap_assets()
    explanation_json = load_prediction_explanations()

    if assets.get("summary") or assets.get("importance"):
        c1, c2 = st.columns(2)
        with c1:
            if assets.get("summary"):
                st.image(assets["summary"], caption="Global SHAP Summary", use_container_width=True)
        with c2:
            if assets.get("importance"):
                st.image(assets["importance"], caption="Global Feature Importance", use_container_width=True)

    if not explanation_json:
        st.warning("No prediction_explanations.json was found.")
        return

    explanations = explanation_json.get("explanations", [])
    if not explanations:
        st.info("No individual explanations are available.")
        return

    labels = [x.get("sample_id", f"sample_{i}") for i, x in enumerate(explanations)]
    selected = st.selectbox("Select an explained stream sample", labels)
    record = next(x for x in explanations if x.get("sample_id") == selected)

    prediction = str(record.get("prediction_display", record.get("prediction", "Unknown")))
    confidence = float(record.get("confidence", 0.0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction", prediction)
    c2.metric("Confidence", f"{confidence:.1%}")
    c3.metric("Chunk", record.get("chunk_index", "N/A"))
    c4.metric("Sample Index", record.get("chunk_local_index", "N/A"))

    st.markdown("### Decision Evidence")
    top = record.get("top_features", [])
    if top:
        evidence = []
        for item in top:
            evidence.append({
                "Latent Feature": item.get("display_name", item.get("feature")),
                "SHAP": float(item.get("shap_value", 0)),
                "Effect": item.get("impact", ""),
                "Original Features": ", ".join(
                    f"{x.get('feature')} ({x.get('importance', 0):.3f})"
                    for x in item.get("original_features", [])
                ),
            })
        st.dataframe(
            pd.DataFrame(evidence).style.format({"SHAP": "{:+.5f}"}),
            use_container_width=True,
            hide_index=True,
        )

    chart_df = _contribution_chart(record)
    if chart_df is not None:
        st.markdown("### Latent SHAP Contributions")
        st.bar_chart(chart_df)

    projected = record.get("projected_original_features", [])
    if projected:
        st.markdown("### Projected Original Features")
        proj = pd.DataFrame(projected)
        st.dataframe(
            proj.style.format({"normalized_importance": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("How to explain this in the viva"):
        st.write(
            "The IDS first represents the input in latent space. SHAP then estimates how individual latent features "
            "support or oppose the predicted class. The saved latent-feature mapping links those latent dimensions "
            "back to original input features, allowing the dashboard to expose concrete feature evidence."
        )

    waterfall_files = load_waterfall_files()
    matching = [
        p for p in waterfall_files
        if str(record.get("sample_id", "")) in Path(p).name
    ]
    if not matching:
        matching = waterfall_files[:1]
    if matching:
        st.markdown("### SHAP Waterfall Artifact")
        st.image(matching[0], caption=Path(matching[0]).name, use_container_width=True)

    mapping = load_latent_mapping()
    if mapping:
        with st.expander("Latent Feature Mapping"):
            st.dataframe(pd.DataFrame(mapping), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
