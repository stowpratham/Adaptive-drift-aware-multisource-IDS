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
        rows.append(
            {
                "Feature": name,
                "SHAP": float(value),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("SHAP")

    return df.set_index("Feature")


def _prediction_reasoning(record):
    """
    Build a simple human-readable explanation from the saved
    SHAP contribution values.
    """

    contributions = record.get("feature_contributions") or {}

    if not contributions:
        return None

    ranked = sorted(
        contributions.items(),
        key=lambda x: abs(float(x[1])),
        reverse=True,
    )

    supporting = []
    opposing = []

    for feature, value in ranked[:8]:

        value = float(value)

        if value > 0:
            supporting.append(
                {
                    "Feature": feature,
                    "SHAP Contribution": value,
                    "Direction": "Supports predicted class",
                }
            )

        elif value < 0:
            opposing.append(
                {
                    "Feature": feature,
                    "SHAP Contribution": value,
                    "Direction": "Opposes predicted class",
                }
            )

    return supporting, opposing


def render() -> None:

    # ---------------------------------------------------------
    # HERO
    # ---------------------------------------------------------

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Explainable AI</div>
            <div class="hero-title">
                Why did the IDS make this decision?
            </div>
            <p class="hero-text">
                SHAP explains the contribution of latent features to each
                prediction, while the saved latent mapping connects those
                latent dimensions back to the original network features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # LOAD XAI ARTIFACTS
    # ---------------------------------------------------------

    assets = load_shap_assets()
    explanation_json = load_prediction_explanations()

    # ---------------------------------------------------------
    # GLOBAL XAI
    # ---------------------------------------------------------

    if assets.get("summary") or assets.get("importance"):

        st.markdown("### Global Explainability")

        c1, c2 = st.columns(2)

        with c1:
            if assets.get("summary"):
                st.image(
                    assets["summary"],
                    caption="Global SHAP Summary",
                    use_container_width=True,
                )

        with c2:
            if assets.get("importance"):
                st.image(
                    assets["importance"],
                    caption="Global Feature Importance",
                    use_container_width=True,
                )

    # ---------------------------------------------------------
    # CHECK INDIVIDUAL EXPLANATIONS
    # ---------------------------------------------------------

    if not explanation_json:
        st.warning(
            "No prediction_explanations.json was found."
        )
        return

    explanations = explanation_json.get(
        "explanations",
        [],
    )

    if not explanations:
        st.info(
            "No individual explanations are available."
        )
        return

    # ---------------------------------------------------------
    # SAMPLE SELECTION
    # ---------------------------------------------------------

    st.markdown("### Individual Prediction Explanation")

    labels = [
        x.get(
            "sample_id",
            f"sample_{i}",
        )
        for i, x in enumerate(explanations)
    ]

    selected = st.selectbox(
        "Select an explained stream sample",
        labels,
    )

    record = next(
        x
        for x in explanations
        if x.get("sample_id") == selected
    )

    # ---------------------------------------------------------
    # BASIC PREDICTION INFORMATION
    # ---------------------------------------------------------

    prediction = str(
        record.get(
            "prediction_display",
            record.get(
                "prediction",
                "Unknown",
            ),
        )
    )

    confidence = float(
        record.get(
            "confidence",
            0.0,
        )
    )

    chunk = record.get(
        "chunk_index",
        "N/A",
    )

    sample_index = record.get(
        "chunk_local_index",
        "N/A",
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Prediction",
        prediction,
    )

    c2.metric(
        "Confidence",
        f"{confidence:.1%}",
    )

    c3.metric(
        "Chunk",
        chunk,
    )

    c4.metric(
        "Sample Index",
        sample_index,
    )

    # ---------------------------------------------------------
    # MAIN ANSWER FOR THE PANEL
    # ---------------------------------------------------------

    reasoning = _prediction_reasoning(record)

    if reasoning:

        supporting, opposing = reasoning

        st.markdown("### Why was this prediction made?")

        st.info(
            f"The IDS classified this sample as **{prediction}** "
            f"with **{confidence:.1%} confidence**. "
            "The SHAP values below show which latent features "
            "supported or opposed the predicted class."
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "#### Supporting Evidence"
            )

            if supporting:

                support_df = pd.DataFrame(
                    supporting
                )

                st.dataframe(
                    support_df.style.format(
                        {
                            "SHAP Contribution": "{:+.5f}"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.write(
                    "No positive SHAP contributions were recorded."
                )

        with col2:

            st.markdown(
                "#### Opposing Evidence"
            )

            if opposing:

                oppose_df = pd.DataFrame(
                    opposing
                )

                st.dataframe(
                    oppose_df.style.format(
                        {
                            "SHAP Contribution": "{:+.5f}"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.write(
                    "No negative SHAP contributions were recorded."
                )

    # ---------------------------------------------------------
    # DECISION EVIDENCE
    # ---------------------------------------------------------

    st.markdown("### Decision Evidence")

    top = record.get(
        "top_features",
        [],
    )

    if top:

        evidence = []

        for item in top:

            evidence.append(
                {
                    "Latent Feature": item.get(
                        "display_name",
                        item.get(
                            "feature"
                        ),
                    ),
                    "SHAP": float(
                        item.get(
                            "shap_value",
                            0,
                        )
                    ),
                    "Effect": item.get(
                        "impact",
                        "",
                    ),
                    "Original Features": ", ".join(
                        f"{x.get('feature')} "
                        f"({x.get('importance', 0):.3f})"
                        for x in item.get(
                            "original_features",
                            [],
                        )
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                evidence
            ).style.format(
                {
                    "SHAP": "{:+.5f}"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------
    # LATENT SHAP CONTRIBUTIONS
    # ---------------------------------------------------------

    chart_df = _contribution_chart(record)

    if chart_df is not None:

        st.markdown(
            "### Latent SHAP Contributions"
        )

        st.bar_chart(chart_df)

    # ---------------------------------------------------------
    # PROJECTED ORIGINAL FEATURES
    # ---------------------------------------------------------

    projected = record.get(
        "projected_original_features",
        [],
    )

    if projected:

        st.markdown(
            "### Projected Original Features"
        )

        proj = pd.DataFrame(
            projected
        )

        if "normalized_importance" in proj.columns:

            st.dataframe(
                proj.style.format(
                    {
                        "normalized_importance": "{:.2f}"
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.dataframe(
                proj,
                use_container_width=True,
                hide_index=True,
            )

    # ---------------------------------------------------------
    # VIVA EXPLANATION
    # ---------------------------------------------------------

    with st.expander(
        "How to explain this in the viva"
    ):

        st.write(
            """
            The IDS first transforms the network traffic features into
            a lower-dimensional latent representation.

            SHAP is then applied to the trained Random Forest model to
            determine how much each latent feature contributes to the
            prediction.

            Positive SHAP contribution means that the feature supports
            the predicted class, while negative SHAP contribution means
            that it pushes the prediction in the opposite direction.

            Finally, the saved latent-feature mapping connects the
            important latent dimensions back to the original network
            features. This allows the dashboard to show concrete
            evidence behind the IDS decision.
            """
        )

    # ---------------------------------------------------------
    # WATERFALL
    # ---------------------------------------------------------

    waterfall_files = load_waterfall_files()

    matching = [
        p
        for p in waterfall_files
        if str(
            record.get(
                "sample_id",
                "",
            )
        ) in Path(p).name
    ]

    if not matching:

        matching = waterfall_files[:1]

    if matching:

        st.markdown(
            "### SHAP Waterfall Explanation"
        )

        st.image(
            matching[0],
            caption=Path(
                matching[0]
            ).name,
            use_container_width=True,
        )

    # ---------------------------------------------------------
    # LATENT FEATURE MAPPING
    # ---------------------------------------------------------

    mapping = load_latent_mapping()

    if mapping:

        with st.expander(
            "Latent Feature Mapping"
        ):

            st.dataframe(
                pd.DataFrame(mapping),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    render()