from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from dashboard.utils.inference import (
    detect_dataset_format,
    get_upload_summary,
    validate_uploaded_csv,
)
from dashboard.utils.model_loader import has_saved_models, get_saved_models


def render() -> None:
    st.title("CSV Log Upload")
    st.markdown(
        "Upload a .csv log file from NSL-KDD or UNSW-NB15 and validate it against the existing IDS schema."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Supported formats: network_log.csv, flow_logs.csv, traffic_logs.csv, unsw_logs.csv, kdd_logs.csv")
        return

    try:
        data = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="replace")))
    except Exception as exc:
        st.error(f"Unable to parse CSV file: {exc}")
        return

    dataset_type = detect_dataset_format(data)
    validation = validate_uploaded_csv(data)
    summary = get_upload_summary(data, dataset_type)

    st.markdown("### Upload Summary")
    st.write(summary)

    if validation["errors"]:
        st.error("Validation failed")
        for error in validation["errors"]:
            st.write(f"- {error}")
        return

    st.success(f"Detected dataset type: {dataset_type or 'Unknown'}")
    st.write("**Columns validated successfully.**")
    st.markdown("---")
    st.subheader("Raw upload preview")
    st.dataframe(data.head(10))

    if not has_saved_models():
        st.warning(
            "Model artifacts are not available in the repository. "
            "CSV inference will be enabled once trained model files are saved to the `models/` directory."
        )
        return

    models = get_saved_models()
    if not models:
        st.warning(
            "Saved model artifacts were discovered, but they could not be loaded. "
            "Ensure the model files are valid joblib or PyTorch artifacts."
        )
        return

    st.info("Saved model artifacts were detected. Prediction support will be enabled when inference is configured.")
    st.markdown(
        "At present, this dashboard loads existing trained model artifacts only. "
        "If inference is implemented with serialized models, predictions will appear here."
    )
