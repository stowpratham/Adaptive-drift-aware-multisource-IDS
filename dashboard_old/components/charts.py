from __future__ import annotations

from typing import Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def accuracy_vs_chunk(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="chunk", y="accuracy", title="Accuracy vs Chunk", markers=True)
    fig.update_layout(yaxis_title="Accuracy", xaxis_title="Chunk", template="plotly_dark")
    return fig


def f1_vs_chunk(df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        df,
        x="chunk",
        y="f1",
        title="F1 Score vs Chunk",
        markers=True
    )

    fig.update_traces(line=dict(dash="dash"))

    fig.update_layout(
        yaxis_title="F1 Score",
        xaxis_title="Chunk",
        template="plotly_dark"
    )

    return fig

def drift_detection_timeline(df: pd.DataFrame, drift_chunks: List[int]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["chunk"], y=df["accuracy"], mode="lines+markers", name="Accuracy", line=dict(color="#22c55e")))
    fig.add_trace(
        go.Scatter(
            x=drift_chunks,
            y=[float(df.loc[df["chunk"] == chunk, "accuracy"].iloc[0]) for chunk in drift_chunks if chunk in df["chunk"].values],
            mode="markers",
            marker=dict(color="red", size=10, symbol="x"),
            name="Drift Chunk",
        )
    )
    fig.update_layout(title="Drift Detection Timeline", xaxis_title="Chunk", yaxis_title="Accuracy", template="plotly_dark")
    return fig


def prediction_distribution_chart(dist: Dict[str, int]) -> go.Figure:
    labels = list(dist.keys())
    values = list(dist.values())
    fig = px.pie(names=labels, values=values, title="Stream Label Distribution")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_dark")
    return fig


def retraining_events_timeline(total_chunks: int, drift_chunks: List[int]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drift_chunks,
            y=[1] * len(drift_chunks),
            mode="markers",
            marker=dict(color="red", size=12, symbol="triangle-up"),
            name="Retrain Event",
        )
    )
    fig.update_layout(
        title="Retraining Events Timeline",
        xaxis_title="Chunk",
        yaxis=dict(visible=False),
        xaxis=dict(range=[0, max(total_chunks, max(drift_chunks) if drift_chunks else 0) + 5]),
        template="plotly_dark",
    )
    return fig


def drift_density_graph(drift_chunks: List[int], total_chunks: int) -> go.Figure:
    fig = px.histogram(x=drift_chunks, nbins=min(20, len(drift_chunks) or 1), title="Drift Density Across Chunks")
    fig.update_layout(xaxis_title="Drift Chunk", yaxis_title="Count", template="plotly_dark")
    return fig
