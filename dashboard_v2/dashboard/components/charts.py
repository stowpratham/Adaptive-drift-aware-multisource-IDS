from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PLOT_BG = "#0b1728"
GRID = "#1b2c43"
TEXT = "#b7c6d9"


def _base(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT),
        margin=dict(l=40, r=25, t=55, b=40),
        legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def stream_performance(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.line(df, x="chunk", y=metric, markers=True, title=title)
    fig.update_traces(line_width=2.5)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02], title=metric.title())
    fig.update_xaxes(title="Stream Chunk")
    return _base(fig)


def all_stream_metrics(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for metric, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]:
        fig.add_trace(go.Scatter(
            x=df["chunk"], y=df[metric], mode="lines+markers", name=label
        ))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02], title="Score")
    fig.update_xaxes(title="Stream Chunk")
    fig.update_layout(title="Model Performance Across the Stream")
    return _base(fig)


def drift_detection_timeline(df: pd.DataFrame, drift_chunks: List[int]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["chunk"], y=df["accuracy"], mode="lines+markers",
        name="Accuracy", line=dict(width=2.5)
    ))
    visible = sorted(set(c for c in drift_chunks if c in set(df["chunk"].tolist())))
    if visible:
        lookup = df.set_index("chunk")["accuracy"]
        fig.add_trace(go.Scatter(
            x=visible, y=[lookup[c] for c in visible],
            mode="markers", name="Drift detected",
            marker=dict(size=11, symbol="x")
        ))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.02], title="Accuracy")
    fig.update_xaxes(title="Stream Chunk")
    fig.update_layout(title="Performance + Drift Events")
    return _base(fig)


def prediction_distribution_chart(dist: Dict[str, int]) -> go.Figure:
    labels = list(dist.keys())
    values = list(dist.values())
    fig = px.pie(names=labels, values=values, title="Available Label Distribution")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _base(fig)


def retraining_events_timeline(total_chunks: int, drift_chunks: List[int]) -> go.Figure:
    visible = [c for c in drift_chunks if c <= total_chunks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=visible, y=[1] * len(visible), mode="markers",
        name="Drift / adaptation boundary",
        marker=dict(size=12, symbol="triangle-up")
    ))
    fig.update_xaxes(range=[0, max(total_chunks, 1)], title="Stream Chunk")
    fig.update_yaxes(visible=False)
    fig.update_layout(title="Drift Events Within Available Stream Metrics")
    return _base(fig)


def drift_density_graph(drift_chunks: List[int], total_chunks: int) -> go.Figure:
    if not drift_chunks:
        return _base(go.Figure().update_layout(title="No drift events recorded"))
    fig = px.histogram(x=drift_chunks, nbins=min(20, max(1, len(drift_chunks))), title="Reported Drift Density")
    fig.update_xaxes(title="Drift Chunk")
    fig.update_yaxes(title="Event Count")
    return _base(fig)


def model_metric_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    fig = px.bar(work, x="Model", y=metric, text=metric, title=f"{metric} Comparison")
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_yaxes(tickformat=".0%", range=[min(0, float(work[metric].min()) - .05), 1.05])
    fig.update_xaxes(title="")
    return _base(fig)
