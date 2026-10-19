from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st


def app_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(37,99,235,.10), transparent 28%),
                radial-gradient(circle at 90% 15%, rgba(14,165,233,.07), transparent 25%),
                #07101f;
            color: #e5edf7;
        }
        [data-testid="stSidebar"] {
            background: #050b16;
            border-right: 1px solid #172338;
        }
        [data-testid="stSidebar"] .stRadio label {
            color: #b9c7d9;
        }
        h1, h2, h3, h4 {
            color: #f5f9ff !important;
        }
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }
        .brand {
            display:flex;
            align-items:center;
            gap:12px;
            padding:8px 2px 4px;
        }
        .brand-icon {
            width:42px;
            height:42px;
            display:flex;
            align-items:center;
            justify-content:center;
            border-radius:12px;
            background:linear-gradient(135deg,#2563eb,#0ea5e9);
            font-size:22px;
        }
        .brand-title { font-weight:800; letter-spacing:.08em; color:#f8fbff; }
        .brand-subtitle { font-size:12px; color:#7f91aa; margin-top:2px; }
        .hero {
            padding: 22px 24px;
            border:1px solid #1d3049;
            border-radius:18px;
            background:linear-gradient(135deg,rgba(15,31,54,.96),rgba(8,18,33,.92));
            box-shadow:0 12px 35px rgba(0,0,0,.18);
            margin-bottom:18px;
        }
        .hero-kicker {
            color:#38bdf8;
            font-size:12px;
            font-weight:800;
            letter-spacing:.12em;
            text-transform:uppercase;
        }
        .hero-title { font-size:30px; font-weight:800; margin:5px 0; color:#f8fbff; }
        .hero-text { color:#91a4bb; margin:0; }
        .status-pill {
            display:inline-block;
            padding:6px 11px;
            border-radius:999px;
            font-size:12px;
            font-weight:700;
            background:#0c2434;
            color:#67e8f9;
            border:1px solid #164e63;
        }
        .section-label {
            color:#7f91aa;
            font-size:12px;
            font-weight:800;
            letter-spacing:.10em;
            text-transform:uppercase;
            margin:12px 0 8px;
        }
        div[data-testid="stMetric"] {
            background:linear-gradient(180deg,#0d1a2c,#0a1525);
            border:1px solid #1b2c43;
            padding:15px 16px;
            border-radius:14px;
            min-height:108px;
        }
        div[data-testid="stMetricLabel"] { color:#8295ad !important; }
        div[data-testid="stMetricValue"] { color:#f4f8ff !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { color:#91a4bb; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_percentage(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if abs(value) <= 1.5:
        return f"{value * 100:.2f}%"
    return f"{value:.2f}%"


def format_int(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    return f"{int(value):,}"


def system_health_status(accuracy: Optional[float], drift_events: Optional[int]) -> Tuple[str, str]:
    if accuracy is None:
        return "UNKNOWN", "#94a3b8"
    acc = accuracy if accuracy <= 1.5 else accuracy / 100
    if acc >= 0.85 and (drift_events is None or drift_events <= 50):
        return "STABLE", "#22c55e"
    if acc >= 0.75:
        return "ADAPTIVE / WARNING", "#f59e0b"
    return "DEGRADED", "#ef4444"


def build_status_badge(status: str, color: str) -> str:
    return f"<span class='status-pill' style='color:{color};border-color:{color}55'>{status}</span>"
