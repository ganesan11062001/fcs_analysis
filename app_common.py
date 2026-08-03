"""
app_common.py — shared helpers used by app.py and pages/*.py.

Kept at the project root (not inside core/) since this is UI-specific glue code
(Streamlit caching, plotting conventions) that core/ should stay free of.
"""

import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.io import load_trace_auto
from core.engine import _coarsen, select_window

DEFAULT_SETTINGS = {
    "segments": 5,
    "points_per_segment": 15,
    "base": 4,
    "kappa": 5.0,
    "min_reliable_n_samples": 10,
}


def init_defaults():
    if "defaults" not in st.session_state:
        st.session_state["defaults"] = dict(DEFAULT_SETTINGS)


@st.cache_data(show_spinner="Parsing trace file...")
def load_trace_cached(file_bytes, filename):
    suffix = os.path.splitext(filename)[1] or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return load_trace_auto(tmp_path)
    finally:
        os.unlink(tmp_path)


def decimate_for_display(time_arr, y_arr, max_points=5000):
    """Down-sample purely for plotting a raw multi-million-row trace in the browser.
    This is a DISPLAY-ONLY decimation -- it never touches the arrays used for the
    actual 'Binning points' analysis control or the correlation engine."""
    n = len(time_arr)
    if n <= max_points:
        return time_arr, y_arr
    factor = int(np.ceil(n / max_points))
    return _coarsen(time_arr, factor), _coarsen(y_arr, factor)


def window_channels(time_arr, channels, t0, t1):
    """Apply the same [t0, t1] time-window trim to every channel array, reusing
    engine.select_window (validated) once per channel."""
    windowed_time = None
    windowed_channels = {}
    for name, arr in channels.items():
        t_w, arr_w = select_window(time_arr, arr, t0, t1)
        windowed_time = t_w
        windowed_channels[name] = arr_w
    return windowed_time, windowed_channels


def time_window_selector(time_arr, key_prefix):
    """A double-ended range slider for trimming the acquisition window, with the
    kept region shaded on a companion plot via add_vrect. Returns (t0, t1)."""
    t_min, t_max = float(time_arr[0]), float(time_arr[-1])
    t0, t1 = st.slider(
        "Time window to analyze (seconds)",
        min_value=t_min,
        max_value=t_max,
        value=(t_min, t_max),
        key=f"{key_prefix}_window_slider",
    )
    return t0, t1


def plot_raw_trace_with_window(time_arr, channels, t0, t1, max_points=5000):
    disp_time, _ = decimate_for_display(time_arr, next(iter(channels.values())), max_points)
    fig = go.Figure()
    for name, arr in channels.items():
        _, disp_y = decimate_for_display(time_arr, arr, max_points)
        fig.add_trace(go.Scatter(x=disp_time, y=disp_y, mode="lines", name=name, line=dict(width=1)))
    fig.add_vrect(x0=t0, x1=t1, fillcolor="LightGreen", opacity=0.2, line_width=0, annotation_text="kept window")
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Photon counts / bin",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h"),
    )
    return fig


def plot_correlation_curve(results, min_reliable_n_samples, title=""):
    """Semilog-x G(tau) plot (tau log-scale, G linear -- not log-log, since G can
    go negative at long lags from noise). Points below the reliability threshold
    are shown with a distinct marker so long-tau statistical unreliability is
    visually obvious, not hidden."""
    fig = go.Figure()
    colors = {"acf_ch1": "#d62728", "acf_ch2": "#2ca02c", "cross": "#1f77b4"}
    labels = {"acf_ch1": "CH1 autocorrelation", "acf_ch2": "CH2 autocorrelation", "cross": "Cross-correlation"}
    any_unreliable = False
    for kind, result in results.items():
        color = colors.get(kind, "gray")
        reliable = result.n_samples >= min_reliable_n_samples
        if not np.all(reliable):
            any_unreliable = True
        fig.add_trace(
            go.Scatter(
                x=result.tau[reliable],
                y=result.g[reliable],
                mode="markers",
                name=labels.get(kind, kind),
                marker=dict(color=color, size=5),
            )
        )
        if np.any(~reliable):
            fig.add_trace(
                go.Scatter(
                    x=result.tau[~reliable],
                    y=result.g[~reliable],
                    mode="markers",
                    name=f"{labels.get(kind, kind)} (low sample count)",
                    marker=dict(color=color, size=7, symbol="x", opacity=0.5),
                )
            )
    fig.update_xaxes(type="log", title="tau (s)")
    fig.update_yaxes(title="G(tau)")
    fig.update_layout(title=title, height=400, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig, any_unreliable


def plot_fit_overlay(tau, g, fit_curve_arr, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tau, y=g, mode="markers", name="data", marker=dict(size=5, color="gray")))
    order = np.argsort(tau)
    fig.add_trace(go.Scatter(x=tau[order], y=fit_curve_arr[order], mode="lines", name="fit", line=dict(color="red")))
    fig.update_xaxes(type="log", title="tau (s)")
    fig.update_yaxes(title="G(tau)")
    fig.update_layout(title=title, height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fit_settings_widgets(key_prefix, label):
    st.markdown(f"**{label} fit settings**")
    n_components = st.radio("Components", [1, 2], key=f"{key_prefix}_ncomp", horizontal=True)
    triplet = st.checkbox("Include triplet/blinking term", key=f"{key_prefix}_triplet")
    n_starts = st.number_input("Multi-start attempts", min_value=1, max_value=20, value=5, key=f"{key_prefix}_nstarts")
    return n_components, triplet, n_starts
