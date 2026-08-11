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

# Validated categorical palette (CVD-safe, fixed order -- see dataviz skill).
# Assigned by entity, not cycled: CH1 always slot 1, CH2 always slot 2, cross
# always slot 3, everywhere in the app (charts, comparison plots, metrics).
SERIES_COLORS = {
    "acf_ch1": "#2a78d6",  # slot 1 -- blue
    "acf_ch2": "#eb6834",  # slot 2 -- orange
    "cross": "#1baf7a",  # slot 3 -- aqua
}
SERIES_LABELS = {
    "acf_ch1": "CH1 autocorrelation",
    "acf_ch2": "CH2 autocorrelation",
    "cross": "Cross-correlation",
}
MUTED_INK = "#898781"  # axis/label text token -- never used for a data mark's identity
GRIDLINE = "#e1e0d9"  # hairline gridline color
REFERENCE_LINE = "#0b0b0b"  # neutral (non-categorical) color for "true value" annotation lines


def init_defaults():
    if "defaults" not in st.session_state:
        st.session_state["defaults"] = dict(DEFAULT_SETTINGS)


def sidebar_about():
    """Small, consistent wayfinding blurb shown in the sidebar on every page --
    matters once the app is a public URL that people may land on directly via
    a shared link to a sub-page, with no context from the Home page."""
    with st.sidebar:
        st.caption(
            "**General-purpose FCS/FCCS data analysis**\n\n"
            "Analyzes ISS VistaVision trace exports (CSV). See the Home page "
            "for an overview of the pipeline."
        )


def format_seconds(x):
    """Human-scale formatting for a time constant (tau_D, dt, etc.): picks
    us/ms/s so a labmate doesn't have to mentally parse 2.1e-04."""
    if x is None or not np.isfinite(x):
        return "n/a"
    ax = abs(x)
    if ax < 1e-3:
        return f"{x * 1e6:.3g} µs"
    if ax < 1.0:
        return f"{x * 1e3:.3g} ms"
    return f"{x:.3g} s"


def apply_chart_style(fig):
    """Hairline, recessive gridlines -- applied to every chart for a consistent look."""
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=GRIDLINE)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, linecolor=GRIDLINE)
    return fig


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
        help=(
            "Keeps only the intensity data between t0 and t1; everything outside is dropped "
            "before binning/correlation. Impact: use this to cut out a bad segment (laser "
            "drift, a bubble, focus loss) -- including bad data biases G(tau) at every lag, "
            "since the correlator's normalization uses the whole kept trace's mean intensity."
        ),
    )
    return t0, t1


def plot_raw_trace_with_window(time_arr, channels, t0, t1, max_points=5000):
    disp_time, _ = decimate_for_display(time_arr, next(iter(channels.values())), max_points)
    fig = go.Figure()
    channel_colors = {"CH1": SERIES_COLORS["acf_ch1"], "CH2": SERIES_COLORS["acf_ch2"]}
    for name, arr in channels.items():
        _, disp_y = decimate_for_display(time_arr, arr, max_points)
        fig.add_trace(
            go.Scatter(
                x=disp_time, y=disp_y, mode="lines", name=name,
                line=dict(width=2, color=channel_colors.get(name)),
            )
        )
    fig.add_vrect(x0=t0, x1=t1, fillcolor="#2a78d6", opacity=0.08, line_width=0, annotation_text="kept window")
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Photon counts / bin",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h") if len(channels) > 1 else dict(),
        showlegend=len(channels) > 1,
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
    )
    return apply_chart_style(fig)


def plot_correlation_curve(results, min_reliable_n_samples, title=""):
    """Semilog-x G(tau) plot (tau log-scale, G linear -- not log-log, since G can
    go negative at long lags from noise). Points below the reliability threshold
    are shown with a distinct marker so long-tau statistical unreliability is
    visually obvious, not hidden."""
    fig = go.Figure()
    any_unreliable = False
    for kind, result in results.items():
        color = SERIES_COLORS.get(kind, MUTED_INK)
        reliable = result.n_samples >= min_reliable_n_samples
        if not np.all(reliable):
            any_unreliable = True
        fig.add_trace(
            go.Scatter(
                x=result.tau[reliable],
                y=result.g[reliable],
                mode="markers",
                name=SERIES_LABELS.get(kind, kind),
                marker=dict(color=color, size=8, line=dict(width=1, color="#fcfcfb")),
            )
        )
        if np.any(~reliable):
            fig.add_trace(
                go.Scatter(
                    x=result.tau[~reliable],
                    y=result.g[~reliable],
                    mode="markers",
                    name=f"{SERIES_LABELS.get(kind, kind)} (low sample count)",
                    marker=dict(color=color, size=9, symbol="x", opacity=0.55),
                )
            )
    fig.update_xaxes(type="log", title="tau (s)")
    fig.update_yaxes(title="G(tau)")
    fig.update_layout(
        title=title, height=400, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"), plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    return apply_chart_style(fig), any_unreliable


def plot_fit_overlay(tau, g, fit_curve_arr, title="", color=None):
    color = color or SERIES_COLORS["acf_ch1"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=tau, y=g, mode="markers", name="data", marker=dict(size=8, color=MUTED_INK, opacity=0.7))
    )
    order = np.argsort(tau)
    fig.add_trace(
        go.Scatter(x=tau[order], y=fit_curve_arr[order], mode="lines", name="fit", line=dict(color=color, width=2))
    )
    fig.update_xaxes(type="log", title="tau (s)")
    fig.update_yaxes(title="G(tau)")
    fig.update_layout(
        title=title, height=350, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"), plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
    )
    return apply_chart_style(fig)


def fit_settings_widgets(key_prefix, label):
    st.markdown(f"**{label} fit settings**")
    n_components = st.radio(
        "Components", [1, 2], key=f"{key_prefix}_ncomp", horizontal=True,
        help=(
            "How many diffusing species the model assumes.\n\n"
            r"Formula: $G(\tau) = \frac{1}{N}\sum_i f_i D_i(\tau)$, $\sum_i f_i = 1$, "
            r"$D_i(\tau) = \frac{1}{1+\tau/\tau_{Di}} \cdot \frac{1}{\sqrt{1+\tau/(\kappa^2 \tau_{Di})}}$"
            "\n\n"
            "Impact: 1 component fits a single tau_D (one population size). 2 components "
            "fits two independent tau_D's plus their amplitude split (f1/f2) -- use this if "
            "the curve visibly has a fast and a slow decay (e.g. free monomer + aggregate). "
            "More parameters also means more risk of an unstable/degenerate fit -- check the "
            "stability badge after fitting."
        ),
    )
    triplet = st.checkbox(
        "Include triplet/blinking term", key=f"{key_prefix}_triplet",
        help=(
            "Adds a fast photophysical-blinking prefactor on top of the diffusion term.\n\n"
            r"Formula: multiplies $G(\tau)$ by $\frac{1-T+T e^{-\tau/\tau_{trip}}}{1-T}$, with "
            r"$\tau_{trip} \sim 1\,\mu s$."
            "\n\n"
            "Impact: only reshapes the fastest (leftmost) part of the curve. Turn on if you "
            "see a fast decay at short tau that the diffusion term alone can't explain "
            "(common with organic-dye triplet states). Leave off if your shortest tau is "
            "already much longer than ~1 us, or it just adds an unconstrained nuisance "
            "parameter."
        ),
    )
    n_starts = st.number_input(
        "Multi-start attempts", min_value=1, max_value=20, value=5, key=f"{key_prefix}_nstarts",
        help=(
            "How many independent random initial guesses are fit and compared.\n\n"
            "Impact: this is purely a stability check, not part of G(tau)'s formula -- it "
            "doesn't change a converged fit's best answer, but more starts make it more "
            "likely to catch a fit that's actually unstable (parameters swing >20% depending "
            "on starting guess, or <60% of starts converge at all). Increase this for "
            "2-component fits, which are more prone to degenerate solutions than 1-component."
        ),
    )
    return n_components, triplet, n_starts
