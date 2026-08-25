import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_common import (
    init_defaults,
    sidebar_about,
    load_trace_cached,
    plot_fit_overlay,
    apply_chart_style,
    format_seconds,
    kappa_from_w,
    SERIES_COLORS,
    REFERENCE_LINE,
)
from core import engine
from core.validation import compare_multitau_to_fft, generate_synthetic_fcs_trace
from core.fitting import fit_curve

st.set_page_config(page_title="Validation", layout="wide")
init_defaults()
sidebar_about()
d = st.session_state["defaults"]
kappa = kappa_from_w(d["w_xy_um"], d["w_z_um"])

st.title("Validation")
st.caption(
    "Two independent correctness checks for the multi-tau engine, distinct from production analysis: "
    "(A) an exact FFT-based brute-force correlation cross-check, and (B) a synthetic-data recovery test."
)

with st.container(border=True):
    st.header("A. Multi-tau vs. exact FFT correlation (raw resolution)")
    st.caption(
        "Runs the multi-tau engine with segments=1 (raw resolution only, no coarsening) and compares its "
        "output, lag by lag, against an independent FFT-accelerated brute-force correlation. These should "
        "match exactly (within floating-point tolerance)."
    )

    val_source = st.radio("Trace for this check", ["Generate a quick synthetic trace", "Upload a real file"], horizontal=True)

    if val_source == "Upload a real file":
        uploaded = st.file_uploader("Trace file", type=["csv"], key="val_upload")
        trace_for_fft = None
        if uploaded is not None:
            trace_for_fft = load_trace_cached(uploaded.getvalue(), uploaded.name)
    else:
        if st.button("Generate synthetic trace for FFT check"):
            st.session_state["val_fft_synth"] = generate_synthetic_fcs_trace(
                tauD=1e-4, dt=1e-5, duration_s=2.0, N_particles=5.0, seed=0
            )
        trace_for_fft = None

    points_per_segment_check = st.number_input("Points to check (segment-0 lags)", min_value=1, max_value=100, value=15)

    if st.button("Run FFT cross-check", type="primary"):
        if val_source == "Upload a real file" and trace_for_fft is not None:
            ch1 = trace_for_fft.channels["CH1"]
            dt = trace_for_fft.dt
        elif "val_fft_synth" in st.session_state:
            ch1 = st.session_state["val_fft_synth"].counts
            dt = st.session_state["val_fft_synth"].dt
        else:
            st.error("Generate a synthetic trace or upload a file first.")
            st.stop()

        with st.spinner("Running FFT cross-check..."):
            max_abs_diff, max_rel_diff, all_close = compare_multitau_to_fft(ch1, ch1, dt, points_per_segment=points_per_segment_check)
        st.session_state["val_fft_result"] = (max_abs_diff, max_rel_diff, all_close)

    if "val_fft_result" in st.session_state:
        max_abs_diff, max_rel_diff, all_close = st.session_state["val_fft_result"]
        if all_close:
            st.success(f"PASS: multi-tau matches exact FFT correlation (max abs diff = {max_abs_diff:.3e}, max rel diff = {max_rel_diff:.3e})")
        else:
            st.error(f"FAIL: multi-tau does NOT match exact FFT correlation (max abs diff = {max_abs_diff:.3e}, max rel diff = {max_rel_diff:.3e})")

with st.container(border=True):
    st.header("B. Synthetic-data recovery (known diffusion time)")
    st.caption(
        "Generates a synthetic photon-count trace with a known input diffusion time, runs it through the "
        "full correlate + fit pipeline, and checks that the known tau_D is recovered within statistical noise."
    )

    c1, c2, c3, c4 = st.columns(4)
    tauD_true = c1.number_input("True tau_D (s)", min_value=1e-7, value=2e-4, format="%.6f")
    duration_s = c2.number_input("Duration (s)", min_value=1.0, value=30.0)
    N_particles = c3.number_input("N particles (true)", min_value=0.1, value=5.0)
    dt_synth = c4.number_input("dt (s)", min_value=1e-6, value=1e-5, format="%.6f")

    if st.button("Generate & fit synthetic trace", type="primary"):
        with st.spinner("Generating synthetic trace and fitting..."):
            synth = generate_synthetic_fcs_trace(tauD=tauD_true, dt=dt_synth, duration_s=duration_s, N_particles=N_particles, kappa=kappa, seed=42)
            tau, g, _ = engine.multiple_tau_correlate(
                synth.counts, synth.counts, synth.dt, segments=d["segments"], points_per_segment=d["points_per_segment"], base=d["base"]
            )
            fr = fit_curve(tau, g, n_components=1, kappa=kappa)
        st.session_state["val_synth"] = (synth, tau, g, fr)

    if "val_synth" in st.session_state:
        synth, tau, g, fr = st.session_state["val_synth"]
        if fr.success:
            pct_error = abs(fr.tauD[0] - synth.tauD_true) / synth.tauD_true * 100
            mcols = st.columns(4)
            mcols[0].metric("Recovered tau_D", format_seconds(fr.tauD[0]))
            mcols[1].metric("True tau_D", format_seconds(synth.tauD_true))
            mcols[2].metric("Error", f"{pct_error:.1f}%")
            mcols[3].metric("Recovered N (true)", f"{fr.N:.3g} ({synth.N_true:.3g})")
        else:
            st.error("Fit did not converge on the synthetic trace.")

        chart_title = "Synthetic G(tau)" if fr.success else "Synthetic G(tau) (fit did not converge)"
        st.plotly_chart(
            plot_fit_overlay(tau, g, fr.fit_curve, title=chart_title, color=SERIES_COLORS["acf_ch1"]),
            width="stretch",
        )

    st.subheader("Monte Carlo repeat (optional)")
    n_repeats = st.number_input("Number of repeats", min_value=2, max_value=200, value=20)
    if st.button("Run Monte Carlo repeat"):
        recovered = []
        progress = st.progress(0.0, text="Starting Monte Carlo repeat...")
        for i in range(int(n_repeats)):
            synth_i = generate_synthetic_fcs_trace(
                tauD=tauD_true, dt=dt_synth, duration_s=duration_s, N_particles=N_particles, kappa=kappa, seed=i
            )
            tau_i, g_i, _ = engine.multiple_tau_correlate(
                synth_i.counts, synth_i.counts, synth_i.dt, segments=d["segments"], points_per_segment=d["points_per_segment"], base=d["base"]
            )
            fr_i = fit_curve(tau_i, g_i, n_components=1, kappa=kappa)
            if fr_i.success:
                recovered.append(fr_i.tauD[0])
            progress.progress((i + 1) / n_repeats, text=f"Repeat {i + 1}/{int(n_repeats)}")

        recovered = np.array(recovered)
        if len(recovered):
            rcols = st.columns(3)
            rcols[0].metric("Mean recovered tau_D", format_seconds(recovered.mean()))
            rcols[1].metric("Std dev", format_seconds(recovered.std()))
            rcols[2].metric("Converged", f"{len(recovered)}/{int(n_repeats)}")
            fig2 = go.Figure(data=[go.Histogram(x=recovered, marker_color=SERIES_COLORS["acf_ch1"])])
            fig2.add_vline(x=tauD_true, line_color=REFERENCE_LINE, line_dash="dash", annotation_text="true tau_D")
            fig2.update_layout(
                xaxis_title="Recovered tau_D (s)", height=350, plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
            )
            st.plotly_chart(apply_chart_style(fig2), width="stretch")
