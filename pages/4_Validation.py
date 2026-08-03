import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_common import init_defaults, load_trace_cached
from core import engine
from core.validation import compare_multitau_to_fft, generate_synthetic_fcs_trace
from core.fitting import fit_curve

st.set_page_config(page_title="Validation", layout="wide")
init_defaults()
d = st.session_state["defaults"]

st.title("Validation")
st.caption(
    "Two independent correctness checks for the multi-tau engine, distinct from production analysis: "
    "(A) an exact FFT-based brute-force correlation cross-check, and (B) a synthetic-data recovery test."
)

st.header("A. Multi-tau vs. exact FFT correlation (raw resolution)")
st.caption(
    "Runs the multi-tau engine with segments=1 (raw resolution only, no coarsening) and compares its "
    "output, lag by lag, against an independent FFT-accelerated brute-force correlation. These should "
    "match exactly (within floating-point tolerance)."
)

val_source = st.radio("Trace for this check", ["Generate a quick synthetic trace", "Upload a real file"], horizontal=True)

if val_source == "Upload a real file":
    uploaded = st.file_uploader("Trace file", type=None, key="val_upload")
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

    max_abs_diff, max_rel_diff, all_close = compare_multitau_to_fft(ch1, ch1, dt, points_per_segment=points_per_segment_check)
    st.session_state["val_fft_result"] = (max_abs_diff, max_rel_diff, all_close)

if "val_fft_result" in st.session_state:
    max_abs_diff, max_rel_diff, all_close = st.session_state["val_fft_result"]
    if all_close:
        st.success(f"PASS: multi-tau matches exact FFT correlation (max abs diff = {max_abs_diff:.3e}, max rel diff = {max_rel_diff:.3e})")
    else:
        st.error(f"FAIL: multi-tau does NOT match exact FFT correlation (max abs diff = {max_abs_diff:.3e}, max rel diff = {max_rel_diff:.3e})")

st.divider()

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
    synth = generate_synthetic_fcs_trace(tauD=tauD_true, dt=dt_synth, duration_s=duration_s, N_particles=N_particles, kappa=d["kappa"], seed=42)
    tau, g, _ = engine.multiple_tau_correlate(
        synth.counts, synth.counts, synth.dt, segments=d["segments"], points_per_segment=d["points_per_segment"], base=d["base"]
    )
    fr = fit_curve(tau, g, n_components=1, kappa=d["kappa"])
    st.session_state["val_synth"] = (synth, tau, g, fr)

if "val_synth" in st.session_state:
    synth, tau, g, fr = st.session_state["val_synth"]
    if fr.success:
        pct_error = abs(fr.tauD[0] - synth.tauD_true) / synth.tauD_true * 100
        st.success(f"Recovered tau_D = {fr.tauD[0]:.4g} s vs. true {synth.tauD_true:.4g} s ({pct_error:.1f}% error); recovered N = {fr.N:.3g} vs. true {synth.N_true:.3g}")
    else:
        st.error("Fit did not converge on the synthetic trace.")

    order = np.argsort(tau)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tau[order], y=g[order], mode="markers", name="synthetic G(tau)"))
    if fr.success:
        fig.add_trace(go.Scatter(x=tau[order], y=fr.fit_curve[order], mode="lines", name="fit"))
    fig.update_xaxes(type="log", title="tau (s)")
    fig.update_yaxes(title="G(tau)")
    st.plotly_chart(fig, width="stretch")

st.subheader("Monte Carlo repeat (optional)")
n_repeats = st.number_input("Number of repeats", min_value=2, max_value=200, value=20)
if st.button("Run Monte Carlo repeat"):
    recovered = []
    progress = st.progress(0.0)
    for i in range(int(n_repeats)):
        synth_i = generate_synthetic_fcs_trace(
            tauD=tauD_true, dt=dt_synth, duration_s=duration_s, N_particles=N_particles, kappa=d["kappa"], seed=i
        )
        tau_i, g_i, _ = engine.multiple_tau_correlate(
            synth_i.counts, synth_i.counts, synth_i.dt, segments=d["segments"], points_per_segment=d["points_per_segment"], base=d["base"]
        )
        fr_i = fit_curve(tau_i, g_i, n_components=1, kappa=d["kappa"])
        if fr_i.success:
            recovered.append(fr_i.tauD[0])
        progress.progress((i + 1) / n_repeats)

    recovered = np.array(recovered)
    if len(recovered):
        st.write(
            f"Recovered tau_D: mean = {recovered.mean():.4g} s, std = {recovered.std():.4g} s "
            f"(true = {tauD_true:.4g} s, {len(recovered)}/{int(n_repeats)} fits converged)"
        )
        fig2 = go.Figure(data=[go.Histogram(x=recovered)])
        fig2.add_vline(x=tauD_true, line_color="red", annotation_text="true tau_D")
        fig2.update_layout(xaxis_title="Recovered tau_D (s)", height=350)
        st.plotly_chart(fig2, width="stretch")
