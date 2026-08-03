import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app_common import init_defaults, load_trace_cached, window_channels
from core.binning import apply_point_binning
from core.correlate import compute_all_correlations
from core.fitting import multi_start_fit
from core.fccs import bound_fraction_from_cross, amplitude_at_zero
from core.export import batch_comparison_to_df, df_to_csv_bytes

st.set_page_config(page_title="Batch / Time Course", layout="wide")
init_defaults()
d = st.session_state["defaults"]

st.title("Batch / Time Course Processing")
st.caption(
    "Process a set of files (e.g. 0hr / 3hr / 48hr) with one shared settings panel applied "
    "identically to every file, and compare results across the series."
)

uploaded_files = st.file_uploader("Trace files", type=None, accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload two or more trace files to begin.")
    st.stop()

st.subheader("1. Labels")
label_df = pd.DataFrame({"filename": [f.name for f in uploaded_files], "label": [f.name for f in uploaded_files]})
label_df = st.data_editor(label_df, key="batch_label_editor", disabled=["filename"], hide_index=True)

st.subheader("2. Shared settings (applied to every file)")
c1, c2, c3, c4 = st.columns(4)
start_s = c1.number_input("Window start (s, blank-equivalent = 0)", min_value=0.0, value=0.0, key="batch_start")
use_end = c2.checkbox("Trim end of window?", value=False, key="batch_use_end")
end_s = c3.number_input("Window end (s)", min_value=0.0, value=30.0, key="batch_end", disabled=not use_end)
bin_points = c4.number_input("Binning points", min_value=1, value=1, key="batch_bin_points")

c5, c6, c7 = st.columns(3)
segments = c5.number_input("Segments", min_value=1, max_value=20, value=d["segments"], key="batch_segments")
points_per_segment = c6.number_input("Points/segment", min_value=1, max_value=100, value=d["points_per_segment"], key="batch_pps")
base = c7.number_input("Grouping base", min_value=2, max_value=16, value=d["base"], key="batch_base")

st.markdown("**Fit settings (applied per channel/cross, shared across all files)**")
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    st.markdown("CH1 ACF")
    ncomp1 = st.radio("Components##ch1", [1, 2], key="batch_ncomp1", horizontal=True)
    trip1 = st.checkbox("Triplet##ch1", key="batch_trip1")
with fcol2:
    st.markdown("CH2 ACF")
    ncomp2 = st.radio("Components##ch2", [1, 2], key="batch_ncomp2", horizontal=True)
    trip2 = st.checkbox("Triplet##ch2", key="batch_trip2")
with fcol3:
    st.markdown("Cross-correlation")
    ncompx = st.radio("Components##x", [1, 2], key="batch_ncompx", horizontal=True)
    tripx = st.checkbox("Triplet##x", key="batch_tripx")
n_starts = st.number_input("Multi-start attempts per fit", min_value=1, max_value=20, value=5, key="batch_nstarts")

if st.button("Run batch", type="primary"):
    batch_results = {}
    rows = []
    progress = st.progress(0.0)
    for i, f in enumerate(uploaded_files):
        label = label_df.loc[label_df["filename"] == f.name, "label"].iloc[0]
        trace = load_trace_cached(f.getvalue(), f.name)
        end = end_s if use_end else float(trace.time[-1])
        w_time, w_channels = window_channels(trace.time, trace.channels, start_s, end)
        _, b_channels, eff_dt, _ = apply_point_binning(w_time, w_channels, bin_points)
        corr = compute_all_correlations(b_channels, eff_dt, segments=segments, points_per_segment=points_per_segment, base=base)

        row = {"filename": f.name, "label": label, "n_channels": trace.n_channels}
        fits = {}
        if "acf_ch1" in corr:
            rep1 = multi_start_fit(corr["acf_ch1"].tau, corr["acf_ch1"].g, n_components=ncomp1, triplet=trip1, n_starts=n_starts, kappa=d["kappa"])
            fits["acf_ch1"] = rep1
            row["tauD_ch1"] = rep1.chosen_result.tauD[0] if rep1.chosen_result.tauD else np.nan
            row["N_ch1"] = rep1.chosen_result.N
            row["stable_ch1"] = rep1.is_stable
        if "acf_ch2" in corr:
            rep2 = multi_start_fit(corr["acf_ch2"].tau, corr["acf_ch2"].g, n_components=ncomp2, triplet=trip2, n_starts=n_starts, kappa=d["kappa"])
            fits["acf_ch2"] = rep2
            row["tauD_ch2"] = rep2.chosen_result.tauD[0] if rep2.chosen_result.tauD else np.nan
            row["N_ch2"] = rep2.chosen_result.N
            row["stable_ch2"] = rep2.is_stable
        if "cross" in corr:
            repx = multi_start_fit(corr["cross"].tau, corr["cross"].g, n_components=ncompx, triplet=tripx, n_starts=n_starts, kappa=d["kappa"])
            fits["cross"] = repx
            row["stable_cross"] = repx.is_stable
        if "acf_ch1" in fits and "acf_ch2" in fits and "cross" in fits:
            g1 = amplitude_at_zero(fits["acf_ch1"].chosen_result)
            g2 = amplitude_at_zero(fits["acf_ch2"].chosen_result)
            gx = amplitude_at_zero(fits["cross"].chosen_result)
            row["bound_fraction_tau_fl"] = bound_fraction_from_cross(gx, g1)
            row["bound_fraction_nt"] = bound_fraction_from_cross(gx, g2)

        rows.append(row)
        batch_results[f.name] = {"trace": trace, "corr": corr, "fits": fits, "label": label}
        progress.progress((i + 1) / len(uploaded_files))

    st.session_state["batch_results"] = batch_results
    st.session_state["batch_comparison_df"] = batch_comparison_to_df(rows)

if "batch_comparison_df" not in st.session_state:
    st.stop()

st.subheader("3. Comparison")
comp_df = st.session_state["batch_comparison_df"]
st.dataframe(comp_df, width="stretch")
st.download_button("Download comparison table (CSV)", df_to_csv_bytes(comp_df), file_name="batch_comparison.csv")

plot_cols = st.columns(2)
if "tauD_ch1" in comp_df or "tauD_ch2" in comp_df:
    melt_cols = [c for c in ["tauD_ch1", "tauD_ch2"] if c in comp_df]
    melted = comp_df.melt(id_vars=["label"], value_vars=melt_cols, var_name="channel", value_name="tauD")
    fig = px.scatter(melted, x="label", y="tauD", color="channel", title="Diffusion time (tau_D) vs. time course")
    plot_cols[0].plotly_chart(fig, width="stretch")

if "bound_fraction_tau_fl" in comp_df:
    fig2 = px.scatter(
        comp_df, x="label", y=["bound_fraction_tau_fl", "bound_fraction_nt"], title="Bound fraction vs. time course"
    )
    plot_cols[1].plotly_chart(fig2, width="stretch")
