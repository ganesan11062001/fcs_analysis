import streamlit as st

from app_common import (
    init_defaults,
    sidebar_about,
    load_trace_cached,
    plot_raw_trace_with_window,
    time_window_selector,
    window_channels,
    plot_correlation_curve,
    plot_fit_overlay,
    fit_settings_widgets,
    format_seconds,
    SERIES_COLORS,
)
import numpy as np

from core.binning import apply_point_binning
from core.correlate import compute_all_correlations, compute_correlation_error
from core.fitting import multi_start_fit
from core.fccs import bound_fraction_from_cross, amplitude_at_zero
from core.export import (
    correlation_results_to_df,
    correlation_results_to_df_with_error,
    correlation_results_to_vistavision_csv_text,
    fit_result_to_df,
    stability_report_to_df,
    fccs_result_to_df,
    df_to_csv_bytes,
)

st.set_page_config(page_title="Single File Analysis", layout="wide")
init_defaults()
sidebar_about()
d = st.session_state["defaults"]

st.title("Single File Analysis")
st.caption("Load one trace, trim it, correlate, and fit -- the core single-file pipeline.")

uploaded = st.file_uploader("VistaVision trace export (single- or dual-channel CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload a trace file to begin.")
    st.stop()

trace = load_trace_cached(uploaded.getvalue(), uploaded.name)
st.session_state["sf_trace"] = trace

info_cols = st.columns(4)
info_cols[0].metric("Channels", trace.n_channels)
info_cols[1].metric("Rows", f"{trace.n_rows:,}")
info_cols[2].metric("Raw dt", format_seconds(trace.dt))
info_cols[3].metric("Raw sampling rate", f"{1.0 / trace.dt:,.0f} Hz")

if trace.used_fallback_parser:
    st.warning(f"Fast parser was bypassed for this file: {trace.fallback_reason}")

with st.container(border=True):
    st.subheader("1. Trim the time window")
    t0, t1 = time_window_selector(trace.time, "sf")
    st.plotly_chart(plot_raw_trace_with_window(trace.time, trace.channels, t0, t1), width="stretch")

windowed_time, windowed_channels = window_channels(trace.time, trace.channels, t0, t1)

with st.container(border=True):
    st.subheader("2. Binning points")
    bin_points = st.number_input(
        "Binning points (group N raw points together before correlating)", min_value=1, value=1, key="sf_bin_points",
        help=(
            "Groups N consecutive raw samples into one averaged point BEFORE correlating "
            "(separate from the multi-tau engine's own segments/points-per-segment/base "
            "coarsening below, which happens after this step).\n\n"
            r"Formula: $\Delta t_{eff} = \Delta t_{raw} \cdot N$, $\ f_{eff} = 1/\Delta t_{eff}$, where "
            r"$\Delta t_{raw}$ is read directly from your CSV's time column. N=1 means no binning is "
            "applied at all -- the 'effective rate' shown below is then just your file's own "
            "raw acquisition rate (e.g. a file recorded at 10 us/point reports 100,000 Hz "
            "here for that reason alone, not because anything was chosen or binned).\n\n"
            "Impact: binning reduces shot-noise per point (fewer, coarser, less noisy points) "
            r"but also blurs out any real correlation decay faster than $\Delta t_{eff}$ -- so N "
            r"should stay well below your fastest expected $\tau_D$, or you'll wash out the "
            "signal you're trying to measure. Most FCS analyses leave this at 1 and let the "
            "multi-tau engine's own coarsening (below) do all the log-spacing."
        ),
    )
    _, binned_channels, effective_dt, effective_rate = apply_point_binning(windowed_time, windowed_channels, bin_points)
    st.caption(f"Effective sampling rate after binning: **{effective_rate:,.1f} Hz** (dt = {format_seconds(effective_dt)})")

with st.container(border=True):
    st.subheader("3. Multi-tau correlation settings")
    c1, c2, c3 = st.columns(3)
    segments = c1.number_input(
        "Segments", min_value=1, max_value=20, value=d["segments"], key="sf_segments",
        help=(
            "How many multi-tau blocks are computed, each at a coarser lag spacing than "
            "the last.\n\n"
            r"Formula: total $\tau$ points $= \text{segments} \times \text{points\_per\_segment}$; "
            r"segment $k$'s lag step $\Delta t_k = \Delta t_{raw} \cdot \text{base}^k$."
            "\n\n"
            "Impact: more segments extend how far out in tau the curve reaches (needed to "
            "see the decay fully return to baseline), but each extra segment needs "
            "proportionally more raw trace length to fill with real data -- too many for a "
            "short trace just adds tau points with very low n_samples (flagged unreliable)."
        ),
    )
    points_per_segment = c2.number_input(
        "Points per segment", min_value=1, max_value=100, value=d["points_per_segment"], key="sf_pps",
        help=(
            "How many correlation points are computed at each segment's lag spacing before "
            "moving to the next (coarser) segment.\n\n"
            r"Formula: within segment $k$, points are computed at lags "
            r"$\tau = \Delta t_{raw} \cdot \text{base}^k \cdot i$, for $i = 1, \dots, \text{points\_per\_segment}$."
            "\n\n"
            "Impact: more points per segment gives finer resolution within each decade of "
            "tau, at the cost of more compute per correlation run. 15 is VistaVision's own "
            "documented default (manual sec. 13.4.1) and is the standard choice."
        ),
    )
    base = c3.number_input(
        "Grouping base", min_value=2, max_value=16, value=d["base"], key="sf_base",
        help=(
            "The coarsening factor applied to the lag spacing between successive "
            "segments.\n\n"
            r"Formula: segment $k$'s effective $\Delta t_k = \Delta t_{raw} \cdot \text{base}^k$."
            "\n\n"
            "Impact: base=4 (the standard multi-tau choice) roughly quadruples the tau "
            "spacing each segment, giving even log-spaced coverage across many decades of "
            "tau without needing hundreds of segments. A larger base covers more decades "
            "per segment but coarsens (loses within-decade resolution) faster."
        ),
    )

    estimate_error = st.checkbox(
        "Also estimate per-point error (sub-block standard error)", value=True, key="sf_estimate_error",
        help="Splits the trace into N contiguous sub-blocks, correlates each independently, and uses the "
        "spread across sub-blocks as each tau point's uncertainty -- the standard block-averaging convention "
        "used broadly across FCS correlator software. Roughly doubles compute time for this step.",
    )
    n_blocks = st.number_input(
        "Number of sub-blocks", min_value=2, max_value=50, value=10, key="sf_n_blocks", disabled=not estimate_error
    )

    if st.button("Run correlation", type="primary"):
        with st.spinner("Running multi-tau correlation..."):
            corr_results = compute_all_correlations(
                binned_channels, effective_dt, segments=segments, points_per_segment=points_per_segment, base=base
            )
            corr_errors = {}
            if estimate_error:
                corr_errors = compute_correlation_error(
                    corr_results, binned_channels, effective_dt,
                    segments=segments, points_per_segment=points_per_segment, base=base, n_blocks=n_blocks,
                )
        st.session_state["sf_corr_results"] = corr_results
        st.session_state["sf_corr_errors"] = corr_errors
        st.session_state["sf_mean_rates"] = [float(np.mean(arr)) / effective_dt for arr in binned_channels.values()]
        st.session_state["sf_sampling_rate_hz"] = 1.0 / effective_dt
        st.session_state.pop("sf_fit_results", None)

if "sf_corr_results" not in st.session_state:
    st.stop()

results = st.session_state["sf_corr_results"]

with st.container(border=True):
    st.subheader("4. Correlation curves")
    fig, any_unreliable = plot_correlation_curve(results, d["min_reliable_n_samples"], title=uploaded.name)
    st.plotly_chart(fig, width="stretch")
    if any_unreliable:
        st.caption(
            f"Points marked with 'x' are based on fewer than {d['min_reliable_n_samples']} averaged samples "
            "(long-tau statistical unreliability, especially near the edge of the acquisition window)."
        )

    errors = st.session_state.get("sf_corr_errors", {})
    dl_cols = st.columns(3)
    dl_cols[0].download_button(
        "Download correlation curves (CSV)",
        df_to_csv_bytes(correlation_results_to_df(results)),
        file_name=f"{uploaded.name}_correlation.csv",
        help="tau, G(tau), n_samples per curve -- no error column.",
    )
    dl_cols[1].download_button(
        "Download with error (CSV)",
        df_to_csv_bytes(correlation_results_to_df_with_error(results, errors)),
        file_name=f"{uploaded.name}_correlation_with_error.csv",
        disabled=not errors,
        help="Same as above plus a per-point standard-error column (enable the error "
        "estimate checkbox above and re-run correlation if this is disabled)."
        if not errors else "tau, G(tau), error, n_samples per curve.",
    )
    dl_cols[2].download_button(
        "Download (VistaVision format)",
        correlation_results_to_vistavision_csv_text(
            results, errors,
            sampling_rate_hz=st.session_state.get("sf_sampling_rate_hz", 1.0 / effective_dt),
            mean_rates=st.session_state.get("sf_mean_rates", []),
        ),
        file_name=f"{uploaded.name}_correlation_vistavision.csv",
        help="Matches VistaVision's [HeaderX]/[Data] structure and tau,G,err column layout. Does NOT "
        "reproduce VistaVision's segment-boundary duplicate rows (an undocumented internal quirk, not "
        "part of the published multi-tau algorithm this engine is validated against).",
    )

with st.container(border=True):
    st.subheader("5. Model fitting")
    fit_results = st.session_state.setdefault("sf_fit_results", {})

    fit_panels = [("acf_ch1", "CH1 autocorrelation"), ("acf_ch2", "CH2 autocorrelation"), ("cross", "Cross-correlation")]
    fit_cols = st.columns(sum(1 for kind, _ in fit_panels if kind in results))
    col_idx = 0
    for kind, label in fit_panels:
        if kind not in results:
            continue
        with fit_cols[col_idx]:
            n_components, triplet, n_starts = fit_settings_widgets(f"sf_{kind}", label)
            if st.button(f"Fit {label}", key=f"sf_fit_btn_{kind}"):
                with st.spinner(f"Fitting {label} ({n_starts} starts)..."):
                    report = multi_start_fit(
                        results[kind].tau, results[kind].g, n_components=n_components, triplet=triplet,
                        n_starts=n_starts, kappa=d["kappa"],
                    )
                fit_results[kind] = report
            if kind in fit_results:
                report = fit_results[kind]
                fr = report.chosen_result
                badge = "STABLE" if report.is_stable else "UNSTABLE -- fit is sensitive to starting guess"
                (st.success if report.is_stable else st.error)(
                    f"{badge} ({report.converged_fraction:.0%} of {n_starts} starts converged)"
                )
                st.plotly_chart(
                    plot_fit_overlay(results[kind].tau, results[kind].g, fr.fit_curve, title=label, color=SERIES_COLORS.get(kind)),
                    width="stretch",
                )
                for i, td in enumerate(fr.tauD):
                    st.metric(f"tau_D{'' if len(fr.tauD) == 1 else f' ({i + 1})'}", format_seconds(td))
                st.metric("N (amplitude)", f"{fr.N:.3g}")
                st.caption(f"Reduced chi-squared = {fr.redchi:.4g}")
                with st.expander("Per-start results (stability check)"):
                    st.dataframe(stability_report_to_df(report, label=label))
                st.download_button(
                    f"Download {label} fit parameters (CSV)",
                    df_to_csv_bytes(fit_result_to_df(fr, label=label)),
                    file_name=f"{uploaded.name}_{kind}_fit.csv",
                    key=f"sf_dl_fit_{kind}",
                )
        col_idx += 1

if "acf_ch1" in fit_results and "acf_ch2" in fit_results and "cross" in fit_results:
    with st.container(border=True):
        st.subheader("6. FCCS bound fraction")
        g_ch1_0 = amplitude_at_zero(fit_results["acf_ch1"].chosen_result)
        g_ch2_0 = amplitude_at_zero(fit_results["acf_ch2"].chosen_result)
        g_x_0 = amplitude_at_zero(fit_results["cross"].chosen_result)
        bound_ch2_species = bound_fraction_from_cross(g_x_0, g_ch1_0)  # CH2 species' bound fraction uses CH1 partner amplitude
        bound_ch1_species = bound_fraction_from_cross(g_x_0, g_ch2_0)  # CH1 species' bound fraction uses CH2 partner amplitude

        bcols = st.columns(2)
        bcols[0].metric("Bound fraction of CH2 species", f"{bound_ch2_species:.3f}")
        bcols[1].metric("Bound fraction of CH1 species", f"{bound_ch1_species:.3f}")
        st.caption("No spectral crosstalk correction applied (uncorrected amplitude-ratio formula).")
        st.download_button(
            "Download FCCS bound fractions (CSV)",
            df_to_csv_bytes(fccs_result_to_df(bound_ch2_species, bound_ch1_species, label=uploaded.name)),
            file_name=f"{uploaded.name}_fccs.csv",
        )
