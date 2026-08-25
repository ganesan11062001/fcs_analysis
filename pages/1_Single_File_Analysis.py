import numpy as np
import pandas as pd
import streamlit as st

from app_common import (
    init_defaults,
    sidebar_about,
    load_trace_cached,
    run_auto_window_search_cached,
    window_length_options_for_trace,
    plot_raw_trace_with_window,
    plot_correlation_curve,
    plot_fit_overlay,
    fit_settings_widgets,
    format_seconds,
    kappa_from_w,
    SERIES_COLORS,
)

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
st.caption(
    "Upload a trace, tell the app how long a window to correlate, and it automatically "
    "slides that window across the whole acquisition to find the cleanest stretch -- no "
    "manual trimming required."
)

uploaded = st.file_uploader("VistaVision trace export (single- or dual-channel CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload a trace file to begin.")
    st.stop()

trace = load_trace_cached(uploaded.getvalue(), uploaded.name)

info_cols = st.columns(4)
info_cols[0].metric("Channels", trace.n_channels)
info_cols[1].metric("Rows", f"{trace.n_rows:,}")
info_cols[2].metric("Raw dt", format_seconds(trace.dt))
info_cols[3].metric("Raw sampling rate", f"{1.0 / trace.dt:,.0f} Hz")

if trace.used_fallback_parser:
    st.warning(f"Fast parser was bypassed for this file: {trace.fallback_reason}")

t_min, t_max = float(trace.time[0]), float(trace.time[-1])
window_options = window_length_options_for_trace(t_min, t_max)
window_length_s = st.selectbox(
    "Time period per window (s)",
    options=window_options,
    index=len(window_options) // 2,
    key="sf_window_length_s",
    help=(
        "The app slides a window of this length across the whole trace in 1-second steps "
        "(0-L, 1-(1+L), 2-(2+L), ... to the end) and automatically keeps the step with the "
        "cleanest correlation curve. Pick a period comparable to how long you'd expect the "
        "signal to stay stable -- shorter periods give more candidates to choose from but "
        "each has less data (noisier); longer periods have less noise per window but fewer "
        "candidates to pick from."
    ),
)

with st.expander("Advanced settings (defaults match VistaVision manual sec. 13.4.1)"):
    c1, c2, c3 = st.columns(3)
    segments = c1.number_input(
        "Segments", min_value=1, max_value=20, value=d["segments"], key="sf_segments",
        help=(
            "How many multi-tau blocks are computed, each at a coarser lag spacing than "
            "the last.\n\n"
            r"Formula: total $\tau$ points $= \text{segments} \times \text{points\_per\_segment}$; "
            r"segment $k$'s lag step $\Delta t_k = \Delta t_{raw} \cdot \text{base}^k$."
            "\n\n"
            "Impact: more segments extend how far out in tau the curve reaches, but each "
            "extra segment needs proportionally more trace length to fill with real data."
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
            "15 is VistaVision's own documented default (manual sec. 13.4.1)."
        ),
    )
    base = c3.number_input(
        "Grouping base", min_value=2, max_value=16, value=d["base"], key="sf_base",
        help=(
            "The coarsening factor applied to the lag spacing between successive segments.\n\n"
            r"Formula: segment $k$'s effective $\Delta t_k = \Delta t_{raw} \cdot \text{base}^k$."
            "\n\n"
            "base=4 (the standard multi-tau choice) roughly quadruples the tau spacing each "
            "segment, giving even log-spaced coverage across many decades of tau."
        ),
    )
    c4, c5 = st.columns(2)
    bin_points = c4.number_input(
        "Binning points (group N raw points together before correlating)", min_value=1, value=1, key="sf_bin_points",
        help=(
            "Groups N consecutive raw samples into one averaged point BEFORE correlating.\n\n"
            r"Formula: $\Delta t_{eff} = \Delta t_{raw} \cdot N$. N=1 means no binning is applied "
            "at all.\n\n"
            "Impact: binning reduces shot-noise per point but also blurs out any real "
            r"correlation decay faster than $\Delta t_{eff}$ -- most FCS analyses leave this at 1."
        ),
    )
    n_blocks = c5.number_input(
        "Sub-blocks for error estimate / window scoring", min_value=2, max_value=50, value=5, key="sf_n_blocks",
        help=(
            "Used two ways: (1) the per-point error bars on the final curve, and (2) scoring "
            "each candidate time window during the automatic search below (higher "
            "signal-to-noise = cleaner window). More sub-blocks = a more stable error estimate "
            "but needs a longer trace per candidate window."
        ),
    )
    wcol1, wcol2 = st.columns(2)
    w_xy_um = wcol1.number_input(
        "w_xy (µm, radial)", min_value=0.01, value=d["w_xy_um"], step=0.01, format="%.3f", key="sf_w_xy",
        help="The confocal detection volume's radial (lateral) beam waist, in microns. Only ever "
        "used to derive kappa below -- the fit model has no separate dependence on its absolute "
        "value. Overrides the Home page's global default for this session/page only.",
    )
    w_z_um = wcol2.number_input(
        "w_z (µm, axial)", min_value=0.01, value=d["w_z_um"], step=0.01, format="%.3f", key="sf_w_z",
        help="The confocal detection volume's axial beam waist, in microns. Only ever used to "
        "derive kappa below. Overrides the Home page's global default for this session/page only.",
    )
    kappa = kappa_from_w(w_xy_um, w_z_um)
    st.caption(
        f"kappa = w_z / w_xy = **{kappa:.3g}** -- the only quantity that actually enters the fit "
        r"model's diffusion term $\frac{1}{\sqrt{1+\tau/(\kappa^2 \tau_D)}}$, not in computing "
        "G(tau) itself. Fixed rather than floated because it's badly degenerate with tau_D "
        "without an independent volume calibration -- a wrong value biases the fitted tau_D, "
        "especially at long tau. Typical confocal setups: kappa ~ 3-6."
    )

run_id = (
    uploaded.name, uploaded.size, window_length_s, segments, points_per_segment, base, bin_points, n_blocks, kappa,
)

auto = run_auto_window_search_cached(
    uploaded.getvalue(), uploaded.name, window_length_s, segments, points_per_segment, base, bin_points, n_blocks
)
best = auto.chosen
valid_candidates = [c for c in auto.candidates if not c.failed]
if not valid_candidates:
    st.error(
        "Every candidate window was too short to score (not enough data for the configured "
        "number of sub-blocks). Try a shorter window length or fewer sub-blocks in Advanced settings."
    )
    st.stop()


def _option_label(c):
    tag = " -- auto-picked" if c.label == best.label else ""
    return f"{c.label} (SNR {c.score:.3g}){tag}"


option_labels = [_option_label(c) for c in valid_candidates]
label_lookup = dict(zip(option_labels, valid_candidates))
default_option = _option_label(best)

if st.session_state.get("sf_run_id") != run_id:
    st.session_state["sf_run_id"] = run_id
    st.session_state.pop("sf_fit_results", None)
    st.session_state["sf_window_choice"] = default_option
    st.session_state["sf_fit_window_acf_ch1"] = default_option
    st.session_state["sf_fit_window_acf_ch2"] = default_option

with st.container(border=True):
    st.subheader("1. Time window")
    selected_option = st.selectbox(
        "Window to use",
        options=option_labels,
        key="sf_window_choice",
        help=(
            "Defaults to the auto-picked window (highest correlation-curve signal-to-noise "
            "ratio among all candidates from the sliding search). Pick any other candidate "
            "to use it instead -- the curve, fit, and exports below all update to match "
            "whichever window is selected here."
        ),
    )
    chosen = label_lookup[selected_option]
    st.plotly_chart(plot_raw_trace_with_window(trace.time, trace.channels, chosen.t0, chosen.t1), width="stretch")
    st.caption(
        f"Slid a {window_length_s}s window across the trace in 1s steps and evaluated "
        f"{len(auto.candidates)} candidates (t = {format_seconds(chosen.t0)} to {format_seconds(chosen.t1)} "
        f"currently selected)."
    )
    with st.expander(f"All {len(auto.candidates)} candidates"):
        cand_df = pd.DataFrame(
            [
                {
                    "window": c.label,
                    "t0 (s)": c.t0,
                    "t1 (s)": c.t1,
                    "SNR score": c.score if not c.failed else None,
                    "auto-picked": c.label == best.label,
                    "currently selected": c.label == chosen.label,
                    "skipped": c.failed,
                }
                for c in auto.candidates
            ]
        )
        st.dataframe(cand_df, width="stretch", hide_index=True)
        st.caption(
            "SNR score = median |G(tau)| / standard-error(tau) across the curve (higher = cleaner). "
            "A window is skipped if it's too short to split into the configured number of "
            "sub-blocks. See the Methodology page for the exact formula."
        )

    with st.expander(f"View all {len(valid_candidates)} candidate curves"):
        st.caption("Click \"Use this window\" under any plot to switch the selection above to it.")
        grid_cols = st.columns(3)
        for i, c in enumerate(valid_candidates):
            with grid_cols[i % 3]:
                with st.container(border=True):
                    tags = []
                    if c.label == best.label:
                        tags.append("auto-picked")
                    if c.label == chosen.label:
                        tags.append("selected")
                    title = c.label + (f" ({', '.join(tags)})" if tags else "")
                    fig, _ = plot_correlation_curve(
                        c.corr_results, d["min_reliable_n_samples"], title=title, height=260
                    )
                    st.plotly_chart(fig, width="stretch", key=f"sf_grid_plot_{c.label}")
                    st.caption(f"SNR {c.score:.3g}")
                    if c.label != chosen.label:
                        # Setting st.session_state[key] for an already-instantiated widget raises --
                        # must be done in an on_click callback, which runs before the rerun, not here.
                        st.button(
                            "Use this window", key=f"sf_grid_use_{c.label}",
                            on_click=lambda opt=_option_label(c): st.session_state.update(sf_window_choice=opt),
                        )
                    else:
                        st.caption("Currently selected")

results = chosen.corr_results
errors = chosen.corr_errors

with st.container(border=True):
    st.subheader("2. Correlation curves")
    st.caption(
        f"Window: **{chosen.label}** "
        f"(t = {format_seconds(chosen.t0)} to {format_seconds(chosen.t1)})"
    )
    fig, any_unreliable = plot_correlation_curve(
        results, d["min_reliable_n_samples"], title=f"{uploaded.name} -- {chosen.label}"
    )
    st.plotly_chart(fig, width="stretch")
    if any_unreliable:
        st.caption(
            f"Points marked with 'x' are based on fewer than {d['min_reliable_n_samples']} averaged samples "
            "(long-tau statistical unreliability, especially near the edge of the acquisition window)."
        )

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
        help="tau, G(tau), error, n_samples per curve.",
    )
    dl_cols[2].download_button(
        "Download (VistaVision format)",
        correlation_results_to_vistavision_csv_text(
            results, errors, sampling_rate_hz=1.0 / chosen.eff_dt, mean_rates=chosen.mean_rates,
            measurement_time_s=chosen.t1 - chosen.t0, raw_data_file=uploaded.name,
        ),
        file_name=f"{uploaded.name}_correlation_vistavision.csv",
        help="Matches a real VistaVision [HeaderV]/[Data] export's structure (metadata header + one "
        "row per quantity: Tau, ChN AutoCorrelation, ChN AutoCorrelation Standard Deviation, "
        "Ch0x1 CrossCorrelation). Does NOT reproduce VistaVision's segment-boundary duplicate Tau "
        "values -- an undocumented internal quirk with no known rule to reproduce without the raw "
        "trace that produced it.",
    )

with st.container(border=True):
    st.subheader("3. Model fitting (optional)")
    st.caption(
        "CH1 and CH2 can each be fit from a different candidate window -- pick any window found "
        "by the search in step 1 independently per channel. Cross-correlation needs both channels "
        "sampled over the exact same window (it's a paired CH1(t)*CH2(t+tau) product), so it's only "
        "available when both channels currently use the same one."
    )
    fit_results = st.session_state.setdefault("sf_fit_results", {})
    fit_window_used = st.session_state.setdefault("sf_fit_window_used", {})

    fit_panels = [("acf_ch1", "CH1 autocorrelation"), ("acf_ch2", "CH2 autocorrelation"), ("cross", "Cross-correlation")]
    fit_cols = st.columns(sum(1 for kind, _ in fit_panels if kind in results))
    col_idx = 0
    for kind, label in fit_panels:
        if kind not in results:
            continue
        with fit_cols[col_idx]:
            if kind in ("acf_ch1", "acf_ch2"):
                fit_window_option = st.selectbox(
                    f"Window for {label} fit", options=option_labels, key=f"sf_fit_window_{kind}",
                    help="Defaults to the window selected in step 1, but you can fit this channel "
                    "from a different candidate window than the other channel.",
                )
                kind_result = label_lookup[fit_window_option].corr_results[kind]
                kind_error = label_lookup[fit_window_option].corr_errors.get(kind)
            else:
                ch1_window = st.session_state.get("sf_fit_window_acf_ch1")
                ch2_window = st.session_state.get("sf_fit_window_acf_ch2")
                if ch1_window != ch2_window:
                    fit_results.pop("cross", None)
                    fit_window_used.pop("cross", None)
                    st.info(
                        "CH1 and CH2 are currently using different windows "
                        f"({ch1_window} vs {ch2_window}), so cross-correlation isn't available -- "
                        "pick the same window for both to enable it."
                    )
                    col_idx += 1
                    continue
                fit_window_option = ch1_window
                kind_result = label_lookup[fit_window_option].corr_results[kind]
                kind_error = label_lookup[fit_window_option].corr_errors.get(kind)
                st.caption(f"Using window: {fit_window_option}")

            if fit_window_used.get(kind) != fit_window_option:
                # The window for this channel changed since the last fit -- drop the stale
                # result rather than show a fit overlaid on data from a different window.
                fit_results.pop(kind, None)

            n_components, triplet, n_starts = fit_settings_widgets(f"sf_{kind}", label)
            if st.button(f"Fit {label}", key=f"sf_fit_btn_{kind}"):
                # Weight each tau point by 1/sigma_G so the optimizer (and the reported
                # reduced chi-squared) actually accounts for per-point uncertainty, instead
                # of treating every point as equally reliable. Points with an unknown/zero
                # sigma (too few sub-blocks reached that tau) get weight 0 -- excluded from
                # the fit rather than assigned a made-up uncertainty.
                if kind_error is not None:
                    weights = np.where(np.isfinite(kind_error) & (kind_error > 0), 1.0 / kind_error, 0.0)
                else:
                    weights = None
                with st.spinner(f"Fitting {label} ({n_starts} starts)..."):
                    report = multi_start_fit(
                        kind_result.tau, kind_result.g, n_components=n_components, triplet=triplet,
                        n_starts=n_starts, kappa=kappa, weights=weights,
                    )
                fit_results[kind] = report
                fit_window_used[kind] = fit_window_option
            if kind in fit_results:
                report = fit_results[kind]
                fr = report.chosen_result
                badge = "STABLE" if report.is_stable else "UNSTABLE -- fit is sensitive to starting guess"
                (st.success if report.is_stable else st.error)(
                    f"{badge} ({report.converged_fraction:.0%} of {n_starts} starts converged)"
                )
                st.plotly_chart(
                    plot_fit_overlay(kind_result.tau, kind_result.g, fr.fit_curve, title=label, color=SERIES_COLORS.get(kind)),
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
        st.subheader("4. FCCS bound fraction")
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
