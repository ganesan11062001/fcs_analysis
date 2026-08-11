"""
core/export.py — turn result dataclasses into pandas DataFrames / CSV text for
download at every stage (raw correlation curves, fit parameters, stability
reports, Kd fits, batch comparisons).
"""

import pandas as pd


def correlation_result_to_df(result):
    return pd.DataFrame({"tau_seconds": result.tau, "G_tau": result.g, "n_samples": result.n_samples})


def correlation_results_to_df(results):
    """results: dict[kind -> CorrelationResult]. One combined long-format DataFrame."""
    frames = []
    for kind, result in results.items():
        df = correlation_result_to_df(result)
        df.insert(0, "kind", kind)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def correlation_results_to_df_with_error(results, errors):
    """Same long format as correlation_results_to_df, plus a per-point standard-error
    column from core.correlate.compute_correlation_error. errors: dict[kind -> ndarray],
    same length/order as each result's tau array."""
    frames = []
    for kind, result in results.items():
        df = correlation_result_to_df(result)
        df.insert(0, "kind", kind)
        df["error"] = errors.get(kind)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def correlation_results_to_vistavision_csv_text(results, errors, sampling_rate_hz, mean_rates):
    """Reproduces VistaVision's exported [HeaderX]/[Data] file structure and its
    tau,G,err column layout (one G+err pair per curve: CH1 autocorrelation, and if
    dual-channel, CH2 autocorrelation and cross-correlation) -- for drop-in
    familiarity with VistaVision's own correlation-export format.

    mean_rates: list of mean count rates, [CH1] or [CH1, CH2], matching VistaVision's
    header line 4.

    Does NOT reproduce VistaVision's segment-boundary duplicate rows (where a coarser
    segment's leading point(s) numerically coincide with the previous segment's
    trailing point(s)) -- that's an undocumented internal quirk of VistaVision's
    correlator, not part of the published multi-tau algorithm (manual section 13.4.1)
    this app's engine is validated against, so it isn't reproduced here.
    """
    any_result = next(iter(results.values()))
    lines = [
        "[HeaderX]",
        str(any_result.segments),
        str(any_result.points_per_segment),
        str(sampling_rate_hz),
        ",".join(f"{r:.10g}" for r in mean_rates),
        "[Data]",
    ]
    kinds_order = [k for k in ("acf_ch1", "acf_ch2", "cross") if k in results]
    tau = results[kinds_order[0]].tau
    for i in range(len(tau)):
        row = [f"{tau[i]:.10g}"]
        for kind in kinds_order:
            row.append(f"{results[kind].g[i]:.10g}")
            err = errors.get(kind)
            row.append(f"{err[i]:.10g}" if err is not None and not pd.isna(err[i]) else "")
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def fit_result_to_df(fit_result, label=""):
    rows = []
    for name, value in fit_result.params.items():
        rows.append(
            {
                "label": label,
                "parameter": name,
                "value": value,
                "stderr": fit_result.params_stderr.get(name, float("nan")),
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["n_components"] = fit_result.n_components
    df.attrs["redchi"] = fit_result.redchi
    df.attrs["success"] = fit_result.success
    return df


def stability_report_to_df(report, label=""):
    rows = []
    for i, r in enumerate(report.per_start_results):
        row = {"label": label, "start_index": i, "success": r.success, "redchi": r.redchi, "N": r.N}
        for j, td in enumerate(r.tauD):
            row[f"tauD{j + 1}"] = td
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["is_stable"] = report.is_stable
    df.attrs["converged_fraction"] = report.converged_fraction
    df.attrs["relative_spreads"] = report.relative_spreads
    return df


def fccs_result_to_df(bound_fraction_tau_fl, bound_fraction_nt, label=""):
    return pd.DataFrame(
        [
            {"label": label, "quantity": "bound_fraction_Tau-FL", "value": bound_fraction_tau_fl},
            {"label": label, "quantity": "bound_fraction_NT_or_BSA", "value": bound_fraction_nt},
        ]
    )


def kd_result_to_df(concentrations, bound_fractions, kd_fit_result):
    df = pd.DataFrame(
        {
            "concentration": concentrations,
            "bound_fraction": bound_fractions,
            "isotherm_fit": kd_fit_result.fit_curve,
        }
    )
    df.attrs["Kd"] = kd_fit_result.Kd
    df.attrs["Kd_stderr"] = kd_fit_result.Kd_stderr
    df.attrs["T_total"] = kd_fit_result.T_total
    return df


def batch_comparison_to_df(rows):
    """rows: list of dicts, one per processed file, e.g.
    {"label": "0hr", "tauD_ch1": ..., "tauD_ch2": ..., "bound_fraction_tau_fl": ...,
     "stable_ch1": True, "stable_ch2": True}. Simple passthrough to a DataFrame,
    kept as a function so the app's row-building convention lives in one place."""
    return pd.DataFrame(rows)


def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")
