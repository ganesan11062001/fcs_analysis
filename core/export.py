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
