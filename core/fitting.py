"""
core/fitting.py — fit standard 3D-diffusion FCS models to G(tau) curves, with a
multi-start stability check that FLAGS (rather than silently accepts) divergent fits.

Scope note (per project decisions): no observation-volume calibration (w_xy/kappa)
exists yet, so results are reported as tau_D (diffusion time, seconds) and fitted
amplitude/N directly -- not converted to an absolute diffusion coefficient or
concentration. FitResult.calibration is an explicit, currently-unset slot so a future
version can add that conversion without changing this module's output schema.
"""

from dataclasses import dataclass, field

import numpy as np
import lmfit

from .models import fcs_1comp, fcs_2comp

STABILITY_REL_SPREAD_THRESHOLD = 0.20
STABILITY_MIN_CONVERGED_FRACTION = 0.6


@dataclass
class CalibrationConstants:
    """Explicit future slot for tau_D -> D and G(0) -> concentration conversion.
    Intentionally unset in v1 -- no w_xy/kappa calibration workflow exists yet."""

    wxy_um: float = None
    kappa_calibrated: float = None


@dataclass
class FitResult:
    n_components: int
    triplet: bool
    params: dict
    params_stderr: dict
    tauD: list
    N: float
    fit_curve: np.ndarray
    redchi: float
    success: bool
    message: str = ""
    calibration: CalibrationConstants = None
    stability: "StabilityReport" = None


@dataclass
class StabilityReport:
    is_stable: bool
    relative_spreads: dict
    converged_fraction: float
    per_start_results: list
    chosen_result: FitResult


def build_model(n_components=1, triplet=False):
    """Returns (lmfit.Model, lmfit.Parameters) with bounds/constraints set, but no
    data-informed initial values yet -- fit_curve()/multi_start_fit() set those."""
    if n_components == 1:
        model = lmfit.Model(fcs_1comp, independent_vars=["tau"])
        params = model.make_params()
        params["N"].set(value=1.0, min=1e-6)
        params["tauD"].set(value=1e-4, min=1e-9)
    elif n_components == 2:
        model = lmfit.Model(fcs_2comp, independent_vars=["tau"])
        params = model.make_params()
        params["N"].set(value=1.0, min=1e-6)
        params["f1"].set(value=0.5, min=0.0, max=1.0)
        params["tauD1"].set(value=1e-5, min=1e-9)
        # tauD2 = tauD1 + delta (delta >= 0) enforces tauD2 >= tauD1, avoiding
        # component label-swap ambiguity between the two species.
        params.add("tauD2_delta", value=1e-3, min=0.0)
        params.add("tauD2", expr="tauD1 + tauD2_delta")
    else:
        raise ValueError("n_components must be 1 or 2.")

    params["kappa"].set(value=5.0, vary=False)
    if triplet:
        params["T"].set(value=0.1, min=0.0, max=0.9, vary=True)
        params["tau_trip"].set(value=1e-6, min=1e-9, vary=True)
    else:
        params["T"].set(value=0.0, vary=False)
        params["tau_trip"].set(value=1e-6, vary=False)
    return model, params


def _guess_amplitude_and_tauD(tau, g):
    """Data-driven initial guess: amplitude from the earliest (smallest-tau) points,
    tauD from where the curve has decayed to half that amplitude."""
    tau = np.asarray(tau, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    order = np.argsort(tau)
    tau_s, g_s = tau[order], g[order]
    n_head = max(1, len(g_s) // 20)
    g0 = float(np.mean(g_s[:n_head]))
    if g0 <= 1e-9:
        g0 = 1e-9
    half = g0 / 2.0
    idx = int(np.argmin(np.abs(g_s - half)))
    tauD_guess = tau_s[idx] if tau_s[idx] > 0 else float(np.median(tau_s[tau_s > 0]))
    N_guess = 1.0 / g0
    return N_guess, tauD_guess


def fit_curve(tau, g, n_components=1, triplet=False, initial_guess=None, weights=None, kappa=5.0):
    """Fit one G(tau) curve. initial_guess is an optional dict of param_name->value
    overrides (used by multi_start_fit to probe different starting points)."""
    tau = np.asarray(tau, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    model, params = build_model(n_components, triplet)
    params["kappa"].set(value=kappa)

    N_guess, tauD_guess = _guess_amplitude_and_tauD(tau, g)
    if n_components == 1:
        params["N"].set(value=N_guess)
        params["tauD"].set(value=tauD_guess)
    else:
        params["N"].set(value=N_guess)
        params["tauD1"].set(value=tauD_guess / 3.0)
        params["tauD2_delta"].set(value=tauD_guess * 3.0)

    if initial_guess:
        for key, value in initial_guess.items():
            if key in params:
                params[key].set(value=value)

    try:
        result = model.fit(g, params, tau=tau, weights=weights)
        success = bool(result.success) and np.isfinite(result.redchi)
        message = result.message or ""
    except Exception as exc:  # noqa: BLE001 - a failed fit is a valid, reportable outcome
        return FitResult(
            n_components=n_components,
            triplet=triplet,
            params={},
            params_stderr={},
            tauD=[],
            N=float("nan"),
            fit_curve=np.full_like(g, np.nan),
            redchi=float("inf"),
            success=False,
            message=f"fit raised: {exc}",
        )

    out_params = {name: p.value for name, p in result.params.items()}
    out_stderr = {name: (p.stderr if p.stderr is not None else float("nan")) for name, p in result.params.items()}
    tauD_list = [out_params["tauD"]] if n_components == 1 else [out_params["tauD1"], out_params["tauD2"]]

    return FitResult(
        n_components=n_components,
        triplet=triplet,
        params=out_params,
        params_stderr=out_stderr,
        tauD=tauD_list,
        N=out_params["N"],
        fit_curve=result.best_fit,
        redchi=float(result.redchi),
        success=success,
        message=message,
    )


def multi_start_fit(tau, g, n_components=1, triplet=False, n_starts=5, kappa=5.0, weights=None):
    """Run the same fit from n_starts different initial tauD guesses (log-spaced
    across the data's own tau range) and flag divergence rather than silently
    trusting whichever result the optimizer returns first.

    Unstable if: any parameter's (max-min)/median across CONVERGED starts exceeds
    STABILITY_REL_SPREAD_THRESHOLD, or fewer than STABILITY_MIN_CONVERGED_FRACTION
    of starts converge.
    """
    tau = np.asarray(tau, dtype=np.float64)
    tau_pos = tau[tau > 0]
    if len(tau_pos) == 0:
        raise ValueError("No positive tau values to fit.")

    lo = 3.0 * np.min(tau_pos)
    hi = np.max(tau_pos) / 3.0
    if hi <= lo:
        hi = lo * 10.0
    start_tauDs = np.exp(np.linspace(np.log(lo), np.log(hi), n_starts))

    per_start_results = []
    for s in start_tauDs:
        if n_components == 1:
            ig = {"tauD": s}
        else:
            ig = {"tauD1": s / 3.0, "tauD2_delta": s * 3.0}
        fr = fit_curve(tau, g, n_components=n_components, triplet=triplet, initial_guess=ig, weights=weights, kappa=kappa)
        per_start_results.append(fr)

    converged = [r for r in per_start_results if r.success]
    converged_fraction = len(converged) / len(per_start_results)

    param_names = ["N"] + (["tauD"] if n_components == 1 else ["tauD1", "tauD2"])
    relative_spreads = {}
    for pname in param_names:
        vals = [r.params[pname] for r in converged if pname in r.params]
        if len(vals) >= 2:
            med = np.median(vals)
            relative_spreads[pname] = float((max(vals) - min(vals)) / med) if med != 0 else float("inf")
        else:
            relative_spreads[pname] = float("nan")

    is_stable = converged_fraction >= STABILITY_MIN_CONVERGED_FRACTION and all(
        spread <= STABILITY_REL_SPREAD_THRESHOLD for spread in relative_spreads.values() if not np.isnan(spread)
    )

    chosen = min(converged, key=lambda r: r.redchi) if converged else per_start_results[0]

    report = StabilityReport(
        is_stable=is_stable,
        relative_spreads=relative_spreads,
        converged_fraction=converged_fraction,
        per_start_results=per_start_results,
        chosen_result=chosen,
    )
    chosen.stability = report
    return report
