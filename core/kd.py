"""
core/kd.py — Kd extraction from a bound-fraction-vs-[NT] concentration series.

Uses the full quadratic 1:1 binding equation (accounts for ligand depletion), not
the simple hyperbolic approximation, since [NT] and [Tau-FL] may be comparable in
magnitude rather than [NT] always being in large excess.
"""

from dataclasses import dataclass

import numpy as np
import lmfit


def quadratic_binding_isotherm(L, Kd, T_total, Fmax=1.0, Fmin=0.0):
    """Fraction of the fixed tracer (Tau-FL, total concentration T_total) that is
    bound, as a function of total titrant concentration L ([NT]), for 1:1 binding:

        f_bound(L) = Fmin + (Fmax-Fmin) *
            ((Kd+T_total+L) - sqrt((Kd+T_total+L)^2 - 4*T_total*L)) / (2*T_total)
    """
    L = np.asarray(L, dtype=np.float64)
    s = Kd + T_total + L
    inside = s**2 - 4.0 * T_total * L
    inside = np.clip(inside, 0.0, None)  # guard tiny negative values from floating-point roundoff
    frac = (s - np.sqrt(inside)) / (2.0 * T_total)
    return Fmin + (Fmax - Fmin) * frac


@dataclass
class KdFitResult:
    Kd: float
    Kd_stderr: float
    T_total: float
    Fmax: float
    Fmin: float
    fit_curve: np.ndarray
    redchi: float
    success: bool
    message: str = ""


def fit_kd(concentrations, bound_fractions, T_total, fit_baseline=False):
    """Fit the quadratic binding isotherm to (concentration, bound_fraction) points.

    T_total (the fixed Tau-FL tracer concentration) is held fixed, not floated.
    Fmax/Fmin are fixed at 1/0 unless fit_baseline=True.
    """
    L = np.asarray(concentrations, dtype=np.float64)
    f = np.asarray(bound_fractions, dtype=np.float64)

    model = lmfit.Model(quadratic_binding_isotherm, independent_vars=["L"])
    params = model.make_params()
    params["Kd"].set(value=float(np.median(L)) if len(L) else 1.0, min=0.0)
    params["T_total"].set(value=T_total, vary=False)
    params["Fmax"].set(value=1.0, vary=fit_baseline)
    params["Fmin"].set(value=0.0, vary=fit_baseline)

    try:
        result = model.fit(f, params, L=L)
        success = bool(result.success) and np.isfinite(result.redchi)
        kd_val = result.params["Kd"].value
        kd_stderr = result.params["Kd"].stderr if result.params["Kd"].stderr is not None else float("nan")
        return KdFitResult(
            Kd=kd_val,
            Kd_stderr=kd_stderr,
            T_total=T_total,
            Fmax=result.params["Fmax"].value,
            Fmin=result.params["Fmin"].value,
            fit_curve=result.best_fit,
            redchi=float(result.redchi),
            success=success,
            message=result.message or "",
        )
    except Exception as exc:  # noqa: BLE001 - a failed Kd fit is a valid, reportable outcome
        return KdFitResult(
            Kd=float("nan"),
            Kd_stderr=float("nan"),
            T_total=T_total,
            Fmax=1.0,
            Fmin=0.0,
            fit_curve=np.full_like(f, np.nan),
            redchi=float("inf"),
            success=False,
            message=f"fit raised: {exc}",
        )
