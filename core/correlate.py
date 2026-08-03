"""
core/correlate.py — orchestrates one-pass multi-tau correlation for single- or
dual-channel traces, on top of the validated engine.py functions.

Pipeline order: load_trace_auto -> select_window -> apply_point_binning ->
compute_all_correlations.
"""

from dataclasses import dataclass

import numpy as np

from . import engine


@dataclass
class CorrelationResult:
    tau: np.ndarray
    g: np.ndarray
    n_samples: np.ndarray
    kind: str  # "acf_ch1" | "acf_ch2" | "cross"
    dt: float
    segments: int
    points_per_segment: int
    base: int


def compute_all_correlations(channels, dt, segments=5, points_per_segment=15, base=4):
    """channels: dict with keys "CH1" and optionally "CH2" (already windowed/binned).

    Dual-channel input yields ACF(CH1), ACF(CH2), and the symmetrized
    cross-correlation, all computed from the same arrays in one call.
    Single-channel input yields only ACF(CH1).
    """
    kwargs = dict(segments=segments, points_per_segment=points_per_segment, base=base)
    results = {}

    ch1 = channels["CH1"]
    tau1, g1, n1 = engine.multiple_tau_correlate(ch1, ch1, dt, **kwargs)
    results["acf_ch1"] = CorrelationResult(tau1, g1, n1, "acf_ch1", dt, segments, points_per_segment, base)

    if "CH2" in channels:
        ch2 = channels["CH2"]
        tau2, g2, n2 = engine.multiple_tau_correlate(ch2, ch2, dt, **kwargs)
        results["acf_ch2"] = CorrelationResult(tau2, g2, n2, "acf_ch2", dt, segments, points_per_segment, base)

        taux, gx, nx = engine.symmetrized_cross_correlate(ch1, ch2, dt, **kwargs)
        results["cross"] = CorrelationResult(taux, gx, nx, "cross", dt, segments, points_per_segment, base)

    return results
