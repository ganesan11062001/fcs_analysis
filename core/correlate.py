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


def compute_correlation_error(full_results, channels, dt, segments=5, points_per_segment=15, base=4, n_blocks=10):
    """Standard sub-block standard-error-of-the-mean estimate for each G(tau) point --
    the same block-averaging convention used broadly across FCS correlator software
    (ALV, correlator.com-style hardware correlators, SymPhoTime): split the trace into
    n_blocks contiguous equal-length sub-traces, correlate each independently with the
    SAME settings, and take the spread across sub-blocks at each tau as its uncertainty.

    NOT a reproduction of any particular instrument's internal error algorithm (e.g.
    VistaVision's own error column, which is undocumented) -- this is a standard,
    independently-defined statistical estimate, so don't expect it to match another
    tool's error values exactly, only to be a valid uncertainty by the same convention.

    Sub-blocks are prefix-aligned with the full-trace tau grid: a shorter sub-block's
    multi-tau loop simply terminates earlier (same dt/segments/points_per_segment/base),
    so results are matched to the full-trace result by position, not by tau value.

    Returns dict[kind] -> np.ndarray of standard errors, same length as
    full_results[kind].tau, with NaN wherever fewer than 2 sub-blocks reached that tau.
    """
    n = len(next(iter(channels.values())))
    block_len = n // n_blocks
    if block_len < 2:
        raise ValueError(f"Trace too short to split into {n_blocks} blocks of >=2 points each.")

    block_results = []
    for b in range(n_blocks):
        sl = slice(b * block_len, (b + 1) * block_len)
        block_channels = {name: arr[sl] for name, arr in channels.items()}
        block_results.append(compute_all_correlations(block_channels, dt, segments=segments, points_per_segment=points_per_segment, base=base))

    errors = {}
    for kind, result in full_results.items():
        n_tau = len(result.tau)
        err = np.full(n_tau, np.nan)
        for i in range(n_tau):
            vals = [br[kind].g[i] for br in block_results if kind in br and len(br[kind].g) > i]
            if len(vals) >= 2:
                err[i] = np.std(vals, ddof=1) / np.sqrt(len(vals))
        errors[kind] = err
    return errors
