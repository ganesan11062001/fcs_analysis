"""
core/autowindow.py — automatic time-window selection for a raw trace.

Rather than asking the user to eyeball and drag a time-window trim, this tries
a small, fixed set of candidate windows across the acquisition (the full
trace, plus the first/middle/second half) and keeps whichever gives the
cleanest correlation curve. "Cleanest" is defined as the highest
signal-to-noise ratio: the median of |G(tau)| / standard-error(tau) across
tau, using the same sub-block error estimate as compute_correlation_error.

This is a simple, fixed grid search, not an exhaustive one -- it exists to
remove manual window-trimming for the common case (a short bad stretch at
the very start or end of acquisition), not to solve arbitrary trace-quality
problems.
"""

from dataclasses import dataclass, field

import numpy as np

from .binning import apply_point_binning
from .correlate import compute_all_correlations, compute_correlation_error
from .engine import select_window


@dataclass
class WindowCandidate:
    label: str
    t0: float
    t1: float
    score: float
    corr_results: dict
    corr_errors: dict
    eff_dt: float = float("nan")
    mean_rates: list = field(default_factory=list)
    failed: bool = False


@dataclass
class AutoWindowResult:
    chosen: WindowCandidate
    candidates: list = field(default_factory=list)


def _snr_score(corr_results, corr_errors):
    """Median |G(tau)/error(tau)| across finite points, averaged over whichever
    of acf_ch1/acf_ch2/cross curves are present. Higher = cleaner curve."""
    per_curve_scores = []
    for kind, result in corr_results.items():
        err = corr_errors.get(kind)
        if err is None or not len(err):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(result.g) / err
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio):
            per_curve_scores.append(float(np.median(ratio)))
    return float(np.mean(per_curve_scores)) if per_curve_scores else float("-inf")


def search_best_window(
    time_arr, channels, segments=5, points_per_segment=15, base=4, bin_points=1, n_blocks=5
):
    """Evaluate 4 candidate windows (full / first half / middle half / second half)
    and return the one with the highest correlation-curve SNR.

    Returns AutoWindowResult(chosen, candidates) -- candidates includes ones that
    failed to score (e.g. too short to split into n_blocks sub-blocks), marked
    with failed=True and score=-inf, so the caller can show why they were skipped.
    """
    t_min, t_max = float(time_arr[0]), float(time_arr[-1])
    span = t_max - t_min

    candidate_spec = [
        ("full trace", t_min, t_max),
        ("first half", t_min, t_min + 0.5 * span),
        ("middle half", t_min + 0.25 * span, t_min + 0.75 * span),
        ("second half", t_min + 0.5 * span, t_max),
    ]

    candidates = []
    for label, c_t0, c_t1 in candidate_spec:
        try:
            w_time = None
            w_channels = {}
            for name, arr in channels.items():
                wt, wa = select_window(time_arr, arr, c_t0, c_t1)
                w_time = wt
                w_channels[name] = wa
            _, b_channels, eff_dt, _ = apply_point_binning(w_time, w_channels, bin_points)
            corr = compute_all_correlations(
                b_channels, eff_dt, segments=segments, points_per_segment=points_per_segment, base=base
            )
            err = compute_correlation_error(
                corr, b_channels, eff_dt,
                segments=segments, points_per_segment=points_per_segment, base=base, n_blocks=n_blocks,
            )
            score = _snr_score(corr, err)
            mean_rates = [float(np.mean(arr)) / eff_dt for arr in b_channels.values()]
            candidates.append(WindowCandidate(label, c_t0, c_t1, score, corr, err, eff_dt, mean_rates))
        except ValueError:
            candidates.append(WindowCandidate(label, c_t0, c_t1, float("-inf"), {}, {}, failed=True))

    scored = [c for c in candidates if not c.failed]
    chosen = max(scored, key=lambda c: c.score) if scored else candidates[0]
    return AutoWindowResult(chosen, candidates)
