"""
core/autowindow.py — automatic time-window selection for a raw trace.

Rather than asking the user to eyeball and drag a time-window trim, the user
picks a window length (e.g. "3 s"), and this slides a window of that length
across the whole acquisition in 1-second steps (0-3, 1-4, 2-5, 3-6, ... to
the end of the trace), correlates every one, and keeps whichever gives the
cleanest correlation curve. "Cleanest" is defined as the highest
signal-to-noise ratio: the median of |G(tau)| / standard-error(tau) across
tau, using the same sub-block error estimate as compute_correlation_error.
Because every candidate is the same length, the score is directly comparable
across all of them.
"""

from dataclasses import dataclass, field

import numpy as np

from .binning import apply_point_binning
from .correlate import compute_all_correlations, compute_correlation_error
from .engine import select_window

STEP_S = 1.0


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


def sliding_window_starts(t_min, t_max, window_length_s, step_s=STEP_S):
    """t_min, t_min+step, t_min+2*step, ... for as long as [start, start+window_length_s]
    still fully fits inside [t_min, t_max], so every candidate is the same length
    and directly comparable. If window_length_s doesn't fit at all, falls back to
    a single candidate spanning the whole trace."""
    span = t_max - t_min
    if window_length_s >= span:
        return [t_min]
    starts = []
    start = t_min
    while start + window_length_s <= t_max:
        starts.append(start)
        start += step_s
    return starts


def search_best_window(
    time_arr, channels, window_length_s, step_s=STEP_S,
    segments=5, points_per_segment=15, base=4, bin_points=1, n_blocks=5,
):
    """Slide a window_length_s-long window across the trace in step_s steps,
    correlate every one, and return the one with the highest correlation-curve
    SNR.

    Returns AutoWindowResult(chosen, candidates) -- candidates includes ones
    that failed to score (e.g. too short to split into n_blocks sub-blocks),
    marked with failed=True and score=-inf, so the caller can show why they
    were skipped.
    """
    t_min, t_max = float(time_arr[0]), float(time_arr[-1])
    starts = sliding_window_starts(t_min, t_max, window_length_s, step_s)

    candidates = []
    for c_t0 in starts:
        c_t1 = min(c_t0 + window_length_s, t_max)
        label = f"{c_t0:.3g}-{c_t1:.3g}s"
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
