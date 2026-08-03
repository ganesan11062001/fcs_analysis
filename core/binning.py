"""
core/binning.py — the "Binning points" control (VistaVision's own term).

This groups N raw points together BEFORE correlating -- a pre-correlation grouping
of the raw trace, distinct from the multi-tau engine's own internal per-segment
coarsening (segments / points_per_segment / base in engine.multiple_tau_correlate).
Both exist in the UI under their own labels so they aren't confused with each other.
"""

import numpy as np

from .engine import _coarsen


def apply_point_binning(time_arr, channels, bin_points):
    """Group `bin_points` consecutive raw samples together (non-overlapping blocks,
    mean-averaged, remainder tail dropped -- same convention as engine._coarsen).

    bin_points=1 is a passthrough (no-op).

    Returns (binned_time, binned_channels, effective_dt, effective_rate_hz).
    """
    bin_points = int(bin_points)
    if bin_points < 1:
        raise ValueError("bin_points must be >= 1.")

    raw_dt = float(np.median(np.diff(time_arr)))
    effective_dt = raw_dt * bin_points

    if bin_points == 1:
        return time_arr, dict(channels), effective_dt, 1.0 / effective_dt

    binned_time = _coarsen(time_arr, bin_points)
    binned_channels = {name: _coarsen(arr, bin_points) for name, arr in channels.items()}
    return binned_time, binned_channels, effective_dt, 1.0 / effective_dt
