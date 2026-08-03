#!/usr/bin/env python3
"""
engine.py — VALIDATED reference correlation engine (vendored).

Validation performed (by the research group, prior to this app):
- At raw (uncoarsened) resolution, multiple_tau_correlate()'s output matched an exact
  FFT-based brute-force correlation bit-for-bit (difference of 0.000000). See
  core/validation.py and tests/test_validation.py for the automated re-check of this claim.
- Recovers correct decay behavior on synthetic data with known diffusion time.
- Matches VistaVision manual section 13.4.1's documented scheme: 5 segments, 15 points
  per segment, grouping/coarsening base of 4, symmetrized cross-correlation.

DO NOT MODIFY the correlation math below (the segment loop, coarsening, or the
normalization by the full-trace mean). Only additive instrumentation is permitted —
see the two deltas called out inline as "ADDED:". tests/test_correlate.py pins the
(tau, G) output against a fixture to guard against accidental drift.
"""

import argparse
import re
import sys

import numpy as np

TIME_COUNT_RE = re.compile(r'"\s*([0-9.eE+-]+)"\s*\t\s*(-?[0-9.eE+-]+)')


def load_trace(path, channel_col=1):
    """Parse a VistaVision-style exported time-binned trace file.

    Supports:
      1. Quoted single-channel format:  "\\t  0.000000"\\t0
      2. Plain "AllInOne" combined format (comma-separated, no quotes):
         \\t  0.000000,         0,         0
         (time, CH1 count, CH2 count all in one file/line)

    channel_col: for the AllInOne format, which count column to return
      (1 = CH1, 2 = CH2). Ignored for the quoted single-channel format.
    """
    times = []
    counts = []
    with open(path, "r", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue

            if "," in line and '"' not in line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= channel_col + 1:
                    try:
                        t = float(parts[0])
                        c = float(parts[channel_col])
                        times.append(t)
                        counts.append(c)
                        continue
                    except ValueError:
                        pass
                print(f"Warning: could not parse line {line_no}: {line!r}", file=sys.stderr)
                continue

            m = TIME_COUNT_RE.search(line)
            if not m:
                cleaned = line.replace('"', "")
                parts = [p for p in re.split(r"\s+", cleaned.strip()) if p]
                if len(parts) >= 2:
                    try:
                        t = float(parts[0])
                        c = float(parts[1])
                        times.append(t)
                        counts.append(c)
                        continue
                    except ValueError:
                        pass
                print(f"Warning: could not parse line {line_no}: {line!r}", file=sys.stderr)
                continue
            t, c = m.group(1), m.group(2)
            times.append(float(t))
            counts.append(float(c))

    if not times:
        raise ValueError(f"No data parsed from {path}. Check the file format.")

    return np.array(times), np.array(counts)


def infer_bin_width(time_arr):
    diffs = np.diff(time_arr)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("Could not infer bin width (fewer than 2 valid samples).")
    return float(np.median(diffs))


def select_window(time_arr, count_arr, start, end):
    if start is None and end is None:
        return time_arr, count_arr
    t0 = time_arr[0] if start is None else start
    t1 = time_arr[-1] if end is None else end
    mask = (time_arr >= t0) & (time_arr <= t1)
    if mask.sum() < 2:
        raise ValueError(
            f"Time window [{t0}, {t1}] selects fewer than 2 points; "
            f"data spans [{time_arr[0]}, {time_arr[-1]}]."
        )
    return time_arr[mask], count_arr[mask]


def _coarsen(a, factor):
    n = (len(a) // factor) * factor
    return a[:n].reshape(-1, factor).mean(axis=1)


def multiple_tau_correlate(trace_a, trace_b, dt, segments=5, points_per_segment=15, base=4):
    """Multi-tau correlator matching VistaVision's documented default scheme
    (manual section 13.4.1): 5 segments x 15 points, multi-tau base 4.

    Returns (tau, G, n_samples) where G(tau) = <I(t) I(t+tau)> / (<I_a><I_b>) - 1
    and n_samples[i] is the number of averaged terms behind G[i] (ADDED: for
    long-tau statistical-reliability warnings; does not affect the G(tau) values).
    VALIDATED: at segment 0 (raw resolution), this exactly matches an
    independent FFT-based brute-force correlation (bit-for-bit).
    """
    a = np.asarray(trace_a, dtype=np.float64)
    b = np.asarray(trace_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("CH1 and CH2 traces must have the same length/timestamps.")

    mean_a_full = a.mean()
    mean_b_full = b.mean()
    if mean_a_full == 0 or mean_b_full == 0:
        raise ValueError("One of the channels has zero mean counts in this window; cannot normalize.")

    taus = []
    Gs = []
    n_samples = []  # ADDED: number of averaged terms behind each G(tau) point
    cur_a, cur_b, cur_dt = a, b, dt

    for seg in range(segments):
        if seg > 0:
            cur_a = _coarsen(cur_a, base)
            cur_b = _coarsen(cur_b, base)
            cur_dt *= base
        if len(cur_a) < 2:
            break
        for lag in range(1, points_per_segment + 1):
            if lag >= len(cur_a):
                break
            num = np.mean(cur_a[:-lag] * cur_b[lag:])
            g = num / (mean_a_full * mean_b_full) - 1.0
            taus.append(lag * cur_dt)
            Gs.append(g)
            n_samples.append(len(cur_a) - lag)  # ADDED

    return np.array(taus), np.array(Gs), np.array(n_samples)  # ADDED: 3rd element


def symmetrized_cross_correlate(trace_a, trace_b, dt, segments=5, points_per_segment=15, base=4):
    """VistaVision-style symmetrized cross-correlation: 0.5 * (CH1xCH2 + CH2xCH1)."""
    tau_fwd, g_fwd, n_fwd = multiple_tau_correlate(trace_a, trace_b, dt, segments, points_per_segment, base)
    _, g_rev, n_rev = multiple_tau_correlate(trace_b, trace_a, dt, segments, points_per_segment, base)
    # ADDED: n_samples is identical in both directions by construction (it depends only on
    # array length/lag, not on which trace is "a" vs "b"); np.minimum is a defensive no-op.
    n_samples = np.minimum(n_fwd, n_rev)
    return tau_fwd, 0.5 * (g_fwd + g_rev), n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch1", required=True)
    ap.add_argument("--ch2", default=None)
    ap.add_argument("--ch1-col", type=int, default=1)
    ap.add_argument("--ch2-col", type=int, default=2)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", default="correlation_output.csv")
    ap.add_argument("--segments", type=int, default=5)
    ap.add_argument("--points-per-segment", type=int, default=15)
    ap.add_argument("--base", type=int, default=4)
    args = ap.parse_args()

    t1, c1 = load_trace(args.ch1, channel_col=args.ch1_col)
    dt1 = infer_bin_width(t1)
    t1w, c1w = select_window(t1, c1, args.start, args.end)

    mt_kwargs = dict(segments=args.segments, points_per_segment=args.points_per_segment, base=args.base)

    if args.ch2:
        t2, c2 = load_trace(args.ch2, channel_col=args.ch2_col)
        t2w, c2w = select_window(t2, c2, args.start, args.end)
        n = min(len(c1w), len(c2w))
        c1w, c2w = c1w[:n], c2w[:n]
        tau, G, n_samples = symmetrized_cross_correlate(c1w, c2w, dt1, **mt_kwargs)  # ADDED: unpack n_samples
    else:
        tau, G, n_samples = multiple_tau_correlate(c1w, c1w, dt1, **mt_kwargs)  # ADDED: unpack n_samples

    with open(args.out, "w") as f:
        f.write("tau_seconds,G_tau,n_samples\n")  # ADDED: n_samples column
        for tt, gg, nn in zip(tau, G, n_samples):
            f.write(f"{tt:.8e},{gg:.8e},{nn:d}\n")


if __name__ == "__main__":
    main()
