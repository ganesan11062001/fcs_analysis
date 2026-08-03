"""
core/io.py — single-pass, auto-detecting loader for VistaVision-exported trace files.

engine.load_trace() is the validated reference parser but requires the caller to
already know the file's channel layout and re-reads the whole file once per channel
column requested. For multi-million-row files that's slow (two full passes for a
dual-channel file) and pushes the single/dual-channel decision onto the caller.

load_trace_auto() fixes both: it auto-detects the column layout from the file itself,
parses it once with a vectorized pandas fast path, and falls back wholesale to
engine.load_trace() (called once per channel) if anything about the fast path looks
untrustworthy -- never a partial/patched hybrid, so behavior stays identical to the
validated parser whenever the fast path is bypassed.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import engine


class FormatDetectionError(ValueError):
    """Raised when the channel layout of a trace file can't be determined."""


@dataclass
class TraceData:
    time: np.ndarray
    channels: dict  # {"CH1": np.ndarray} or {"CH1": np.ndarray, "CH2": np.ndarray}
    dt: float
    n_channels: int
    source_path: str
    used_fallback_parser: bool = False
    fallback_reason: str = ""
    n_rows: int = field(default=0)

    def __post_init__(self):
        self.n_rows = len(self.time)


def _iter_probe_lines(path, n_probe_lines):
    lines = []
    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            lines.append(line)
            if len(lines) >= n_probe_lines:
                break
    return lines


def detect_channel_count(path, n_probe_lines=5):
    """Peek the first few non-blank data lines to determine channel layout.

    Returns 1 (single-channel: time, count) or 2 (dual "AllInOne": time, CH1, CH2).
    Raises FormatDetectionError if the file looks like the legacy quoted format
    (no reliable field count) or if lines disagree on field count.
    """
    lines = _iter_probe_lines(path, n_probe_lines)
    if not lines:
        raise FormatDetectionError(f"{path}: file is empty or has no data lines.")

    field_counts = set()
    for line in lines:
        if '"' in line:
            raise FormatDetectionError(
                f"{path}: line uses quoted legacy format, cannot auto-detect field count "
                "from it; use the fallback parser."
            )
        n_fields = len([p for p in line.split(",") if p != ""]) or len(line.split(","))
        field_counts.add(len(line.split(",")))

    if len(field_counts) != 1:
        raise FormatDetectionError(
            f"{path}: inconsistent field counts across probed lines ({sorted(field_counts)}); "
            "cannot reliably auto-detect single- vs dual-channel format."
        )

    n_fields = field_counts.pop()
    if n_fields == 2:
        return 1
    if n_fields == 3:
        return 2
    raise FormatDetectionError(
        f"{path}: unexpected field count {n_fields} per line (expected 2 or 3)."
    )


def _fast_parse(path, n_channels):
    ncols = n_channels + 1
    names = ["time"] + [f"CH{i}" for i in range(1, n_channels + 1)]
    df = pd.read_csv(
        path,
        header=None,
        names=names,
        sep=",",
        engine="c",
        skipinitialspace=True,
        dtype=np.float64,
    )
    if df.shape[1] != ncols:
        raise ValueError(f"expected {ncols} columns, got {df.shape[1]}")
    if df.isna().any().any():
        raise ValueError("NaNs present after fast parse")
    time = df["time"].to_numpy(dtype=np.float64)
    channels = {f"CH{i}": df[f"CH{i}"].to_numpy(dtype=np.float64) for i in range(1, n_channels + 1)}
    return time, channels


def _fallback_parse(path, n_channels):
    time = None
    channels = {}
    for i in range(1, n_channels + 1):
        t, c = engine.load_trace(path, channel_col=i)
        if time is None:
            time = t
        channels[f"CH{i}"] = c
    return time, channels


def load_trace_auto(path):
    """Load a VistaVision trace file, auto-detecting single- vs dual-channel format.

    Tries a vectorized fast path first; falls back wholesale to the validated
    engine.load_trace() (once per channel) if the fast path can't be trusted,
    guaranteeing identical output to the reference parser whenever that happens.
    """
    used_fallback = False
    fallback_reason = ""

    try:
        n_channels = detect_channel_count(str(path))
    except FormatDetectionError as exc:
        # Legacy/ambiguous format: fall back and let engine.load_trace's own
        # quoted-format regex path (or whitespace-split fallback) handle it.
        # We don't know the channel count in this case; assume single-channel,
        # which is what the legacy quoted format was designed for.
        used_fallback = True
        fallback_reason = str(exc)
        n_channels = 1

    if not used_fallback:
        try:
            time, channels = _fast_parse(path, n_channels)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any fast-path issue triggers fallback
            used_fallback = True
            fallback_reason = f"fast parser failed ({exc}); used validated line-by-line parser instead."
            time, channels = _fallback_parse(path, n_channels)
    else:
        time, channels = _fallback_parse(path, n_channels)

    dt = engine.infer_bin_width(time)

    return TraceData(
        time=time,
        channels=channels,
        dt=dt,
        n_channels=n_channels,
        source_path=str(path),
        used_fallback_parser=used_fallback,
        fallback_reason=fallback_reason,
    )
