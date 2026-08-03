"""
core/validation.py — independent correctness checks for the multi-tau engine.

Two distinct tools, both explicitly NOT for production analysis:

1. fft_direct_correlate(): an exact, FFT-accelerated brute-force correlation at
   EVERY integer lag (no coarsening). This is a different algorithm from
   engine.multiple_tau_correlate -- used only to cross-check that the multi-tau
   engine's raw-resolution (segment 0, i.e. segments=1) output matches it exactly.

2. generate_synthetic_fcs_trace(): builds a synthetic photon-count trace with a
   known input diffusion time, so the full correlate+fit pipeline's recovery of
   that known tauD can be sanity-checked after any change to the codebase.
"""

from dataclasses import dataclass

import numpy as np
from scipy.fft import next_fast_len

from . import engine
from .models import diffusion_term


def fft_direct_correlate(a, b, max_lag):
    """Exact linear cross-correlation of a and b at every lag 1..max_lag, computed
    via FFT (zero-padded to avoid circular wraparound), normalized identically to
    engine.multiple_tau_correlate: g[lag] = mean(a[:-lag]*b[lag:]) / (mean(a)*mean(b)) - 1.

    A distinct algorithm from the multi-tau engine (no coarsening at all) -- used
    only for validation, not production analysis.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    if len(b) != n:
        raise ValueError("a and b must have the same length.")
    if max_lag >= n:
        raise ValueError("max_lag must be smaller than the trace length.")

    mean_a = a.mean()
    mean_b = b.mean()

    # Zero-pad well past n to guarantee no circular wraparound contaminates the
    # lags we care about (1..max_lag).
    m = next_fast_len(2 * n)
    A = np.fft.rfft(a, m)
    B = np.fft.rfft(b, m)
    full_corr = np.fft.irfft(np.conj(A) * B, m)  # full_corr[k] = sum_i a[i]*b[i+k], k=0..m-1 (linear, unwrapped)

    lags = np.arange(1, max_lag + 1)
    raw_sum = full_corr[lags]
    overlap = n - lags  # number of terms in cur_a[:-lag]*cur_b[lag:] at raw resolution
    mean_term = raw_sum / overlap
    g = mean_term / (mean_a * mean_b) - 1.0
    return g


def compare_multitau_to_fft(trace_a, trace_b, dt, points_per_segment=15, rtol=1e-8, atol=1e-12):
    """Runs the multi-tau engine with segments=1 (raw resolution only, no
    coarsening) and compares its output lag-by-lag against fft_direct_correlate.

    Returns (max_abs_diff, max_rel_diff, all_close).
    """
    tau_mt, g_mt, _ = engine.multiple_tau_correlate(
        trace_a, trace_b, dt, segments=1, points_per_segment=points_per_segment, base=4
    )
    g_fft = fft_direct_correlate(trace_a, trace_b, max_lag=len(g_mt))

    abs_diff = np.abs(g_mt - g_fft)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.abs(abs_diff / g_fft)
    rel_diff = rel_diff[np.isfinite(rel_diff)]

    max_abs_diff = float(np.max(abs_diff)) if len(abs_diff) else float("nan")
    max_rel_diff = float(np.max(rel_diff)) if len(rel_diff) else float("nan")
    all_close = bool(np.allclose(g_mt, g_fft, rtol=rtol, atol=atol))
    return max_abs_diff, max_rel_diff, all_close


@dataclass
class SyntheticFCSTrace:
    time: np.ndarray
    counts: np.ndarray
    tauD_true: float
    N_true: float
    kappa: float
    dt: float


def generate_synthetic_fcs_trace(tauD, dt, duration_s, N_particles, kappa=5.0, mean_count_rate=50.0, seed=None):
    """Build a synthetic single-channel photon-count trace whose autocorrelation
    matches the standard single-component 3D-diffusion FCS shape with the given
    tauD and N_particles, by construction (Wiener-Khinchin spectral synthesis),
    then adds Poisson shot noise on top.

    Returns a SyntheticFCSTrace (time, counts shaped like load_trace_auto's output,
    plus the known ground-truth parameters for comparison against a later fit).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s / dt))
    if n < 16:
        raise ValueError("duration_s/dt is too short to synthesize a meaningful trace.")

    lags = np.arange(n) * dt
    target_acf = (1.0 / N_particles) * diffusion_term(lags, tauD, kappa)

    # Build a symmetric (even, real) autocovariance array of length m=2n so its FFT
    # (the power spectral density, Wiener-Khinchin) is real and non-negative.
    m = 2 * n
    r_sym = np.zeros(m)
    r_sym[:n] = target_acf
    r_sym[m - n + 1:] = target_acf[1:][::-1]

    psd = np.fft.rfft(r_sym).real
    psd = np.clip(psd, 0.0, None)

    half_len = m // 2 + 1
    white = rng.standard_normal(half_len) + 1j * rng.standard_normal(half_len)
    colored_spectrum = white * np.sqrt(psd)
    # Keep DC and (if m even) Nyquist bins real, as required for a real irfft output.
    colored_spectrum[0] = colored_spectrum[0].real
    if m % 2 == 0:
        colored_spectrum[-1] = colored_spectrum[-1].real

    fluctuation = np.fft.irfft(colored_spectrum, m)[:n]
    # Rescale so the realized fluctuation's variance matches the target ACF at
    # lag 0 exactly (a single finite-sample realization won't match it perfectly).
    empirical_var = fluctuation.var()
    if empirical_var > 0:
        fluctuation *= np.sqrt(target_acf[0] / empirical_var)

    intensity = mean_count_rate * (1.0 + fluctuation)
    intensity = np.clip(intensity, 0.0, None)
    counts = rng.poisson(intensity).astype(np.float64)
    time = np.arange(n) * dt

    return SyntheticFCSTrace(time=time, counts=counts, tauD_true=tauD, N_true=N_particles, kappa=kappa, dt=dt)
