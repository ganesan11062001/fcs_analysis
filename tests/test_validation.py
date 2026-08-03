import numpy as np

from core import engine
from core.validation import fft_direct_correlate, compare_multitau_to_fft
from core.fitting import fit_curve


def test_fft_direct_correlate_matches_multitau_segment0(rng):
    n = 5000
    a = rng.poisson(5, n).astype(float)
    b = rng.poisson(3, n).astype(float)
    dt = 1e-4

    max_abs_diff, max_rel_diff, all_close = compare_multitau_to_fft(a, a, dt, points_per_segment=15)
    assert all_close, f"autocorrelation mismatch: max_abs={max_abs_diff}, max_rel={max_rel_diff}"

    max_abs_diff, max_rel_diff, all_close = compare_multitau_to_fft(a, b, dt, points_per_segment=15)
    assert all_close, f"cross-correlation mismatch: max_abs={max_abs_diff}, max_rel={max_rel_diff}"


def test_fft_direct_correlate_matches_manual_brute_force(rng):
    n = 300
    a = rng.poisson(5, n).astype(float)
    b = rng.poisson(3, n).astype(float)

    g_fft = fft_direct_correlate(a, b, max_lag=10)
    mean_a, mean_b = a.mean(), b.mean()
    g_manual = np.array(
        [np.mean(a[:-lag] * b[lag:]) / (mean_a * mean_b) - 1.0 for lag in range(1, 11)]
    )
    assert np.allclose(g_fft, g_manual, rtol=1e-10, atol=1e-12)


def test_synthetic_trace_is_physically_sane(known_tauD_synthetic):
    synth = known_tauD_synthetic
    assert np.all(synth.counts >= 0)
    assert synth.counts.mean() > 0
    # Poisson-like: variance should be roughly on the same order as the mean
    # once the added correlated fluctuation is accounted for (loose sanity check).
    assert synth.counts.var() > 0


def test_synthetic_trace_fit_recovers_known_tauD(known_tauD_synthetic):
    synth = known_tauD_synthetic
    tau, g, _ = engine.multiple_tau_correlate(synth.counts, synth.counts, synth.dt, segments=5, points_per_segment=15, base=4)
    fr = fit_curve(tau, g, n_components=1)
    assert fr.success
    pct_error = abs(fr.tauD[0] - synth.tauD_true) / synth.tauD_true
    assert pct_error < 0.2, f"recovered tauD off by {pct_error:.1%}"
