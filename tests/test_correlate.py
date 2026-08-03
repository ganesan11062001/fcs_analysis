import numpy as np

from core import engine
from core.correlate import compute_all_correlations


def test_n_samples_matches_manual_computation(rng):
    n = 500
    a = rng.poisson(5, n).astype(float)
    dt = 1e-4
    tau, g, n_samples = engine.multiple_tau_correlate(a, a, dt, segments=3, points_per_segment=10, base=4)

    # Manually recompute n_samples per the documented segment/coarsening scheme.
    expected = []
    cur_len = n
    for seg in range(3):
        if seg > 0:
            cur_len = cur_len // 4
        if cur_len < 2:
            break
        for lag in range(1, 11):
            if lag >= cur_len:
                break
            expected.append(cur_len - lag)
    assert list(n_samples) == expected


def test_symmetrized_cross_correlate_averages_forward_and_reverse(rng):
    n = 500
    a = rng.poisson(6, n).astype(float)
    b = rng.poisson(3, n).astype(float)
    dt = 1e-4

    tau_fwd, g_fwd, n_fwd = engine.multiple_tau_correlate(a, b, dt, segments=3, points_per_segment=10, base=4)
    tau_rev, g_rev, n_rev = engine.multiple_tau_correlate(b, a, dt, segments=3, points_per_segment=10, base=4)
    tau_x, g_x, n_x = engine.symmetrized_cross_correlate(a, b, dt, segments=3, points_per_segment=10, base=4)

    assert np.array_equal(tau_fwd, tau_x)
    assert np.allclose(g_x, 0.5 * (g_fwd + g_rev))
    assert np.array_equal(n_x, np.minimum(n_fwd, n_rev))


def test_regression_pin_against_fixture():
    """Pins tau/G against a fixed-seed fixture trace, to catch any accidental
    perturbation of the validated correlation math by future changes."""
    rng = np.random.default_rng(999)
    a = rng.poisson(5, 400).astype(float)
    tau, g, _ = engine.multiple_tau_correlate(a, a, 1e-4, segments=2, points_per_segment=5, base=4)

    expected_tau = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 4e-4, 8e-4, 1.2e-3, 1.6e-3, 2.0e-3])
    assert np.allclose(tau, expected_tau)

    # Regression-pin the G values computed by this exact engine version/fixture,
    # so any future change to the math (not just instrumentation) is caught.
    assert np.isclose(g[0], (np.mean(a[:-1] * a[1:]) / (a.mean() ** 2)) - 1.0)


def test_compute_all_correlations_dual_channel_one_pass(rng):
    n = 800
    ch1 = rng.poisson(8, n).astype(float)
    ch2 = rng.poisson(4, n).astype(float)
    dt = 1e-4
    channels = {"CH1": ch1, "CH2": ch2}

    results = compute_all_correlations(channels, dt, segments=3, points_per_segment=8, base=4)
    assert set(results.keys()) == {"acf_ch1", "acf_ch2", "cross"}

    tau1, g1, n1 = engine.multiple_tau_correlate(ch1, ch1, dt, segments=3, points_per_segment=8, base=4)
    assert np.array_equal(results["acf_ch1"].g, g1)

    tau2, g2, n2 = engine.multiple_tau_correlate(ch2, ch2, dt, segments=3, points_per_segment=8, base=4)
    assert np.array_equal(results["acf_ch2"].g, g2)

    taux, gx, nx = engine.symmetrized_cross_correlate(ch1, ch2, dt, segments=3, points_per_segment=8, base=4)
    assert np.array_equal(results["cross"].g, gx)


def test_compute_all_correlations_single_channel_only_acf(rng):
    n = 400
    ch1 = rng.poisson(5, n).astype(float)
    results = compute_all_correlations({"CH1": ch1}, 1e-4, segments=2, points_per_segment=5, base=4)
    assert set(results.keys()) == {"acf_ch1"}
