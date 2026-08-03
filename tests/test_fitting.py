import numpy as np

from core.models import fcs_1comp, fcs_2comp
from core.fitting import fit_curve, multi_start_fit, FitResult


def test_1component_fit_recovers_noiseless_parameters():
    tau = np.logspace(-6, 0, 200)
    true_N, true_tauD = 2.5, 2e-4
    g = fcs_1comp(tau, true_N, true_tauD, kappa=5.0)

    fr = fit_curve(tau, g, n_components=1)
    assert fr.success
    assert abs(fr.N - true_N) / true_N < 1e-3
    assert abs(fr.tauD[0] - true_tauD) / true_tauD < 1e-3


def test_2component_fit_recovers_noiseless_parameters():
    tau = np.logspace(-6, 0, 200)
    g = fcs_2comp(tau, 3.0, 0.3, 1e-5, 5e-3, kappa=5.0)

    fr = fit_curve(tau, g, n_components=2)
    assert fr.success
    assert abs(fr.tauD[0] - 1e-5) / 1e-5 < 1e-2
    assert abs(fr.tauD[1] - 5e-3) / 5e-3 < 1e-2


def test_fit_result_calibration_defaults_to_none():
    tau = np.logspace(-6, 0, 100)
    g = fcs_1comp(tau, 2.0, 1e-4, kappa=5.0)
    fr = fit_curve(tau, g, n_components=1)
    assert fr.calibration is None


def test_multi_start_fit_stable_on_well_conditioned_curve():
    rng = np.random.default_rng(1)
    tau = np.logspace(-6, 0, 200)
    g = fcs_1comp(tau, 2.5, 2e-4, kappa=5.0)
    g_noisy = g + rng.normal(0, 0.002, size=g.shape)

    report = multi_start_fit(tau, g_noisy, n_components=1, n_starts=5)
    assert report.is_stable
    assert report.converged_fraction >= 0.6
    assert abs(report.chosen_result.tauD[0] - 2e-4) / 2e-4 < 0.2


def test_multi_start_fit_flags_instability_on_overparameterized_model():
    """A 2-component fit on essentially 1-component data is a classic unidentifiable
    (near-degenerate) case -- different starts should disagree wildly on which
    tau is the 'real' component, and this must be flagged, not silently accepted."""
    rng = np.random.default_rng(3)
    tau = np.logspace(-6, 0, 60)
    g = fcs_1comp(tau, 2.0, 1e-4, kappa=5.0)
    g_noisy = g + rng.normal(0, 0.01, size=g.shape)

    report = multi_start_fit(tau, g_noisy, n_components=2, n_starts=8)
    assert not report.is_stable
