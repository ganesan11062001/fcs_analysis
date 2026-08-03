import numpy as np

from core.fccs import bound_fraction_from_cross
from core.kd import quadratic_binding_isotherm, fit_kd


def test_bound_fraction_matches_hand_computed_value():
    g_cross0 = 0.02
    g_partner0 = 0.1
    assert np.isclose(bound_fraction_from_cross(g_cross0, g_partner0), 0.2)


def test_bound_fraction_nan_on_zero_partner_amplitude():
    assert np.isnan(bound_fraction_from_cross(0.02, 0.0))


def test_isotherm_reduces_to_hyperbolic_when_tracer_small():
    Kd_true = 5.0
    T_total = 1e-3
    L = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 50])
    f_quad = quadratic_binding_isotherm(L, Kd_true, T_total)
    f_hyp = L / (Kd_true + L)
    assert np.max(np.abs(f_quad - f_hyp)) < 1e-3


def test_isotherm_zero_at_zero_ligand():
    assert quadratic_binding_isotherm(np.array([0.0]), 5.0, 0.001)[0] == 0.0


def test_fit_kd_recovers_known_value_from_noisy_data():
    rng = np.random.default_rng(4)
    Kd_true = 5.0
    T_total = 0.001
    L = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 50])
    f_true = quadratic_binding_isotherm(L, Kd_true, T_total)
    f_noisy = f_true + rng.normal(0, 0.01, size=f_true.shape)

    result = fit_kd(L, f_noisy, T_total=T_total)
    assert result.success
    assert abs(result.Kd - Kd_true) / Kd_true < 0.2
