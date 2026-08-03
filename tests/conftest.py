import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def _write_csv(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(row) + "\n")


@pytest.fixture
def single_channel_csv(tmp_path, rng):
    """A small single-channel VistaVision-style CSV: '\\t  time,   count'."""
    n = 2000
    dt = 1e-4
    counts = rng.poisson(5, n)
    path = tmp_path / "single_channel.csv"
    rows = [(f"\t  {i * dt:.6f}", f"{'':>9}{c}") for i, c in enumerate(counts)]
    _write_csv(path, rows)
    return path, dt, counts.astype(np.float64)


@pytest.fixture
def dual_channel_csv(tmp_path, rng):
    """A small dual-channel 'AllInOne' VistaVision-style CSV."""
    n = 2000
    dt = 1e-4
    ch1 = rng.poisson(8, n)
    ch2 = rng.poisson(4, n)
    path = tmp_path / "dual_channel.csv"
    rows = [(f"\t  {i * dt:.6f}", f"{'':>9}{a}", f"{'':>9}{b}") for i, (a, b) in enumerate(zip(ch1, ch2))]
    _write_csv(path, rows)
    return path, dt, ch1.astype(np.float64), ch2.astype(np.float64)


@pytest.fixture(scope="session")
def known_tauD_synthetic():
    from core.validation import generate_synthetic_fcs_trace

    return generate_synthetic_fcs_trace(
        tauD=2e-4, dt=1e-5, duration_s=30.0, N_particles=5.0, kappa=5.0, seed=42
    )
