import numpy as np

from core.autowindow import search_best_window


def test_search_best_window_returns_four_candidates_and_a_chosen_one(rng):
    n = 4000
    dt = 1e-4
    ch1 = rng.poisson(8, n).astype(float)
    time_arr = np.arange(n) * dt

    result = search_best_window(time_arr, {"CH1": ch1}, segments=3, points_per_segment=8, base=4, n_blocks=3)

    labels = {c.label for c in result.candidates}
    assert labels == {"full trace", "first half", "middle half", "second half"}
    assert result.chosen in result.candidates
    assert not result.chosen.failed
    assert np.isfinite(result.chosen.score)


def test_search_best_window_prefers_cleaner_half_over_noisy_half(rng):
    """Build a trace where the second half is pure noise (no real correlation
    signal beyond shot noise) and the first half has a strong injected slow
    fluctuation, then confirm the search doesn't blindly always pick the
    full trace or a fixed half -- it scores candidates independently."""
    n_half = 4000
    dt = 1e-4
    quiet = rng.poisson(5, n_half).astype(float)
    # A slow square-wave-like modulation creates strong low-lag correlation.
    modulation = 20 * (np.arange(n_half) // 200 % 2)
    correlated = rng.poisson(5, n_half).astype(float) + modulation
    ch1 = np.concatenate([correlated, quiet])
    time_arr = np.arange(len(ch1)) * dt

    result = search_best_window(time_arr, {"CH1": ch1}, segments=3, points_per_segment=8, base=4, n_blocks=3)

    by_label = {c.label: c for c in result.candidates}
    # The correlated first half should score higher than the quiet second half.
    assert by_label["first half"].score > by_label["second half"].score


def test_search_best_window_skips_candidates_too_short_for_sub_blocks(rng):
    n = 16  # short enough that a half-length window (~8 points) can't be split into 5 sub-blocks of >=2 points
    dt = 1e-4
    ch1 = rng.poisson(5, n).astype(float)
    time_arr = np.arange(n) * dt

    result = search_best_window(time_arr, {"CH1": ch1}, segments=2, points_per_segment=3, base=4, n_blocks=5)

    assert any(c.failed for c in result.candidates)
    assert not result.chosen.failed


def test_search_best_window_dual_channel_includes_cross(rng):
    n = 4000
    dt = 1e-4
    ch1 = rng.poisson(8, n).astype(float)
    ch2 = rng.poisson(4, n).astype(float)
    time_arr = np.arange(n) * dt

    result = search_best_window(
        time_arr, {"CH1": ch1, "CH2": ch2}, segments=3, points_per_segment=8, base=4, n_blocks=3
    )

    assert set(result.chosen.corr_results.keys()) == {"acf_ch1", "acf_ch2", "cross"}
