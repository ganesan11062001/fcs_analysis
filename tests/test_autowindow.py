import numpy as np

from core.autowindow import search_best_window, sliding_window_starts


def test_sliding_window_starts_steps_by_one_second():
    starts = sliding_window_starts(t_min=0.0, t_max=10.0, window_length_s=3.0, step_s=1.0)
    # 0-3, 1-4, 2-5, ..., 7-10 -- stops once a full window no longer fits.
    assert starts == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_sliding_window_starts_falls_back_to_whole_trace_when_too_long():
    starts = sliding_window_starts(t_min=0.0, t_max=5.0, window_length_s=10.0, step_s=1.0)
    assert starts == [0.0]


def test_search_best_window_generates_one_candidate_per_sliding_start(rng):
    dt = 1e-3
    n = 10000  # ~10 s trace
    ch1 = rng.poisson(8, n).astype(float)
    time_arr = np.arange(n) * dt

    result = search_best_window(
        time_arr, {"CH1": ch1}, window_length_s=3.0, step_s=1.0, segments=3, points_per_segment=8, base=4, n_blocks=3
    )

    expected_starts = sliding_window_starts(float(time_arr[0]), float(time_arr[-1]), 3.0, 1.0)
    assert len(result.candidates) == len(expected_starts)
    assert result.chosen in result.candidates
    assert not result.chosen.failed
    assert np.isfinite(result.chosen.score)
    # All candidates should be (approximately) the same length -- fair to compare.
    for c in result.candidates:
        assert np.isclose(c.t1 - c.t0, 3.0, atol=1e-6)


def test_search_best_window_prefers_the_window_over_a_correlated_region(rng):
    """Build a trace that's quiet everywhere except a strong slow modulation
    between t=3s and t=6s, then confirm a sliding window overlapping that
    region scores higher than one fully outside it."""
    dt = 1e-3
    n = 10000  # ~10 s trace
    counts = rng.poisson(5, n).astype(float)
    modulation_region = slice(3000, 6000)  # t in [3, 6) s
    modulation = 20 * (np.arange(3000) // 200 % 2)
    counts[modulation_region] += modulation
    time_arr = np.arange(n) * dt

    result = search_best_window(
        time_arr, {"CH1": counts}, window_length_s=3.0, step_s=1.0, segments=3, points_per_segment=8, base=4, n_blocks=3
    )

    by_t0 = {round(c.t0, 3): c for c in result.candidates}
    inside_modulated_region = by_t0[3.0]  # window [3, 6) s, fully inside the modulated region
    fully_quiet = by_t0[6.0]  # window [6, 9) s, fully outside it
    assert inside_modulated_region.score > fully_quiet.score


def test_search_best_window_skips_candidates_too_short_for_sub_blocks(rng):
    dt = 1e-3
    n = 10000
    ch1 = rng.poisson(5, n).astype(float)
    time_arr = np.arange(n) * dt

    # A 3 s window at n_blocks=100 sub-blocks needs >=200 points/sub-block worth of
    # span per block; with only ~3000 points per 3s window this can't be satisfied.
    result = search_best_window(
        time_arr, {"CH1": ch1}, window_length_s=3.0, step_s=1.0, segments=2, points_per_segment=3, base=4, n_blocks=2000
    )

    assert any(c.failed for c in result.candidates)


def test_search_best_window_dual_channel_includes_cross(rng):
    dt = 1e-3
    n = 10000
    ch1 = rng.poisson(8, n).astype(float)
    ch2 = rng.poisson(4, n).astype(float)
    time_arr = np.arange(n) * dt

    result = search_best_window(
        time_arr, {"CH1": ch1, "CH2": ch2}, window_length_s=3.0, step_s=1.0,
        segments=3, points_per_segment=8, base=4, n_blocks=3,
    )

    assert set(result.chosen.corr_results.keys()) == {"acf_ch1", "acf_ch2", "cross"}
