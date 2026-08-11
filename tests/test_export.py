import numpy as np

from core.correlate import compute_all_correlations, compute_correlation_error
from core.export import correlation_results_to_df_with_error, correlation_results_to_vistavision_csv_text


def _dual_results_and_errors(rng, n=1000, dt=1e-4, n_blocks=4):
    ch1 = rng.poisson(8, n).astype(float)
    ch2 = rng.poisson(4, n).astype(float)
    channels = {"CH1": ch1, "CH2": ch2}
    results = compute_all_correlations(channels, dt, segments=3, points_per_segment=6, base=4)
    errors = compute_correlation_error(results, channels, dt, segments=3, points_per_segment=6, base=4, n_blocks=n_blocks)
    return results, errors


def test_correlation_results_to_df_with_error_has_matching_error_column(rng):
    results, errors = _dual_results_and_errors(rng)
    df = correlation_results_to_df_with_error(results, errors)

    assert set(df["kind"].unique()) == {"acf_ch1", "acf_ch2", "cross"}
    assert "error" in df.columns
    for kind in ("acf_ch1", "acf_ch2", "cross"):
        sub = df[df["kind"] == kind].reset_index(drop=True)
        assert np.allclose(sub["error"].to_numpy(), errors[kind], equal_nan=True)


def test_vistavision_csv_text_structure(rng):
    results, errors = _dual_results_and_errors(rng)
    text = correlation_results_to_vistavision_csv_text(results, errors, sampling_rate_hz=10000, mean_rates=[8.0, 4.0])
    lines = text.strip("\n").split("\n")

    assert lines[0] == "[HeaderX]"
    assert lines[1] == "3"  # segments
    assert lines[2] == "6"  # points_per_segment
    assert lines[3] == "10000"  # sampling rate
    assert lines[4] == "8,4"  # mean rates
    assert lines[5] == "[Data]"

    data_lines = lines[6:]
    assert len(data_lines) == len(results["acf_ch1"].tau)
    # tau, G_ch1, err_ch1, G_ch2, err_ch2, G_cross, err_cross -> 7 columns
    assert len(data_lines[0].split(",")) == 7

    first_tau, g_ch1_0 = data_lines[0].split(",")[0], data_lines[0].split(",")[1]
    assert np.isclose(float(first_tau), results["acf_ch1"].tau[0])
    assert np.isclose(float(g_ch1_0), results["acf_ch1"].g[0])


def test_vistavision_csv_text_single_channel_has_three_columns(rng):
    n = 1000
    ch1 = rng.poisson(8, n).astype(float)
    channels = {"CH1": ch1}
    results = compute_all_correlations(channels, 1e-4, segments=3, points_per_segment=6, base=4)
    errors = compute_correlation_error(results, channels, 1e-4, segments=3, points_per_segment=6, base=4, n_blocks=4)

    text = correlation_results_to_vistavision_csv_text(results, errors, sampling_rate_hz=10000, mean_rates=[8.0])
    lines = text.strip("\n").split("\n")
    assert lines[4] == "8"
    data_lines = lines[6:]
    assert len(data_lines[0].split(",")) == 3  # tau, G_ch1, err_ch1
