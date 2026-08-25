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
    text = correlation_results_to_vistavision_csv_text(
        results, errors, sampling_rate_hz=10000, mean_rates=[8.0, 4.0], measurement_time_s=30.0,
    )
    lines = text.split("\n")

    assert lines[0] == "[HeaderV]"
    assert "Version, 3" in lines
    assert "Sections,3" in lines
    assert "PtsPerSection, 6" in lines
    assert "SampleFrequency, 10000" in lines
    assert "ChannelCount, 2" in lines
    assert "MeasurementTime(sec), 30" in lines
    assert "CPS, 8, 4" in lines
    assert "CrossChannelIDs,01" in lines
    assert "[Data]" in lines

    assert "Tau" in lines
    tau_row = lines[lines.index("Tau") + 1]
    assert np.allclose([float(v) for v in tau_row.split(",")], results["acf_ch1"].tau)

    assert "Ch0 AutoCorrelation" in lines
    ch0_row = lines[lines.index("Ch0 AutoCorrelation") + 1]
    assert np.allclose([float(v) for v in ch0_row.split(",")], results["acf_ch1"].g)

    assert "Ch1 AutoCorrelation" in lines
    assert "Ch0x1 CrossCorrelation" in lines
    assert "Ch0 AutoCorrelation Standard Deviation" in lines
    assert "Ch1 AutoCorrelation Standard Deviation" in lines
    assert "Ch0x1 CrossCorrelation Standard Deviation" in lines


def test_vistavision_csv_text_single_channel_omits_ch1_and_cross(rng):
    n = 1000
    ch1 = rng.poisson(8, n).astype(float)
    channels = {"CH1": ch1}
    results = compute_all_correlations(channels, 1e-4, segments=3, points_per_segment=6, base=4)
    errors = compute_correlation_error(results, channels, 1e-4, segments=3, points_per_segment=6, base=4, n_blocks=4)

    text = correlation_results_to_vistavision_csv_text(
        results, errors, sampling_rate_hz=10000, mean_rates=[8.0], measurement_time_s=10.0,
    )
    lines = text.split("\n")
    assert "ChannelCount, 1" in lines
    assert "CPS, 8" in lines
    assert "Ch0 AutoCorrelation" in lines
    assert "Ch1 AutoCorrelation" not in lines
    assert "Ch0x1 CrossCorrelation" not in lines
    assert "CrossChannelIDs,01" not in lines


def test_vistavision_csv_text_nan_error_written_as_zero(rng):
    """compute_correlation_error can produce NaN at tau points too few sub-blocks
    reach; the text format has no other way to express that, so it's written as 0."""
    n = 2000
    ch1 = rng.poisson(8, n).astype(float)
    channels = {"CH1": ch1}
    results = compute_all_correlations(channels, 1e-4, segments=5, points_per_segment=10, base=4)
    errors = compute_correlation_error(results, channels, 1e-4, segments=5, points_per_segment=10, base=4, n_blocks=3)
    assert np.any(np.isnan(errors["acf_ch1"]))

    text = correlation_results_to_vistavision_csv_text(
        results, errors, sampling_rate_hz=10000, mean_rates=[8.0], measurement_time_s=10.0,
    )
    lines = text.split("\n")
    err_row = lines[lines.index("Ch0 AutoCorrelation Standard Deviation") + 1]
    assert "nan" not in err_row.lower()
