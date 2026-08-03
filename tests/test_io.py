import numpy as np

from core import engine
from core.io import detect_channel_count, load_trace_auto


def test_detect_channel_count_single(single_channel_csv):
    path, dt, counts = single_channel_csv
    assert detect_channel_count(str(path)) == 1


def test_detect_channel_count_dual(dual_channel_csv):
    path, dt, ch1, ch2 = dual_channel_csv
    assert detect_channel_count(str(path)) == 2


def test_load_trace_auto_single_matches_reference(single_channel_csv):
    path, dt, counts = single_channel_csv
    trace = load_trace_auto(str(path))
    assert trace.n_channels == 1
    assert not trace.used_fallback_parser
    assert np.array_equal(trace.channels["CH1"], counts)

    t_ref, c_ref = engine.load_trace(str(path), channel_col=1)
    assert np.array_equal(trace.time, t_ref)
    assert np.array_equal(trace.channels["CH1"], c_ref)


def test_load_trace_auto_dual_matches_reference(dual_channel_csv):
    path, dt, ch1, ch2 = dual_channel_csv
    trace = load_trace_auto(str(path))
    assert trace.n_channels == 2
    assert not trace.used_fallback_parser
    assert np.array_equal(trace.channels["CH1"], ch1)
    assert np.array_equal(trace.channels["CH2"], ch2)

    t_ref1, c_ref1 = engine.load_trace(str(path), channel_col=1)
    t_ref2, c_ref2 = engine.load_trace(str(path), channel_col=2)
    assert np.array_equal(trace.time, t_ref1)
    assert np.array_equal(trace.channels["CH1"], c_ref1)
    assert np.array_equal(trace.channels["CH2"], c_ref2)


def test_load_trace_auto_infers_dt(single_channel_csv):
    path, dt, counts = single_channel_csv
    trace = load_trace_auto(str(path))
    assert abs(trace.dt - dt) / dt < 1e-9


def test_load_trace_auto_falls_back_on_malformed_line(tmp_path):
    path = tmp_path / "malformed.csv"
    lines = [f"\t  {i * 1e-4:.6f},         {i % 5}\n" for i in range(20)]
    lines[10] = "garbage,not,numbers,here\n"  # inconsistent field count -> triggers fallback
    with open(path, "w") as f:
        f.writelines(lines)

    trace = load_trace_auto(str(path))
    # Malformed line should be skipped by the validated fallback parser (with a
    # stderr warning), not crash the whole load.
    assert trace.n_rows == 19
