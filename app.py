import streamlit as st

from app_common import init_defaults, DEFAULT_SETTINGS

st.set_page_config(page_title="FCS/FCCS Analysis", layout="wide")
init_defaults()

st.title("FCS/FCCS Analysis for Tau Aggregation Studies")

st.markdown(
    """
This tool analyzes Fluorescence Correlation/Cross-Correlation Spectroscopy (FCS/FCCS)
traces exported from an ISS VistaVision instrument, without needing VistaVision itself.

**Use the pages in the sidebar:**
- **Single File Analysis** — load one trace, trim the time window, run the multi-tau
  correlation engine, and fit diffusion models to extract diffusion time and (for
  dual-channel data) the FCCS bound fraction.
- **Batch / Time Course** — process a set of files (e.g. 0hr/3hr/48hr) with identical
  settings and compare results.
- **Kd Fitting** — fit a binding isotherm to bound-fraction-vs-[NT] data from a
  concentration series to extract Kd.
- **Validation** — re-run the exact-FFT-vs-multi-tau cross-check and generate
  synthetic data with a known diffusion time to sanity-check the whole pipeline.

**Global defaults** (used to pre-fill settings on every page; each page lets you
override them per file/run):
"""
)

col1, col2, col3, col4, col5 = st.columns(5)
d = st.session_state["defaults"]
d["segments"] = col1.number_input("Segments", min_value=1, max_value=20, value=d["segments"])
d["points_per_segment"] = col2.number_input("Points/segment", min_value=1, max_value=100, value=d["points_per_segment"])
d["base"] = col3.number_input("Grouping base", min_value=2, max_value=16, value=d["base"])
d["kappa"] = col4.number_input("kappa (structure param, fixed)", min_value=0.1, value=d["kappa"], step=0.1)
d["min_reliable_n_samples"] = col5.number_input(
    "Min samples for a reliable tau point", min_value=1, value=d["min_reliable_n_samples"]
)

st.caption(
    "Defaults: 5 segments x 15 points/segment, base 4 -- matches VistaVision manual "
    "section 13.4.1. kappa has no calibration workflow yet (see README); tau_D and "
    "amplitude/N are reported directly rather than converted to an absolute diffusion "
    "coefficient or concentration."
)
