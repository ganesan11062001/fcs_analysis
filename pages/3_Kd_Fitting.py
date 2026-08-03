import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.kd import fit_kd, quadratic_binding_isotherm
from core.export import kd_result_to_df, df_to_csv_bytes

st.set_page_config(page_title="Kd Fitting", layout="wide")

st.title("Kd Fitting from a Concentration Series")
st.caption(
    "Given bound-fraction results across a series of files at varying [NT] (fixed [Tau-FL]), "
    "fit the 1:1 binding isotherm (full quadratic/ligand-depletion form) to extract Kd."
)

source = st.radio(
    "Data source",
    ["Manual entry", "Pull from last Batch run"],
    horizontal=True,
    help="'Pull from last Batch run' uses the bound_fraction_tau_fl column computed on the Batch / Time Course page.",
)

if source == "Pull from last Batch run":
    if "batch_comparison_df" not in st.session_state or "bound_fraction_tau_fl" not in st.session_state["batch_comparison_df"]:
        st.warning("No dual-channel batch run with bound fractions found yet. Run one on the Batch / Time Course page first.")
        st.stop()
    base_df = st.session_state["batch_comparison_df"][["label", "bound_fraction_tau_fl"]].copy()
    base_df.insert(1, "NT_concentration", np.nan)
    base_df["include"] = True
    base_df = base_df.rename(columns={"bound_fraction_tau_fl": "bound_fraction"})
else:
    base_df = pd.DataFrame(
        {"label": ["sample 1", "sample 2", "sample 3"], "NT_concentration": [0.0, 0.0, 0.0], "bound_fraction": [0.0, 0.0, 0.0], "include": True}
    )

st.subheader("1. Concentration series data")
st.caption("Type the [NT] concentration for each file/sample directly into the table below.")
kd_table = st.data_editor(base_df, key="kd_table_editor", num_rows="dynamic", hide_index=True)

T_total = st.number_input("Fixed Tau-FL tracer concentration ([T]_total)", min_value=1e-9, value=0.001, format="%.6f")

if st.button("Fit Kd", type="primary"):
    active = kd_table[kd_table["include"].fillna(True)]
    active = active.dropna(subset=["NT_concentration", "bound_fraction"])
    if len(active) < 3:
        st.error("Need at least 3 included data points with both a concentration and a bound fraction.")
    else:
        result = fit_kd(active["NT_concentration"].to_numpy(), active["bound_fraction"].to_numpy(), T_total=T_total)
        st.session_state["kd_fit_result"] = result
        st.session_state["kd_fit_data"] = active

if "kd_fit_result" in st.session_state:
    result = st.session_state["kd_fit_result"]
    active = st.session_state["kd_fit_data"]

    if result.success:
        st.success(f"Kd = {result.Kd:.4g} +/- {result.Kd_stderr:.4g}  (reduced chi^2 = {result.redchi:.4g})")
    else:
        st.error(f"Kd fit did not converge: {result.message}")

    L_smooth = np.linspace(0, active["NT_concentration"].max() * 1.1, 200)
    f_smooth = quadratic_binding_isotherm(L_smooth, result.Kd, result.T_total, result.Fmax, result.Fmin)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=active["NT_concentration"], y=active["bound_fraction"], mode="markers", name="data"))
    fig.add_trace(go.Scatter(x=L_smooth, y=f_smooth, mode="lines", name="isotherm fit"))
    fig.update_layout(xaxis_title="[NT] concentration", yaxis_title="Bound fraction", height=400)
    st.plotly_chart(fig, width="stretch")

    st.download_button(
        "Download Kd fit (CSV)",
        df_to_csv_bytes(kd_result_to_df(active["NT_concentration"], active["bound_fraction"], result)),
        file_name="kd_fit.csv",
    )
