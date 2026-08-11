import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_common import sidebar_about, apply_chart_style, MUTED_INK, SERIES_COLORS
from core.kd import fit_kd, quadratic_binding_isotherm
from core.export import kd_result_to_df, df_to_csv_bytes

st.set_page_config(page_title="Kd Fitting", layout="wide")
sidebar_about()

st.title("Kd Fitting from a Concentration Series")
st.caption(
    "Given bound-fraction results across a series of files at varying ligand concentration "
    "[L] (fixed tracer concentration [T]), fit the 1:1 binding isotherm (full "
    "quadratic/ligand-depletion form) to extract Kd."
)

source = st.radio(
    "Data source",
    ["Manual entry", "Pull from last Batch run"],
    horizontal=True,
    help="'Pull from last Batch run' uses the bound_fraction_ch2_species column computed on the Batch / Time Course page.",
)

if source == "Pull from last Batch run":
    if "batch_comparison_df" not in st.session_state or "bound_fraction_ch2_species" not in st.session_state["batch_comparison_df"]:
        st.warning("No dual-channel batch run with bound fractions found yet. Run one on the Batch / Time Course page first.")
        st.stop()
    base_df = st.session_state["batch_comparison_df"][["label", "bound_fraction_ch2_species"]].copy()
    base_df.insert(1, "ligand_concentration", np.nan)
    base_df["include"] = True
    base_df = base_df.rename(columns={"bound_fraction_ch2_species": "bound_fraction"})
else:
    base_df = pd.DataFrame(
        {"label": ["sample 1", "sample 2", "sample 3"], "ligand_concentration": [0.0, 0.0, 0.0], "bound_fraction": [0.0, 0.0, 0.0], "include": True}
    )

with st.container(border=True):
    st.subheader("1. Concentration series data")
    st.caption("Type the ligand concentration [L] for each file/sample directly into the table below.")
    kd_table = st.data_editor(base_df, key="kd_table_editor", num_rows="dynamic", hide_index=True)

    T_total = st.number_input("Fixed tracer concentration ([T]_total)", min_value=1e-9, value=0.001, format="%.6f")

    if st.button("Fit Kd", type="primary"):
        active = kd_table[kd_table["include"].fillna(True)]
        active = active.dropna(subset=["ligand_concentration", "bound_fraction"])
        if len(active) < 3:
            st.error("Need at least 3 included data points with both a concentration and a bound fraction.")
        else:
            with st.spinner("Fitting binding isotherm..."):
                result = fit_kd(active["ligand_concentration"].to_numpy(), active["bound_fraction"].to_numpy(), T_total=T_total)
            st.session_state["kd_fit_result"] = result
            st.session_state["kd_fit_data"] = active

if "kd_fit_result" in st.session_state:
    with st.container(border=True):
        st.subheader("2. Fit result")
        result = st.session_state["kd_fit_result"]
        active = st.session_state["kd_fit_data"]

        if result.success:
            kcols = st.columns(3)
            kcols[0].metric("Kd", f"{result.Kd:.4g}", help="Dissociation constant, same units as ligand_concentration.")
            kcols[1].metric("Kd stderr", f"{result.Kd_stderr:.4g}")
            kcols[2].metric("Reduced chi^2", f"{result.redchi:.4g}")
        else:
            st.error(f"Kd fit did not converge: {result.message}")

        L_smooth = np.linspace(0, active["ligand_concentration"].max() * 1.1, 200)
        f_smooth = quadratic_binding_isotherm(L_smooth, result.Kd, result.T_total, result.Fmax, result.Fmin)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=active["ligand_concentration"], y=active["bound_fraction"], mode="markers", name="data",
                marker=dict(size=9, color=MUTED_INK, opacity=0.8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=L_smooth, y=f_smooth, mode="lines", name="isotherm fit",
                line=dict(color=SERIES_COLORS["acf_ch1"], width=2),
            )
        )
        fig.update_layout(
            xaxis_title="Ligand concentration [L]", yaxis_title="Bound fraction", height=400,
            legend=dict(orientation="h"), plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        )
        st.plotly_chart(apply_chart_style(fig), width="stretch")

        st.download_button(
            "Download Kd fit (CSV)",
            df_to_csv_bytes(kd_result_to_df(active["ligand_concentration"], active["bound_fraction"], result)),
            file_name="kd_fit.csv",
        )
