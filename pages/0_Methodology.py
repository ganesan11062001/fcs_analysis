import streamlit as st

from app_common import sidebar_about

st.set_page_config(page_title="Methodology", layout="wide")
sidebar_about()

st.title("Methodology & Formulas")
st.caption(
    "What each stage of the pipeline actually computes, in order. Every number the app "
    "reports traces back to one of the formulas below -- nothing here is instrument-specific, "
    "it's the standard math used across FCS/FCCS analysis generally."
)

with st.container(border=True):
    st.subheader("The pipeline, in order")
    st.markdown(
        """
1. **Load** — parse the CSV's time column + 1 or 2 intensity columns.
2. **Auto-select the time window** — evaluate a few candidate windows and keep the
   cleanest one, no manual trimming needed.
3. **Bin** *(optional)* — group N raw points together before correlating.
4. **Correlate** — compute the autocorrelation of each channel, and (if 2 channels) the
   symmetrized cross-correlation, on a multi-tau lag grid.
5. **Fit** — fit a diffusion model to each correlation curve to get diffusion time(s) and
   amplitude.
6. **FCCS bound fraction** *(2-channel only)* — combine the three fitted amplitudes.
7. **Kd** *(not yet available in this deployment)* — fit a binding isotherm across a
   concentration series of bound fractions.
        """
    )

with st.container(border=True):
    st.subheader("1. Correlation function")
    st.markdown("The core quantity computed at every lag time τ:")
    st.latex(r"G(\tau) = \frac{\langle \delta I(t)\,\delta I(t+\tau) \rangle}{\langle I \rangle^2}")
    st.markdown(
        r"where $\delta I(t) = I(t) - \langle I \rangle$. For a two-channel trace, the "
        r"**cross-correlation** replaces one $I$ with the other channel's intensity, and is "
        r"symmetrized to remove lag-direction bias:"
    )
    st.latex(r"G_\times(\tau) = \tfrac{1}{2}\left[G_{1\to2}(\tau) + G_{2\to1}(\tau)\right]")

with st.container(border=True):
    st.subheader("2. Automatic time-window selection")
    st.markdown(
        "Instead of asking you to eyeball and drag a time-window trim, the app evaluates a "
        "small, fixed set of candidate windows -- the **full trace**, and the **first half**, "
        "**middle half**, and **second half** of it -- and keeps whichever produces the "
        "cleanest correlation curve. Cleanliness is scored as the median signal-to-noise "
        "ratio across tau, using the sub-block error estimate from step 4:"
    )
    st.latex(r"\text{SNR} = \operatorname{median}_\tau \frac{|G(\tau)|}{\sigma_{G(\tau)}}")
    st.markdown(
        "For a dual-channel trace, the score is averaged across CH1, CH2, and the "
        "cross-correlation. A candidate window is skipped if it's too short to split into "
        "the configured number of sub-blocks. This is a fixed 4-candidate grid search, not an "
        "exhaustive scan -- it targets the common case of a bad stretch at the very start or "
        "end of acquisition (photobleaching, focus drift, a bubble), not arbitrary trace-quality "
        "problems."
    )

with st.container(border=True):
    st.subheader("3. Multi-tau lag grid")
    st.markdown(
        "Rather than evaluating every lag linearly (too slow for million-point traces), "
        "lags are grouped into **segments**, each coarser than the last by a fixed **base**:"
    )
    st.latex(r"\Delta t_k = \Delta t_0 \cdot \text{base}^{\,k}, \qquad k = 0, 1, \dots, \text{segments}-1")
    st.latex(r"\tau = \Delta t_k \cdot i, \qquad i = 1, \dots, \text{points\_per\_segment}")
    st.markdown(
        "This gives even log-spaced τ coverage across many decades without the O(n²) cost "
        "of a brute-force correlation. Each τ point also tracks how many terms were actually "
        "averaged into it (`n_samples`) — points near the tail of the trace are averaged over "
        "fewer samples and are flagged as statistically less reliable."
    )

with st.container(border=True):
    st.subheader("4. Per-point error (optional)")
    st.markdown(
        "The trace is split into **N contiguous sub-blocks**; each is independently run "
        "through steps 1–2 with identical settings. The uncertainty at each τ is the "
        "standard error of G(τ) across sub-blocks:"
    )
    st.latex(r"\sigma_{G(\tau)} = \frac{\text{std}\left(\{G_b(\tau)\}_{b=1}^{N}\right)}{\sqrt{N}}")
    st.caption(
        "A standard block-averaging convention used broadly across FCS correlator software "
        "— not a reproduction of any one instrument's internal error algorithm."
    )

with st.container(border=True):
    st.subheader("5. Diffusion model (the fit)")
    st.markdown("Standard confocal 3D-diffusion model, for a single diffusing species:")
    st.latex(
        r"G(\tau) = \frac{1}{N} \cdot \frac{1}{1+\tau/\tau_D} \cdot "
        r"\frac{1}{\sqrt{1+\tau/(\kappa^2 \tau_D)}}"
    )
    st.markdown(
        r"$N$ is the average number of particles in the detection volume (amplitude), "
        r"$\tau_D$ is the diffusion time, and $\kappa = w_z/w_{xy}$ is the fixed structure "
        r"parameter (axial:radial ratio of the detection volume). For **two** independently "
        r"diffusing species (component count = 2):"
    )
    st.latex(
        r"G(\tau) = \frac{1}{N}\left[f_1 \cdot D(\tau,\tau_{D1}) + (1-f_1)\cdot D(\tau,\tau_{D2})\right]"
    )
    st.markdown("An optional fast blinking/triplet-state prefactor can be multiplied in:")
    st.latex(r"G(\tau) \;\leftarrow\; G(\tau)\cdot \frac{1-T+T\,e^{-\tau/\tau_{trip}}}{1-T}")

with st.container(border=True):
    st.subheader("6. FCCS bound fraction")
    st.markdown(
        "Combines the fitted zero-lag amplitudes of both autocorrelations and the "
        "cross-correlation (uncorrected amplitude-ratio form — no spectral crosstalk "
        "correction applied):"
    )
    st.latex(r"f_{\text{bound, CH2 species}} = \frac{G_\times(0)}{G_{\text{CH1}}(0)}")
    st.latex(r"f_{\text{bound, CH1 species}} = \frac{G_\times(0)}{G_{\text{CH2}}(0)}")

with st.container(border=True):
    st.subheader("7. Kd — binding isotherm")
    st.markdown(
        "Fits bound fraction vs. ligand concentration $L$ to the full quadratic 1:1 "
        "binding equation (accounts for ligand depletion, valid even when $L$ and the "
        "fixed tracer concentration $T$ are comparable in magnitude — not just the "
        "simple hyperbolic approximation valid only when $L \\gg T$):"
    )
    st.latex(
        r"f_{\text{bound}}(L) = F_{min} + (F_{max}-F_{min}) \cdot "
        r"\frac{(K_d+T+L) - \sqrt{(K_d+T+L)^2 - 4TL}}{2T}"
    )

st.caption(
    "Correctness of the correlation step (sections 1 and 3) is checked against a brute-force "
    "FFT correlation and a synthetic trace with a known diffusion time on the Validation page."
)
