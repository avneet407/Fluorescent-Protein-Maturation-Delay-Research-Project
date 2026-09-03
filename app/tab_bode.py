# --- Bode Plot tab: frequency response of the full maturation model ------
# Plots magnitude/phase for the model+params currently set in the
# Simulation tab, optionally overlaid with the synthetic-data ground truth
# and/or the most recent Least Squares Fitting result for comparison.

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from bode_plot import (
    bode_1step,
    bode_2step,
    analytical_cutoff_1step,
    analytical_cutoff_2step,
    numerical_cutoff,
)

from app.shared import FORMULA_1STEP, FORMULA_2STEP


def render_bode_tab(sim_state):
    is_two_step = sim_state["is_two_step"]
    alpha = sim_state["alpha"]
    km = sim_state["km"]
    k1 = sim_state["k1"]
    k2 = sim_state["k2"]
    kb = sim_state["kb"]
    kd = sim_state["kd"]

    st.markdown(
        "Frequency response of the maturation model, showing how strongly the "
        "reporter attenuates and delays gene expression fluctuations at each "
        "frequency, and the cutoff frequency above which dynamics are lost to "
        "maturation/bleaching. Plots the synthetic-data ground truth (if "
        "generated) and the most recent least-squares fit (if run) for "
        "comparison, using the **Simulation** tab's rate constants at the "
        "time **Plot Bode Response** is clicked."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        w_start_exp = st.number_input(
            "Frequency range: 10^ (start)",
            value=-4.0 if is_two_step else -3.0, step=1.0,
            help="Suggested: -4 for 2-step, -3 for 1-step models.",
        )
    with col_b:
        w_end_exp = st.number_input(
            "Frequency range: 10^ (end)",
            value=2.0 if is_two_step else 1.0, step=1.0,
            help="Suggested: 2 for 2-step, 1 for 1-step models.",
        )
    with col_c:
        w_points = st.number_input(
            "Number of frequency points",
            min_value=10, value=1200 if is_two_step else 1000, step=100,
        )

    bode_button = st.button("Plot Bode Response", type="primary")

    if bode_button:
        with st.spinner("Computing Bode response..."):
            w = np.logspace(w_start_exp, w_end_exp, int(w_points))

            if is_two_step:
                w, mag, phase = bode_2step(alpha, k1, k2, kb, kd, w)
            else:
                w, mag, phase = bode_1step(alpha, km, kb, kd, w)

            synthetic_params = st.session_state.get("synthetic_params")
            mag_syn = phase_syn = syn_label = None
            if synthetic_params is not None:
                if synthetic_params["is_two_step"]:
                    _, mag_syn, phase_syn = bode_2step(
                        synthetic_params["alpha"], synthetic_params["k1"], synthetic_params["k2"],
                        synthetic_params["kb"], synthetic_params["kd"], w,
                    )
                    syn_label = "Synthetic input (2-step)"
                else:
                    _, mag_syn, phase_syn = bode_1step(
                        synthetic_params["alpha"], synthetic_params["km"],
                        synthetic_params["kb"], synthetic_params["kd"], w,
                    )
                    syn_label = "Synthetic input (1-step)"

            fit_result = st.session_state.get("fit_result")
            mag_fit = phase_fit = fit_label = None
            if fit_result is not None:
                if fit_result["is_two_step"]:
                    _, mag_fit, phase_fit = bode_2step(
                        fit_result["alpha"], fit_result["k1"], fit_result["k2"],
                        fit_result["kb"], fit_result["kd"], w,
                    )
                    fit_label = f"Least-squares fit (2-step, {fit_result['source_label']})"
                else:
                    _, mag_fit, phase_fit = bode_1step(
                        fit_result["alpha"], fit_result["km"],
                        fit_result["kb"], fit_result["kd"], w,
                    )
                    fit_label = f"Least-squares fit (1-step, {fit_result['source_label']})"

            wc_numerical = numerical_cutoff(w, mag)
            if is_two_step:
                wc_analytical = analytical_cutoff_2step(k1, k2, kb, kd)
            else:
                wc_analytical = analytical_cutoff_1step(km, kb, kd)

        st.session_state["bode_result"] = {
            "w": w, "mag": mag, "phase": phase,
            "mag_syn": mag_syn, "phase_syn": phase_syn, "syn_label": syn_label,
            "mag_fit": mag_fit, "phase_fit": phase_fit, "fit_label": fit_label,
            "wc_numerical": wc_numerical, "wc_analytical": wc_analytical,
            "formula_latex": FORMULA_2STEP if is_two_step else FORMULA_1STEP,
        }

    bode_result = st.session_state.get("bode_result")
    if bode_result is None:
        st.info("Set the frequency range and click **Plot Bode Response**.")
    else:
        br = bode_result
        if br.get("formula_latex"):
            st.latex(br["formula_latex"])

        fig4, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6))

        ax_mag.semilogx(br["w"], br["mag"])
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_xlabel("Frequency (rad/sec)")
        ax_mag.set_title("Bode Plot")
        ax_mag.grid(True, which="both", linestyle="--", alpha=0.5)

        ax_phase.semilogx(br["w"], br["phase"])
        ax_phase.set_ylabel("Phase (degrees)")
        ax_phase.set_xlabel("Frequency (rad/sec)")
        ax_phase.grid(True, which="both", linestyle="--", alpha=0.5)

        has_overlay = br["mag_syn"] is not None or br["mag_fit"] is not None
        if br["mag_syn"] is not None:
            ax_mag.semilogx(br["w"], br["mag_syn"], color="tab:green", linestyle="--", label=br["syn_label"])
            ax_phase.semilogx(br["w"], br["phase_syn"], color="tab:green", linestyle="--", label=br["syn_label"])
        if br["mag_fit"] is not None:
            ax_mag.semilogx(br["w"], br["mag_fit"], color="tab:red", linestyle=":", label=br["fit_label"])
            ax_phase.semilogx(br["w"], br["phase_fit"], color="tab:red", linestyle=":", label=br["fit_label"])

        if has_overlay:
            ax_mag.legend(fontsize=8)
            ax_phase.legend(fontsize=8)

        fig4.tight_layout()
        st.pyplot(fig4)

        if not has_overlay:
            st.info(
                "No synthetic-data ground truth or least-squares fit available yet to "
                "plot. Generate synthetic data and/or run a fit in the Least Squares "
                "Fitting tab to see curves here."
            )

        cutoff_col1, cutoff_col2 = st.columns(2)
        if br["wc_numerical"] is not None:
            cutoff_col1.metric("Numerical -3 dB cutoff (rad/sec)", f"{br['wc_numerical']:.6f}")
        else:
            cutoff_col1.warning("Cutoff not reached within the selected frequency range.")

        if br["wc_analytical"] is not None:
            cutoff_col2.metric("Analytical -3 dB cutoff (rad/sec)", f"{br['wc_analytical']:.6f}")
        else:
            cutoff_col2.warning("No positive real cutoff root found.")
