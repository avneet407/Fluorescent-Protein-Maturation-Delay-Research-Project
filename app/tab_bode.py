# --- Bode Plot tab: frequency response of the full maturation model ------
# Lets the user directly enter one or more parameter sets (each 1-step or
# 2-step) and overlay their magnitude/phase Bode curves on one graph, rather
# than pulling parameters from the Simulation tab.

import numpy as np
import pandas as pd
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


def render_bode_tab():
    st.markdown(
        "Frequency response of the full maturation + photobleaching model "
        "(magnitude and phase), showing how strongly the reporter "
        "attenuates and delays gene expression fluctuations at each "
        "frequency. Define one or more parameter sets below (each 1-step "
        "or 2-step), click **Run** to add each to the graph, and compare "
        "them all on the same plot."
    )

    st.session_state.setdefault("bode_param_sets", [])
    param_sets = st.session_state["bode_param_sets"]

    st.subheader("Add a parameter set")

    add_cols = st.columns([2, 1])
    with add_cols[0]:
        new_label = st.text_input(
            "Label (optional)", key="bode_new_label",
            placeholder=f"Set {len(param_sets) + 1}",
        )
    with add_cols[1]:
        new_is_two_step = st.selectbox(
            "Model", ["1-step (I -> M)", "2-step (I -> X -> M)"], key="bode_new_model",
        ) == "2-step (I -> X -> M)"

    if new_is_two_step:
        pc = st.columns(5)
        new_alpha = pc[0].number_input(
            "alpha - fluorescence scaling factor", min_value=0.0, value=1.0, step=0.1,
            key="bode_new_alpha",
        )
        new_k1 = pc[1].number_input(
            "k1 - rate I -> X", min_value=0.0, value=0.20, step=0.01, format="%.3f",
            key="bode_new_k1",
        )
        new_k2 = pc[2].number_input(
            "k2 - rate X -> M", min_value=0.0, value=0.10, step=0.01, format="%.3f",
            key="bode_new_k2",
        )
        new_kb = pc[3].number_input(
            "kb - photobleaching rate (M -> B)", min_value=0.0, value=0.02, step=0.005, format="%.4f",
            key="bode_new_kb",
        )
        new_kd = pc[4].number_input(
            "kd - degradation / dilution rate", min_value=0.0, value=0.0, step=0.005, format="%.4f",
            key="bode_new_kd",
        )
    else:
        pc = st.columns(4)
        new_alpha = pc[0].number_input(
            "alpha - fluorescence scaling factor", min_value=0.0, value=1.0, step=0.1,
            key="bode_new_alpha",
        )
        new_km = pc[1].number_input(
            "km - rate I -> M", min_value=0.0, value=0.15, step=0.01, format="%.3f",
            key="bode_new_km",
        )
        new_kb = pc[2].number_input(
            "kb - photobleaching rate (M -> B)", min_value=0.0, value=0.02, step=0.005, format="%.4f",
            key="bode_new_kb",
        )
        new_kd = pc[3].number_input(
            "kd - degradation / dilution rate", min_value=0.0, value=0.0, step=0.005, format="%.4f",
            key="bode_new_kd",
        )

    if st.button("Run (add to graph)", type="primary", key="bode_add_set"):
        label = new_label.strip() or f"Set {len(param_sets) + 1}"
        new_set = {
            "label": label, "is_two_step": new_is_two_step,
            "alpha": new_alpha, "kb": new_kb, "kd": new_kd,
        }
        if new_is_two_step:
            new_set["k1"] = new_k1
            new_set["k2"] = new_k2
        else:
            new_set["km"] = new_km
        param_sets.append(new_set)

    if param_sets:
        with st.expander(f"Parameter sets on graph ({len(param_sets)})", expanded=False):
            for i, ps in enumerate(param_sets):
                row_cols = st.columns([5, 1])
                with row_cols[0]:
                    model_label = "2-step" if ps["is_two_step"] else "1-step"
                    if ps["is_two_step"]:
                        details = (
                            f"alpha={ps['alpha']:.3f}, k1={ps['k1']:.3f}, k2={ps['k2']:.3f}, "
                            f"kb={ps['kb']:.4f}, kd={ps['kd']:.4f}"
                        )
                    else:
                        details = (
                            f"alpha={ps['alpha']:.3f}, km={ps['km']:.3f}, "
                            f"kb={ps['kb']:.4f}, kd={ps['kd']:.4f}"
                        )
                    st.write(f"**{ps['label']}** ({model_label}) — {details}")
                with row_cols[1]:
                    if st.button("Remove", key=f"bode_remove_{i}"):
                        param_sets.pop(i)
                        st.rerun()
            if st.button("Clear all", key="bode_clear_all"):
                st.session_state["bode_param_sets"] = []
                st.rerun()

    st.divider()
    st.subheader("Frequency range")
    freq_cols = st.columns(3)
    with freq_cols[0]:
        w_start_exp = st.number_input(
            "Frequency range: 10^ (start)", value=-4.0, step=1.0, key="bode_w_start",
        )
    with freq_cols[1]:
        w_end_exp = st.number_input(
            "Frequency range: 10^ (end)", value=2.0, step=1.0, key="bode_w_end",
        )
    with freq_cols[2]:
        w_points = st.number_input(
            "Number of frequency points", min_value=10, value=1200, step=100, key="bode_w_points",
        )

    if not param_sets:
        st.info("Add at least one parameter set above and click **Run** to see the Bode plot.")
        return

    w = np.logspace(w_start_exp, w_end_exp, int(w_points))

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6))
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_xlabel("Frequency (rad/sec)")
    ax_mag.set_title("Bode Plot")
    ax_mag.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.set_xlabel("Frequency (rad/sec)")
    ax_phase.grid(True, which="both", linestyle="--", alpha=0.5)

    has_one_step = any(not ps["is_two_step"] for ps in param_sets)
    has_two_step = any(ps["is_two_step"] for ps in param_sets)

    summary_rows = []
    for ps in param_sets:
        if ps["is_two_step"]:
            _, mag, phase = bode_2step(ps["alpha"], ps["k1"], ps["k2"], ps["kb"], ps["kd"], w)
            wc_analytical = analytical_cutoff_2step(ps["k1"], ps["k2"], ps["kb"], ps["kd"])
        else:
            _, mag, phase = bode_1step(ps["alpha"], ps["km"], ps["kb"], ps["kd"], w)
            wc_analytical = analytical_cutoff_1step(ps["km"], ps["kb"], ps["kd"])
        wc_numerical = numerical_cutoff(w, mag)

        ax_mag.semilogx(w, mag, label=ps["label"])
        ax_phase.semilogx(w, phase, label=ps["label"])

        summary_rows.append({
            "Label": ps["label"],
            "Model": "2-step" if ps["is_two_step"] else "1-step",
            "Numerical cutoff (rad/s)": wc_numerical,
            "Analytical cutoff (rad/s)": wc_analytical,
        })

    ax_mag.legend(fontsize=8)
    ax_phase.legend(fontsize=8)
    fig.tight_layout()

    if has_one_step:
        st.latex(FORMULA_1STEP)
    if has_two_step:
        st.latex(FORMULA_2STEP)

    st.pyplot(fig)

    st.markdown("**-3 dB cutoff frequencies**")
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df.set_index("Label"))
