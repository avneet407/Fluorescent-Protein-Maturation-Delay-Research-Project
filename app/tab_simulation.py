"""Simulation tab: run the maturation ODE model with live parameters."""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp

from Maturation_Models import model_1step, model_2step


# --- Simulation tab: run the ODE model with live parameters --------------
# Returns a `sim_state` dict (`is_two_step`, `I0`, `km`/`k1`/`k2`, `kb`, `kd`,
# `alpha`, etc.) for the Data and Bode Plot tabs to use as their "current
# settings" defaults, and, on Run Simulation, integrates the model and
# plots I/M/B/F vs. time.
def render_simulation_tab():
    st.markdown(
        "Simulate a 1-step or 2-step protein maturation model and view the "
        "resulting fluorescence curve. Adjust initial conditions and rate "
        "constants below, then click **Run Simulation**. These values are "
        "also used to generate synthetic data in the **Data** tab and as "
        "the default rate constants in the **Bode Plot** tab."
    )

    model_choice = st.radio(
        "Maturation model",
        options=["1-step (I -> M)", "2-step (I -> X -> M)"],
        horizontal=True,
    )
    is_two_step = model_choice.startswith("2")

    st.subheader("Initial conditions")
    ic_cols = st.columns(4)
    with ic_cols[0]:
        I0 = st.number_input(
            "I0 - immature protein at t=0", min_value=0.0, value=100.0, step=10.0,
            help="Suggested: 100. Pool of just-translated protein present when translation is halted.",
        )
    X0 = 0.0
    with ic_cols[1]:
        if is_two_step:
            X0 = st.number_input(
                "X0 - intermediate protein at t=0", min_value=0.0, value=0.0, step=10.0,
                help="Suggested: 0. Usually no protein has reached the intermediate stage yet.",
            )
    with ic_cols[2]:
        M0 = st.number_input(
            "M0 - mature (fluorescent) protein at t=0", min_value=0.0, value=0.0, step=10.0,
            help="Suggested: 0. No protein has finished maturing at t=0.",
        )
    with ic_cols[3]:
        B0 = st.number_input(
            "B0 - bleached protein at t=0", min_value=0.0, value=0.0, step=10.0,
            help="Suggested: 0. No photobleaching has occurred yet.",
        )

    st.subheader("Rate constants")
    rc_cols = st.columns(6)
    with rc_cols[0]:
        if is_two_step:
            k1 = st.number_input(
                "k1 - rate I -> X", min_value=0.0, value=0.20, step=0.01, format="%.3f",
                help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.20.",
            )
        else:
            km = st.number_input(
                "km - rate I -> M", min_value=0.0, value=0.15, step=0.01, format="%.3f",
                help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.15.",
            )
    with rc_cols[1]:
        if is_two_step:
            k2 = st.number_input(
                "k2 - rate X -> M", min_value=0.0, value=0.10, step=0.01, format="%.3f",
                help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.10.",
            )
    with rc_cols[2]:
        kd = st.number_input(
            "kd - degradation / dilution rate", min_value=0.0, value=0.0, step=0.005, format="%.4f",
            help="Suggested range: 0.001-0.05 /sec. Applies to all species. Default: 0 (growth halted).",
        )
    with rc_cols[3]:
        kb = st.number_input(
            "kb - photobleaching rate (M -> B)", min_value=0.0, value=0.02, step=0.005, format="%.4f",
            help="Suggested range: 0.0-0.1 /sec. Illustrative default: 0.02.",
        )
    with rc_cols[4]:
        u = st.number_input(
            "u - production rate", min_value=0.0, value=0.0, step=0.1, format="%.3f",
            help="Suggested: 0 if translation is blocked (e.g. chloramphenicol chase). "
                 "Use a positive value to model ongoing translation.",
        )
    with rc_cols[5]:
        alpha = st.number_input(
            "alpha - fluorescence scaling factor", min_value=0.0, value=1.0, step=0.1,
            help="Suggested: 1.0. Brightness per unit mature protein.",
        )

    st.subheader("Derived parameters")
    if is_two_step:
        a_val = k1 + kd
        c_val = k2 + kd
        b_val = kb + kd
        G3_val = alpha * k1 * k2
        G3I0_val = G3_val * I0
        dp_cols = st.columns(5)
        dp_cols[0].metric("a = k1 + kd", f"{a_val:.4f}")
        dp_cols[1].metric("c = k2 + kd", f"{c_val:.4f}")
        dp_cols[2].metric("b = kb + kd", f"{b_val:.4f}")
        dp_cols[3].metric("G3 = alpha * k1 * k2", f"{G3_val:.4f}")
        dp_cols[4].metric("G3 * I0", f"{G3I0_val:.4f}")
    else:
        a_val = km + kd
        b_val = kb + kd
        G_val = alpha * km
        GI0_val = G_val * I0
        dp_cols = st.columns(4)
        dp_cols[0].metric("a = km + kd", f"{a_val:.4f}")
        dp_cols[1].metric("b = kb + kd", f"{b_val:.4f}")
        dp_cols[2].metric("G = alpha * km", f"{G_val:.4f}")
        dp_cols[3].metric("G * I0", f"{GI0_val:.4f}")

    st.subheader("Simulation time")
    st_cols = st.columns(3)
    with st_cols[0]:
        t_end = st.number_input("End time (sec)", min_value=1.0, value=60.0, step=10.0)
    with st_cols[1]:
        n_points = st.number_input("Number of time points", min_value=10, value=300, step=10)
    with st_cols[2]:
        st.write("")
        run = st.button("Run Simulation", type="primary")

    st.divider()

    if run:
        st.session_state["sim_kb"] = kb
        st.session_state["sim_kd"] = kd
        st.session_state["sim_alpha"] = alpha
        st.session_state["sim_run_version"] = st.session_state.get("sim_run_version", 0) + 1

        t_eval = np.linspace(0, t_end, int(n_points))

        if is_two_step:
            params = {"u": u, "k1": k1, "k2": k2, "kb": kb, "kd": kd}
            y0 = [I0, X0, M0, B0]
            with st.spinner("Running simulation..."):
                sol = solve_ivp(model_2step, (t_eval[0], t_eval[-1]), y0,
                                 t_eval=t_eval, args=(params,), method="RK45")
            I_sim, X_sim, M_sim, B_sim = sol.y
            F_sim = alpha * M_sim
            st.session_state["sim_result"] = {
                "is_two_step": True,
                "t": sol.t, "I": I_sim, "X": X_sim, "M": M_sim, "B": B_sim, "F": F_sim,
            }
        else:
            params = {"u": u, "km": km, "kb": kb, "kd": kd}
            y0 = [I0, M0, B0]
            with st.spinner("Running simulation..."):
                sol = solve_ivp(model_1step, (t_eval[0], t_eval[-1]), y0,
                                 t_eval=t_eval, args=(params,), method="RK45")
            I_sim, M_sim, B_sim = sol.y
            F_sim = alpha * M_sim
            st.session_state["sim_result"] = {
                "is_two_step": False,
                "t": sol.t, "I": I_sim, "M": M_sim, "B": B_sim, "F": F_sim,
            }

    sim_result = st.session_state.get("sim_result")

    if sim_result is None:
        st.info("Set your parameters above and click **Run Simulation**.")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))

        if sim_result["is_two_step"]:
            ax.plot(sim_result["t"], sim_result["I"], label="I (immature)")
            ax.plot(sim_result["t"], sim_result["X"], label="X (intermediate)")
            ax.plot(sim_result["t"], sim_result["M"], label="M (mature)")
            ax.plot(sim_result["t"], sim_result["B"], label="B (bleached)")
            ax.plot(sim_result["t"], sim_result["F"], "--", label="F = alpha * M (fluorescence)", linewidth=2)
            ax.set_title("2-step maturation model")
        else:
            ax.plot(sim_result["t"], sim_result["I"], label="I (immature)")
            ax.plot(sim_result["t"], sim_result["M"], label="M (mature)")
            ax.plot(sim_result["t"], sim_result["B"], label="B (bleached)")
            ax.plot(sim_result["t"], sim_result["F"], "--", label="F = alpha * M (fluorescence)", linewidth=2)
            ax.set_title("1-step maturation model")

        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Amount/ Mean Intensity")
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)

    return {
        "is_two_step": is_two_step,
        "I0": I0, "X0": X0, "M0": M0, "B0": B0,
        "u": u, "alpha": alpha,
        "k1": k1 if is_two_step else None,
        "k2": k2 if is_two_step else None,
        "km": None if is_two_step else km,
        "kb": kb, "kd": kd,
        "t_end": t_end, "n_points": n_points,
    }
