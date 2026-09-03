# --- Variable Bleaching tab -------------------------------------------
# Fixes kd at 0 (growth halted) and lets the user add multiple kb
# (photobleaching rate) values, each with its own label/key, then overlays
# the resulting fluorescence F(t) curve for every kb value on one graph,
# with km/k1,k2, alpha, u, and initial conditions held fixed across curves.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from Maturation_Models import simulate_1step, simulate_2step
from gaussian_noise import simulate_1step_noisy, simulate_2step_noisy, add_measurement_noise


def render_variable_bleaching_tab():
    st.markdown(
        "Simulates the full maturation model with kd fixed at 0 (growth "
        "halted), overlaying the fluorescence F(t) curve for several kb "
        "(photobleaching rate) values on one graph. All other rate "
        "constants and initial conditions below are shared across every "
        "kb value."
    )

    st.session_state.setdefault("vb_kb_sets", [])
    kb_sets = st.session_state["vb_kb_sets"]

    model_choice = st.radio(
        "Maturation model",
        options=["1-step (I -> M)", "2-step (I -> X -> M)"],
        horizontal=True, key="vb_model",
    )
    is_two_step = model_choice.startswith("2")

    st.subheader("Initial conditions")
    ic_cols = st.columns(4)
    with ic_cols[0]:
        I0 = st.number_input(
            "I0 - immature protein at t=0", min_value=0.0, value=100.0, step=10.0,
            help="Suggested: 100. Pool of just-translated protein present when translation is halted.",
            key="vb_I0",
        )
    X0 = 0.0
    with ic_cols[1]:
        if is_two_step:
            X0 = st.number_input(
                "X0 - intermediate protein at t=0", min_value=0.0, value=0.0, step=10.0,
                help="Suggested: 0. Usually no protein has reached the intermediate stage yet.",
                key="vb_X0",
            )
    with ic_cols[2]:
        M0 = st.number_input(
            "M0 - mature (fluorescent) protein at t=0", min_value=0.0, value=0.0, step=10.0,
            help="Suggested: 0. No protein has finished maturing at t=0.",
            key="vb_M0",
        )
    with ic_cols[3]:
        B0 = st.number_input(
            "B0 - bleached protein at t=0", min_value=0.0, value=0.0, step=10.0,
            help="Suggested: 0. No photobleaching has occurred yet.",
            key="vb_B0",
        )

    st.subheader("Fixed rate constants (kd = 0, growth halted)")
    rc_cols = st.columns(4)
    with rc_cols[0]:
        if is_two_step:
            k1 = st.number_input(
                "k1 - rate I -> X", min_value=0.0, value=0.20, step=0.01, format="%.3f",
                help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.20.",
                key="vb_k1",
            )
        else:
            km = st.number_input(
                "km - rate I -> M", min_value=0.0, value=0.15, step=0.01, format="%.3f",
                help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.15.",
                key="vb_km",
            )
    with rc_cols[1]:
        if is_two_step:
            k2 = st.number_input(
                "k2 - rate X -> M", min_value=0.0, value=0.10, step=0.01, format="%.3f",
                help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.10.",
                key="vb_k2",
            )
    with rc_cols[2]:
        u = st.number_input(
            "u - production rate", min_value=0.0, value=0.0, step=0.1, format="%.3f",
            help="Suggested: 0 if translation is blocked (e.g. chloramphenicol chase). "
                 "Use a positive value to model ongoing translation.",
            key="vb_u",
        )
    with rc_cols[3]:
        alpha = st.number_input(
            "alpha - fluorescence scaling factor", min_value=0.0, value=1.0, step=0.1,
            help="Suggested: 1.0. Brightness per unit mature protein.",
            key="vb_alpha",
        )

    kd = 0.0
    st.caption("kd - degradation / dilution rate: fixed at 0 (growth halted).")

    st.subheader("Simulation time")
    st_cols = st.columns(2)
    with st_cols[0]:
        t_end = st.number_input("End time (sec)", min_value=1.0, value=60.0, step=10.0, key="vb_t_end")
    with st_cols[1]:
        n_points = st.number_input(
            "Number of time points", min_value=10, value=300, step=10, key="vb_n_points",
        )

    st.divider()
    st.subheader("Add a kb value")
    add_cols = st.columns([2, 1])
    with add_cols[0]:
        new_label = st.text_input(
            "Label (optional)", key="vb_new_label",
            placeholder=f"kb = ... (Set {len(kb_sets) + 1})",
        )
    with add_cols[1]:
        new_kb = st.number_input(
            "kb - photobleaching rate (M -> B)", min_value=0.0, value=0.02, step=0.005, format="%.4f",
            key="vb_new_kb",
        )

    if st.button("Run (add to graph)", type="primary", key="vb_add_kb"):
        label = new_label.strip() or f"kb = {new_kb:.4f}"
        kb_sets.append({"label": label, "kb": new_kb})

    if kb_sets:
        with st.expander(f"kb values on graph ({len(kb_sets)})", expanded=False):
            for i, ks in enumerate(kb_sets):
                row_cols = st.columns([5, 1])
                with row_cols[0]:
                    st.write(f"**{ks['label']}** — kb={ks['kb']:.4f}")
                with row_cols[1]:
                    if st.button("Remove", key=f"vb_remove_{i}"):
                        kb_sets.pop(i)
                        st.rerun()
            if st.button("Clear all", key="vb_clear_all"):
                st.session_state["vb_kb_sets"] = []
                st.rerun()

    if not kb_sets:
        st.info("Add at least one kb value above and click **Run** to see the fluorescence curves.")
        return

    t_eval = np.linspace(0, t_end, int(n_points))

    fig, ax = plt.subplots(figsize=(9, 5))
    for ks in kb_sets:
        if is_two_step:
            params = {"u": u, "k1": k1, "k2": k2, "kb": ks["kb"], "kd": kd, "alpha": alpha}
            _, _, _, _, _, F = simulate_2step(t_eval, params, I0=I0, X0=X0, M0=M0, B0=B0)
        else:
            params = {"u": u, "km": km, "kb": ks["kb"], "kd": kd, "alpha": alpha}
            _, _, _, _, F = simulate_1step(t_eval, params, I0=I0, M0=M0, B0=B0)
        ax.plot(t_eval, F, label=ks["label"])

    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("F = alpha * M (fluorescence)")
    ax.set_title(
        ("2-step" if is_two_step else "1-step")
        + " maturation model — fluorescence across kb values (kd = 0)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()

    st.pyplot(fig)

    st.divider()
    st.subheader("Generate synthetic data")
    st.markdown(
        "Adds Gaussian noise to km (or k1/k2) and kb (drawn fresh at every "
        "simulated time step, before the model is integrated forward), "
        "and/or Gaussian measurement noise added directly to the resulting "
        "Mean intensity trace, applied to every kb value's fluorescence "
        "curve above."
    )

    if is_two_step:
        noise_cols = st.columns(3)
        with noise_cols[0]:
            k1_noise_std = st.number_input(
                "k1 noise std dev", min_value=0.0, value=0.02, step=0.01, format="%.4f",
                help="Standard deviation of the Gaussian noise added to k1 at each time step.",
                key="vb_k1_noise_std",
            )
        with noise_cols[1]:
            k2_noise_std = st.number_input(
                "k2 noise std dev", min_value=0.0, value=0.02, step=0.01, format="%.4f",
                help="Standard deviation of the Gaussian noise added to k2 at each time step.",
                key="vb_k2_noise_std",
            )
        with noise_cols[2]:
            kb_noise_std = st.number_input(
                "kb noise std dev", min_value=0.0, value=0.005, step=0.001, format="%.4f",
                help="Standard deviation of the Gaussian noise added to kb at each time step.",
                key="vb_kb_noise_std",
            )
    else:
        noise_cols = st.columns(2)
        with noise_cols[0]:
            km_noise_std = st.number_input(
                "km noise std dev", min_value=0.0, value=0.02, step=0.01, format="%.4f",
                help="Standard deviation of the Gaussian noise added to km at each time step.",
                key="vb_km_noise_std",
            )
        with noise_cols[1]:
            kb_noise_std = st.number_input(
                "kb noise std dev", min_value=0.0, value=0.005, step=0.001, format="%.4f",
                help="Standard deviation of the Gaussian noise added to kb at each time step.",
                key="vb_kb_noise_std",
            )

    measurement_noise_std = st.number_input(
        "Measurement noise std dev (Mean intensity)",
        min_value=0.0, value=0.0, step=0.5,
        help="Standard deviation of independent Gaussian noise added directly "
             "to each simulated Mean intensity trace, on top of any "
             "rate-constant noise above. Set to 0 to disable.",
        key="vb_measurement_noise_std",
    )

    use_seed = st.checkbox("Fix random seed (reproducible noise)", value=False, key="vb_use_seed")
    seed = None
    if use_seed:
        seed = int(st.number_input("Random seed", min_value=0, value=0, step=1, key="vb_seed_val"))

    generate_button = st.button("Generate Synthetic Data", type="primary", key="vb_generate_synthetic")

    if generate_button:
        synthetic_results = []
        with st.spinner("Generating synthetic data..."):
            for idx, ks in enumerate(kb_sets):
                curve_seed = None if seed is None else seed + 2 * idx
                measurement_seed = None if seed is None else seed + 2 * idx + 1
                if is_two_step:
                    params_syn = {"u": u, "k1": k1, "k2": k2, "kb": ks["kb"], "kd": kd, "alpha": alpha}
                    _, _, _, _, _, F_syn = simulate_2step_noisy(
                        t_eval, params_syn, I0=I0, X0=X0, M0=M0, B0=B0,
                        k1_std=k1_noise_std, k2_std=k2_noise_std, kb_std=kb_noise_std,
                        seed=curve_seed,
                    )
                else:
                    params_syn = {"u": u, "km": km, "kb": ks["kb"], "kd": kd, "alpha": alpha}
                    _, _, _, _, F_syn = simulate_1step_noisy(
                        t_eval, params_syn, I0=I0, M0=M0, B0=B0,
                        km_std=km_noise_std, kb_std=kb_noise_std,
                        seed=curve_seed,
                    )
                F_syn = add_measurement_noise(F_syn, measurement_noise_std, seed=measurement_seed)
                synthetic_results.append({"label": ks["label"], "kb": ks["kb"], "t": t_eval, "F": F_syn})

        st.session_state["vb_synthetic_results"] = synthetic_results

    synthetic_results = st.session_state.get("vb_synthetic_results")
    if synthetic_results is None:
        st.info(
            "Click **Generate Synthetic Data** to create noisy fluorescence "
            "traces for the kb values above."
        )
    else:
        fig_syn, ax_syn = plt.subplots(figsize=(9, 5))
        for sr in synthetic_results:
            ax_syn.plot(sr["t"], sr["F"], marker="o", markersize=2, linewidth=1, label=sr["label"])
        ax_syn.set_xlabel("Time (sec)")
        ax_syn.set_ylabel("Mean intensity (synthetic)")
        ax_syn.set_title("Synthetic data across kb values")
        ax_syn.legend(fontsize=8)
        fig_syn.tight_layout()
        st.pyplot(fig_syn)

        combined_rows = [
            {"Set": sr["label"], "kb": sr["kb"], "Time": t_val, "Mean": f_val}
            for sr in synthetic_results
            for t_val, f_val in zip(sr["t"], sr["F"])
        ]
        st.dataframe(pd.DataFrame(combined_rows))
