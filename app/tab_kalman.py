# --- Kalman Filter tab: reconstruct I/(X)/M/u from a noisy fluorescence trace
# Ports Kalman_Filter/1-step_Kalman_Filter.py and 2-step_Kalman_Filter.py:
# user enters calibrated rate constants (e.g. obtained from Least Squares
# Fitting), alpha, the measurement interval dt, process noise, and
# measurement noise sigma_F, plus the synthetic sinusoidal gene-expression
# signal used to generate a trace to filter. Every run is saved to the
# Kalman Filter History tab (kalman_history.json).

from datetime import datetime

import streamlit as st

from Kalman_Filter_Model import run_kalman_1step, run_kalman_2step
from history_store import append_kalman_entry

from app.shared import render_kalman_result


def render_kalman_tab():
    st.markdown(
        "Runs a Kalman filter over a synthetic noisy fluorescence trace to "
        "reconstruct the immature/(intermediate)/mature protein pools and "
        "the underlying gene expression rate u(t), given calibrated rate "
        "constants (e.g. from **Least Squares Fitting**). u is modelled as a "
        "random walk driven by process noise `qu`, not fit directly -- its "
        "reconstructed trajectory is the Kalman filter's main output."
    )

    model_choice = st.radio(
        "Maturation model", options=["1-step (I -> M)", "2-step (I -> X -> M)"],
        horizontal=True, key="kalman_model",
    )
    is_two_step = model_choice.startswith("2")

    st.subheader("Calibrated rate constants")
    rc_cols = st.columns(4)
    with rc_cols[0]:
        if is_two_step:
            km1 = st.number_input(
                "km1 - rate I -> X", min_value=0.0, value=0.20, step=0.01, format="%.3f",
                key="kalman_km1",
            )
        else:
            km = st.number_input(
                "km - rate I -> M", min_value=0.0, value=0.15, step=0.01, format="%.3f",
                key="kalman_km",
            )
    with rc_cols[1]:
        if is_two_step:
            km2 = st.number_input(
                "km2 - rate X -> M", min_value=0.0, value=0.10, step=0.01, format="%.3f",
                key="kalman_km2",
            )
    with rc_cols[2]:
        kb = st.number_input(
            "kb - photobleaching rate (M -> B)", min_value=0.0, value=0.03, step=0.005, format="%.4f",
            key="kalman_kb",
        )
    with rc_cols[3]:
        kd = st.number_input(
            "kd - degradation / dilution rate", min_value=0.0, value=0.01, step=0.005, format="%.4f",
            key="kalman_kd",
        )

    alpha = st.number_input(
        "alpha - fluorescence scaling factor", min_value=0.0, value=2.0, step=0.1,
        help="Suggested: obtained by binomial partitioning or from a Least Squares fit.",
        key="kalman_alpha",
    )

    st.subheader("Measurement timing and noise")
    noise_cols = st.columns(4)
    with noise_cols[0]:
        dt = st.number_input(
            "dt - time between measurements", min_value=1e-6, value=5.0, step=0.5,
            key="kalman_dt",
        )
    with noise_cols[1]:
        n_steps = st.number_input(
            "n_steps - number of measurements", min_value=2, value=60, step=1,
            key="kalman_n_steps",
        )
    with noise_cols[2]:
        sigma_F = st.number_input(
            "sigma_F - measurement noise std dev", min_value=0.0, value=5.0, step=0.5,
            key="kalman_sigma_F",
        )
    with noise_cols[3]:
        st.write("")

    st.subheader("Process noise (diagonal of Qc)")
    if is_two_step:
        q_cols = st.columns(4)
        with q_cols[0]:
            qI = st.number_input("qI", min_value=0.0, value=1e-4, format="%.6f", key="kalman_qI")
        with q_cols[1]:
            qX = st.number_input("qX", min_value=0.0, value=1e-4, format="%.6f", key="kalman_qX")
        with q_cols[2]:
            qM = st.number_input("qM", min_value=0.0, value=1e-4, format="%.6f", key="kalman_qM")
        with q_cols[3]:
            qu = st.number_input(
                "qu", min_value=0.0, value=1.0, format="%.4f", key="kalman_qu",
                help="Large qu: u is allowed to jump a lot between measurements. "
                     "Small qu: u is expected to barely move, so large jumps in the "
                     "true trajectory are treated as statistically unlikely.",
            )
    else:
        q_cols = st.columns(3)
        with q_cols[0]:
            qI = st.number_input("qI", min_value=0.0, value=1e-4, format="%.6f", key="kalman_qI")
        with q_cols[1]:
            qM = st.number_input("qM", min_value=0.0, value=1e-4, format="%.6f", key="kalman_qM")
        with q_cols[2]:
            qu = st.number_input(
                "qu", min_value=0.0, value=1.0, format="%.4f", key="kalman_qu",
                help="Large qu: u is allowed to jump a lot between measurements. "
                     "Small qu: u is expected to barely move, so large jumps in the "
                     "true trajectory are treated as statistically unlikely.",
            )

    st.subheader("Synthetic gene expression signal (to generate the trace to filter)")
    st.caption(
        "true_u(t) = u_base + u_amp * sin(2*pi * t / (n_steps*dt) * u_freq_cycles). "
        "Only used to generate the synthetic ground truth and noisy measurements below "
        "-- the filter itself never sees true_u."
    )
    u_cols = st.columns(3)
    with u_cols[0]:
        u_base = st.number_input("u_base", value=5.0, step=0.5, key="kalman_u_base")
    with u_cols[1]:
        u_amp = st.number_input("u_amp", value=3.0, step=0.5, key="kalman_u_amp")
    with u_cols[2]:
        u_freq_cycles = st.number_input("u_freq_cycles", value=2.0, step=0.5, key="kalman_u_freq")

    use_seed = st.checkbox("Fix random seed (reproducible noise)", value=False, key="kalman_use_seed")
    seed = None
    if use_seed:
        seed = int(st.number_input("Random seed", min_value=0, value=0, step=1, key="kalman_seed_val"))

    run_button = st.button("Run Kalman Filter", type="primary", key="kalman_run")

    if run_button:
        if is_two_step:
            params = {
                "km1": km1, "km2": km2, "kb": kb, "kd": kd, "alpha": alpha,
                "dt": dt, "n_steps": n_steps, "qI": qI, "qX": qX, "qM": qM, "qu": qu,
                "sigma_F": sigma_F, "u_base": u_base, "u_amp": u_amp,
                "u_freq_cycles": u_freq_cycles, "seed": seed,
            }
        else:
            params = {
                "km": km, "kb": kb, "kd": kd, "alpha": alpha,
                "dt": dt, "n_steps": n_steps, "qI": qI, "qM": qM, "qu": qu,
                "sigma_F": sigma_F, "u_base": u_base, "u_amp": u_amp,
                "u_freq_cycles": u_freq_cycles, "seed": seed,
            }

        try:
            with st.spinner("Running Kalman filter..."):
                result = run_kalman_2step(params) if is_two_step else run_kalman_1step(params)
        except ValueError as e:
            st.error(str(e))
        else:
            st.session_state["kalman_result"] = result
            append_kalman_entry({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "params": params,
                "result": result,
            })

    kalman_result = st.session_state.get("kalman_result")
    if kalman_result is None:
        st.info("Set your parameters above and click **Run Kalman Filter**.")
    else:
        st.success(
            f"Kalman filter complete ({'2-step' if kalman_result['is_two_step'] else '1-step'} model). "
            "Saved to **Kalman Filter History**."
        )
        render_kalman_result(kalman_result)
