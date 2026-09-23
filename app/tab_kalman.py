# --- Kalman Filter tab: reconstruct I/(X)/M/u from a noisy fluorescence trace
# Ports Kalman_Filter/1-step_Kalman_Filter.py and 2-step_Kalman_Filter.py
# (including their RTS smoother): user enters calibrated ("estimated") rate
# constants (e.g. obtained from Least Squares Fitting), alpha, process
# noise, and the filter's assumed measurement noise sigma_F. Filters the
# noisy trace generated in the **Synthetic Gene Expression** tab (its own,
# possibly different, "true" rate constants and u(t) formula produce that
# trace); this tab's plots compare the forward filter's and the backward
# RTS smoother's reconstructions (from the estimated rate constants here)
# against that tab's ground truth, plus a comparative RMSE table for both
# against every reconstructed quantity. Every run is saved to the Kalman
# Filter History tab (kalman_history.json).

from datetime import datetime

import streamlit as st

from Kalman_Filter_Model import run_kalman_1step, run_kalman_2step
from history_store import append_kalman_entry

from app.shared import render_kalman_result


def render_kalman_tab():
    st.markdown(
        "Runs a Kalman filter (forward pass) plus an RTS smoother (backward "
        "pass, using the whole trace including future measurements -- more "
        "accurate, but only computable after the run finishes) over the "
        "noisy fluorescence trace generated in the **Synthetic Gene "
        "Expression** tab, to reconstruct the immature/(intermediate)/mature "
        "protein pools and the underlying gene expression rate u(t), given "
        "*estimated* rate constants (e.g. from **Least Squares Fitting**) "
        "-- which may differ from that tab's *true* rate constants used to "
        "generate the trace. u is modelled as a random walk driven by "
        "process noise `qu`, not fit directly -- its reconstructed "
        "trajectory is the main output, and typically benefits the most "
        "from smoothing."
    )

    synth = st.session_state.get("synthetic_expression_result")
    if synth is None:
        st.info(
            "Generate a noisy trace in the **Synthetic Gene Expression** "
            "tab first (in this section, above this one)."
        )
        return

    is_two_step = synth["is_two_step"]
    st.caption(
        f"Filtering the {'2-step' if is_two_step else '1-step'} trace "
        f"generated in **Synthetic Gene Expression** "
        f"(n_steps={synth['n_steps']}, dt={synth['dt']:.4g} min, "
        f"u(t) = `{synth['u_expr']}`)."
    )

    st.subheader("Estimated rate constants (e.g. from Least Squares Fitting)")
    rc_cols = st.columns(4)
    with rc_cols[0]:
        if is_two_step:
            km1 = st.number_input(
                "km1 - rate I -> X (/min)", min_value=0.0, value=0.20, step=0.01, format="%.3f",
                key="kalman_km1",
            )
        else:
            km = st.number_input(
                "km - rate I -> M (/min)", min_value=0.0, value=0.15, step=0.01, format="%.3f",
                key="kalman_km",
            )
    with rc_cols[1]:
        if is_two_step:
            km2 = st.number_input(
                "km2 - rate X -> M (/min)", min_value=0.0, value=0.10, step=0.01, format="%.3f",
                key="kalman_km2",
            )
    with rc_cols[2]:
        kb = st.number_input(
            "kb - photobleaching rate (M -> B) (/min)", min_value=0.0, value=0.03, step=0.005, format="%.4f",
            key="kalman_kb",
        )
    with rc_cols[3]:
        kd = st.number_input(
            "kd - degradation / dilution rate (/min)", min_value=0.0, value=0.01, step=0.005, format="%.4f",
            key="kalman_kd",
        )

    alpha = st.number_input(
        "alpha - fluorescence scaling factor", min_value=0.0, value=2.0, step=0.1,
        help="Suggested: obtained by binomial partitioning or from a Least Squares fit.",
        key="kalman_alpha",
    )

    sigma_F = st.number_input(
        "sigma_F - assumed measurement noise std dev", min_value=0.0, value=float(synth["sigma_F"]), step=0.5,
        help="The filter's own assumption about measurement noise (used to build R). "
             "Defaults to the Synthetic Gene Expression tab's true value, but can be "
             "changed to explore a mis-specified filter.",
        key="kalman_sigma_F",
    )

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

    run_button = st.button("Run Kalman Filter", type="primary", key="kalman_run")

    if run_button:
        if is_two_step:
            params = {
                "km1": km1, "km2": km2, "kb": kb, "kd": kd, "alpha": alpha,
                "dt": synth["dt"], "qI": qI, "qX": qX, "qM": qM, "qu": qu, "sigma_F": sigma_F,
            }
        else:
            params = {
                "km": km, "kb": kb, "kd": kd, "alpha": alpha,
                "dt": synth["dt"], "qI": qI, "qM": qM, "qu": qu, "sigma_F": sigma_F,
            }

        try:
            with st.spinner("Running Kalman filter..."):
                filt = (
                    run_kalman_2step(synth["z_n"], params) if is_two_step
                    else run_kalman_1step(synth["z_n"], params)
                )
        except ValueError as e:
            st.error(str(e))
        else:
            result = {
                "is_two_step": is_two_step,
                "t": synth["t"], "z_n": synth["z_n"],
                "true_I": synth["true_I"], "true_M": synth["true_M"], "true_u": synth["true_u"],
                "alpha_true": synth["alpha"], "alpha_est": alpha,
                **filt,
            }
            if is_two_step:
                result["true_X"] = synth["true_X"]

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
