# --- Synthetic Gene Expression tab: user-defined u(t) -> noisy trace -----
# Generates ground-truth I/(X)/M trajectories and a noisy fluorescence trace
# from a user-typed u(t) formula plus "true" rate constants, for the Kalman
# Filter tab to filter. The Kalman Filter tab uses its own, potentially
# different, calibrated/least-squares rate constants against this same
# noisy trace, so its plots can compare the filter's reconstruction against
# this tab's ground truth.

import streamlit as st

from synthetic_expression import simulate_true_1step, simulate_true_2step, EXPRESSION_HELP
from gaussian_noise import add_measurement_noise

from app.shared import render_synthetic_expression_result


def render_synthetic_expression_tab():
    st.markdown(
        "Define a ground-truth gene expression signal u(t) as a math "
        "formula, plus the *true* rate constants driving the maturation "
        "model, to generate a noisy fluorescence trace (I, X, M all start "
        "at 0). The **Kalman Filter** tab filters this same noisy trace "
        "using its own, separately entered, calibrated/least-squares rate "
        "constants -- so its plots can compare the filter's reconstruction "
        "against this tab's ground truth, even when the two sets of rate "
        "constants don't exactly match."
    )

    model_choice = st.radio(
        "Maturation model", options=["1-step (I -> M)", "2-step (I -> X -> M)"],
        horizontal=True, key="synexpr_model",
    )
    is_two_step = model_choice.startswith("2")

    st.subheader("True rate constants")
    rc_cols = st.columns(4)
    with rc_cols[0]:
        if is_two_step:
            km1 = st.number_input(
                "km1 - rate I -> X (/min)", min_value=0.0, value=0.20, step=0.01, format="%.3f",
                key="synexpr_km1",
            )
        else:
            km = st.number_input(
                "km - rate I -> M (/min)", min_value=0.0, value=0.15, step=0.01, format="%.3f",
                key="synexpr_km",
            )
    with rc_cols[1]:
        if is_two_step:
            km2 = st.number_input(
                "km2 - rate X -> M (/min)", min_value=0.0, value=0.10, step=0.01, format="%.3f",
                key="synexpr_km2",
            )
    with rc_cols[2]:
        kb = st.number_input(
            "kb - photobleaching rate (M -> B) (/min)", min_value=0.0, value=0.03, step=0.005, format="%.4f",
            key="synexpr_kb",
        )
    with rc_cols[3]:
        kd = st.number_input(
            "kd - degradation / dilution rate (/min)", min_value=0.0, value=0.01, step=0.005, format="%.4f",
            key="synexpr_kd",
        )

    alpha = st.number_input(
        "alpha - fluorescence scaling factor", min_value=0.0, value=2.0, step=0.1,
        key="synexpr_alpha",
    )

    st.subheader("Measurement timing and noise")
    noise_cols = st.columns(3)
    with noise_cols[0]:
        dt = st.number_input(
            "dt - time between measurements (min)", min_value=1e-6, value=5.0, step=0.5,
            key="synexpr_dt",
        )
    with noise_cols[1]:
        n_steps = st.number_input(
            "n_steps - number of measurements", min_value=2, value=60, step=1,
            key="synexpr_n_steps",
        )
    with noise_cols[2]:
        sigma_F = st.number_input(
            "sigma_F - measurement noise std dev", min_value=0.0, value=5.0, step=0.5,
            key="synexpr_sigma_F",
        )

    st.subheader("Gene expression signal u(t)")
    u_expr = st.text_input(
        "u(t) equation", value="5 + 3*sin(2*pi*t/300*2)", key="synexpr_u_expr",
        help=EXPRESSION_HELP,
    )

    use_seed = st.checkbox("Fix random seed (reproducible noise)", value=False, key="synexpr_use_seed")
    seed = None
    if use_seed:
        seed = int(st.number_input("Random seed", min_value=0, value=0, step=1, key="synexpr_seed_val"))

    generate_button = st.button("Generate Synthetic Data", type="primary", key="synexpr_generate")

    if generate_button:
        try:
            if is_two_step:
                sim = simulate_true_2step(u_expr, km1, km2, kb, kd, alpha, dt, int(n_steps))
            else:
                sim = simulate_true_1step(u_expr, km, kb, kd, alpha, dt, int(n_steps))
        except ValueError as e:
            st.error(str(e))
        else:
            z_n = add_measurement_noise(sim["true_F"], sigma_F, seed=seed)

            result = {
                "is_two_step": is_two_step,
                "t": sim["t"], "dt": float(dt), "n_steps": int(n_steps),
                "u_expr": u_expr,
                "true_u": sim["true_u"], "true_I": sim["true_I"], "true_M": sim["true_M"],
                "true_F": sim["true_F"], "z_n": z_n,
                "km": None if is_two_step else km,
                "km1": km1 if is_two_step else None,
                "km2": km2 if is_two_step else None,
                "kb": kb, "kd": kd, "alpha": alpha,
                "sigma_F": sigma_F, "seed": seed,
            }
            if is_two_step:
                result["true_X"] = sim["true_X"]

            st.session_state["synthetic_expression_result"] = result

    result = st.session_state.get("synthetic_expression_result")
    if result is None:
        st.info("Set your parameters above and click **Generate Synthetic Data**.")
        return

    st.success(
        f"Synthetic data generated ({'2-step' if result['is_two_step'] else '1-step'} model, "
        f"{result['n_steps']} points, dt={result['dt']:.4g}). Available to the "
        "**Kalman Filter** tab as the trace to filter."
    )
    render_synthetic_expression_result(result)
