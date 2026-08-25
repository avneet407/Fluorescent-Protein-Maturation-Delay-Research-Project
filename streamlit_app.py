import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from Maturation_Models import (
    model_1step,
    model_2step,
    simulate_1step,
    simulate_2step,
    residuals_1step,
    residuals_2step,
)
from bode_plot import (
    bode_1step,
    bode_2step,
    analytical_cutoff_1step,
    analytical_cutoff_2step,
    numerical_cutoff,
)

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(page_title="Fluorescent Protein Maturation", layout="wide")
st.title("Fluorescent Protein Maturation Delay Model")

sim_tab, data_tab, bode_tab = st.tabs(["Simulation", "Experimental Data", "Bode Plot"])

model_choice = st.sidebar.radio(
    "Maturation model",
    options=["1-step (I -> M)", "2-step (I -> X -> M)"],
)
is_two_step = model_choice.startswith("2")

st.sidebar.header("Initial conditions")
I0 = st.sidebar.number_input(
    "I0 - immature protein at t=0", min_value=0.0, value=100.0, step=10.0,
    help="Suggested: 100. Pool of just-translated protein present when translation is halted.",
)
X0 = 0.0
if is_two_step:
    X0 = st.sidebar.number_input(
        "X0 - intermediate protein at t=0", min_value=0.0, value=0.0, step=10.0,
        help="Suggested: 0. Usually no protein has reached the intermediate stage yet.",
    )
M0 = st.sidebar.number_input(
    "M0 - mature (fluorescent) protein at t=0", min_value=0.0, value=0.0, step=10.0,
    help="Suggested: 0. No protein has finished maturing at t=0.",
)
B0 = st.sidebar.number_input(
    "B0 - bleached protein at t=0", min_value=0.0, value=0.0, step=10.0,
    help="Suggested: 0. No photobleaching has occurred yet.",
)

st.sidebar.header("Rate constants")
if is_two_step:
    k1 = st.sidebar.number_input(
        "k1 - rate I -> X", min_value=0.0, value=0.20, step=0.01, format="%.3f",
        help="Suggested range: 0.05-0.5 /min. Illustrative default: 0.20.",
    )
    k2 = st.sidebar.number_input(
        "k2 - rate X -> M", min_value=0.0, value=0.10, step=0.01, format="%.3f",
        help="Suggested range: 0.05-0.5 /min. Illustrative default: 0.10.",
    )
else:
    km = st.sidebar.number_input(
        "km - rate I -> M", min_value=0.0, value=0.15, step=0.01, format="%.3f",
        help="Suggested range: 0.05-0.5 /min. Illustrative default: 0.15.",
    )

kd = st.sidebar.number_input(
    "kd - degradation / dilution rate", min_value=0.0, value=0.01, step=0.005, format="%.4f",
    help="Suggested range: 0.001-0.05 /min. Applies to all species. Illustrative default: 0.01.",
)
kb = st.sidebar.number_input(
    "kb - photobleaching rate (M -> B)", min_value=0.0, value=0.02, step=0.005, format="%.4f",
    help="Suggested range: 0.0-0.1 /min. Illustrative default: 0.02.",
)
u = st.sidebar.number_input(
    "u - production rate", min_value=0.0, value=0.0, step=0.1, format="%.3f",
    help="Suggested: 0 if translation is blocked (e.g. chloramphenicol chase). "
         "Use a positive value to model ongoing translation.",
)
alpha = st.sidebar.number_input(
    "alpha - fluorescence scaling factor", min_value=0.0, value=1.0, step=0.1,
    help="Suggested: 1.0. Brightness per unit mature protein.",
)

st.sidebar.header("Simulation time")
t_end = st.sidebar.number_input("End time", min_value=1.0, value=60.0, step=10.0)
n_points = st.sidebar.number_input("Number of time points", min_value=10, value=300, step=10)

run = st.sidebar.button("Run Simulation", type="primary")

with sim_tab:
    st.markdown(
        "Simulate a 1-step or 2-step protein maturation model and view the "
        "resulting fluorescence curve. Adjust initial conditions and rate "
        "constants in the sidebar, then click **Run Simulation**."
    )

    if run:
        t_eval = np.linspace(0, t_end, int(n_points))

        fig, ax = plt.subplots(figsize=(9, 5))

        if is_two_step:
            params = {"u": u, "k1": k1, "k2": k2, "kb": kb, "kd": kd}
            y0 = [I0, X0, M0, B0]
            sol = solve_ivp(model_2step, (t_eval[0], t_eval[-1]), y0,
                             t_eval=t_eval, args=(params,), method="RK45")
            I, X, M, B = sol.y
            F = alpha * M

            ax.plot(sol.t, I, label="I (immature)")
            ax.plot(sol.t, X, label="X (intermediate)")
            ax.plot(sol.t, M, label="M (mature)")
            ax.plot(sol.t, B, label="B (bleached)")
            ax.plot(sol.t, F, "--", label="F = alpha * M (fluorescence)", linewidth=2)
            ax.set_title("2-step maturation model")
        else:
            params = {"u": u, "km": km, "kb": kb, "kd": kd}
            y0 = [I0, M0, B0]
            sol = solve_ivp(model_1step, (t_eval[0], t_eval[-1]), y0,
                             t_eval=t_eval, args=(params,), method="RK45")
            I, M, B = sol.y
            F = alpha * M

            ax.plot(sol.t, I, label="I (immature)")
            ax.plot(sol.t, M, label="M (mature)")
            ax.plot(sol.t, B, label="B (bleached)")
            ax.plot(sol.t, F, "--", label="F = alpha * M (fluorescence)", linewidth=2)
            ax.set_title("1-step maturation model")

        ax.set_xlabel("Time")
        ax.set_ylabel("Amount / Fluorescence (a.u.)")
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)
    else:
        st.info("Set your parameters in the sidebar and click **Run Simulation**.")

with data_tab:
    st.markdown(
        "Upload a trench intensity CSV (columns: `Slice, Mean, StdDev, Min, Max`) "
        "to plot **Mean intensity vs. Slice**."
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV file: {e}")
        else:
            if "Slice" not in data.columns or "Mean" not in data.columns:
                st.error(
                    "CSV must contain 'Slice' and 'Mean' columns. "
                    f"Found columns: {list(data.columns)}"
                )
            else:
                fig2, ax2 = plt.subplots(figsize=(9, 5))
                ax2.plot(data["Slice"], data["Mean"], marker="o", markersize=3)
                ax2.set_xlabel("Slice")
                ax2.set_ylabel("Mean intensity")
                ax2.set_title(uploaded_file.name)
                fig2.tight_layout()

                st.pyplot(fig2)
                st.dataframe(data)

                st.divider()
                st.subheader("Fit maturation model parameters to data")
                st.markdown(
                    "Select the region of the trace where the biological maturation "
                    "model applies (e.g. exclude segments dominated by photobleaching "
                    "or imaging transients), then fit model parameters to the "
                    "baseline-corrected fluorescence trace."
                )

                slice_min = float(data["Slice"].min())
                slice_max = float(data["Slice"].max())

                col1, col2 = st.columns(2)
                with col1:
                    slice_start, slice_end = st.slider(
                        "Fitting region (Slice range)",
                        min_value=slice_min, max_value=slice_max,
                        value=(slice_min, slice_max),
                        help="Suggested: the region where fluorescence rises smoothly, "
                             "before/after any transients or bleaching dominate.",
                    )
                with col2:
                    baseline = st.number_input(
                        "Baseline intensity to subtract",
                        value=float(data["Mean"].min()),
                        help="Suggested: the minimum Mean intensity in the trace, or a "
                             "measured background/dark-frame value.",
                    )

                fit_model_choice = st.radio(
                    "Model to fit",
                    options=["1-step (I -> M)", "2-step (I -> X -> M)"],
                    key="fit_model_choice",
                )
                fit_is_two_step = fit_model_choice.startswith("2")

                u_step = st.number_input(
                    "u - production rate (fixed during fit)",
                    min_value=0.0, value=0.0, step=0.1,
                    help="Suggested: 0 if translation was blocked during imaging "
                         "(e.g. chloramphenicol chase).",
                )

                slice_all = data["Slice"].to_numpy(dtype=float)
                F_raw_all = data["Mean"].to_numpy(dtype=float)
                mask = (slice_all >= slice_start) & (slice_all <= slice_end)
                t_raw = slice_all[mask]
                F_raw = F_raw_all[mask]

                if len(t_raw) < 5:
                    st.warning(
                        "Selected slice region contains too few data points "
                        "(need at least 5)."
                    )
                else:
                    t_fit = t_raw - t_raw[0]
                    F_meas = F_raw - baseline

                    with st.expander("Advanced: initial parameter guesses"):
                        I0_guess = st.number_input(
                            "I0 initial guess", value=float(max(np.max(F_meas), 1.0)),
                        )
                        if fit_is_two_step:
                            k1_guess = st.number_input("k1 initial guess", value=0.08, format="%.3f")
                            k2_guess = st.number_input("k2 initial guess", value=0.06, format="%.3f")
                        else:
                            km_guess = st.number_input("km initial guess", value=0.10, format="%.3f")
                        kb_guess = st.number_input("kb initial guess", value=0.01, format="%.4f")
                        kd_guess = st.number_input("kd initial guess", value=0.005, format="%.4f")
                        alpha_guess = st.number_input("alpha initial guess", value=1.0, format="%.3f")

                    fit_button = st.button("Fit Parameters", type="primary")

                    if fit_button:
                        fixed = {"u": u_step}

                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        ax3.plot(t_raw, F_raw, "k--", linewidth=1.5, label="Raw selected data")
                        ax3.plot(t_raw, F_meas, "o", markersize=4, label="Baseline-corrected data")

                        if fit_is_two_step:
                            x0 = np.array([I0_guess, k1_guess, k2_guess, kb_guess, kd_guess, alpha_guess])
                            bounds = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 5.0, 5.0, 10.0])

                            fit = least_squares(residuals_2step, x0, bounds=bounds, args=(t_fit, F_meas, fixed))
                            I0_hat, k1_hat, k2_hat, kb_hat, kd_hat, alpha_hat = fit.x

                            fitted_params = {
                                "u": u_step, "k1": k1_hat, "k2": k2_hat,
                                "kb": kb_hat, "kd": kd_hat, "alpha": alpha_hat,
                            }
                            _, I_fit, X_fit, M_fit, B_fit, F_fit = simulate_2step(
                                t_fit, fitted_params, I0=I0_hat, X0=0.0, M0=0.0, B0=0.0
                            )

                            st.success("Fit complete (2-step model)")
                            res_col1, res_col2, res_col3 = st.columns(3)
                            res_col1.metric("I0", f"{I0_hat:.4f}")
                            res_col1.metric("k1", f"{k1_hat:.4f}")
                            res_col2.metric("k2", f"{k2_hat:.4f}")
                            res_col2.metric("kb", f"{kb_hat:.4f}")
                            res_col3.metric("kd", f"{kd_hat:.4f}")
                            res_col3.metric("alpha", f"{alpha_hat:.4f}")

                            ax3.plot(t_raw, F_fit, "-", linewidth=2, label="2-step model fit")
                            ax3.set_title("2-step fit on selected slice region")
                        else:
                            x0 = np.array([I0_guess, km_guess, kb_guess, kd_guess, alpha_guess])
                            bounds = ([0.0, 0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 5.0, 10.0])

                            fit = least_squares(residuals_1step, x0, bounds=bounds, args=(t_fit, F_meas, fixed))
                            I0_hat, km_hat, kb_hat, kd_hat, alpha_hat = fit.x

                            fitted_params = {
                                "u": u_step, "km": km_hat,
                                "kb": kb_hat, "kd": kd_hat, "alpha": alpha_hat,
                            }
                            _, I_fit, M_fit, B_fit, F_fit = simulate_1step(
                                t_fit, fitted_params, I0=I0_hat, M0=0.0, B0=0.0
                            )

                            st.success("Fit complete (1-step model)")
                            res_col1, res_col2, res_col3 = st.columns(3)
                            res_col1.metric("I0", f"{I0_hat:.4f}")
                            res_col1.metric("km", f"{km_hat:.4f}")
                            res_col2.metric("kb", f"{kb_hat:.4f}")
                            res_col2.metric("kd", f"{kd_hat:.4f}")
                            res_col3.metric("alpha", f"{alpha_hat:.4f}")

                            ax3.plot(t_raw, F_fit, "-", linewidth=2, label="1-step model fit")
                            ax3.set_title("1-step fit on selected slice region")

                        st.caption(f"Baseline subtracted = {baseline:.4f}")

                        residuals = fit.fun
                        sse = float(np.sum(residuals ** 2))
                        rmse = float(np.sqrt(np.mean(residuals ** 2)))
                        ss_tot = float(np.sum((F_meas - np.mean(F_meas)) ** 2))
                        r_squared = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")

                        st.markdown("**Goodness of fit**")
                        err_col1, err_col2, err_col3 = st.columns(3)
                        err_col1.metric("SSE (least-squares cost)", f"{sse:.4f}")
                        err_col2.metric("RMSE", f"{rmse:.4f}")
                        err_col3.metric("R²", f"{r_squared:.4f}")
                        if not fit.success:
                            st.warning(f"Solver did not fully converge: {fit.message}")

                        ax3.set_xlabel("Slice")
                        ax3.set_ylabel("Mean intensity")
                        ax3.legend()
                        fig3.tight_layout()
                        st.pyplot(fig3)
    else:
        st.info("Upload a CSV file to see the Mean intensity vs. Slice plot.")

with bode_tab:
    st.markdown(
        "Frequency response of the maturation model currently selected in the "
        "sidebar, using its rate constants (`km`/`k1`,`k2`, `kb`, `kd`, `alpha`). "
        "This shows how strongly the reporter attenuates and delays gene "
        "expression fluctuations at each frequency, and gives a cutoff "
        "frequency above which dynamics are lost to maturation/bleaching."
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
        w = np.logspace(w_start_exp, w_end_exp, int(w_points))

        if is_two_step:
            w, mag, phase = bode_2step(alpha, k1, k2, kb, kd, w)
            title_suffix = "2-step maturation model"
        else:
            w, mag, phase = bode_1step(alpha, km, kb, kd, w)
            title_suffix = "1-step maturation model"

        fig4, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6))

        ax_mag.semilogx(w, mag)
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_xlabel("Frequency (rad/time)")
        ax_mag.set_title(f"Bode Plot: {title_suffix}")
        ax_mag.grid(True, which="both", linestyle="--", alpha=0.5)

        ax_phase.semilogx(w, phase)
        ax_phase.set_ylabel("Phase (degrees)")
        ax_phase.set_xlabel("Frequency (rad/time)")
        ax_phase.grid(True, which="both", linestyle="--", alpha=0.5)

        fig4.tight_layout()
        st.pyplot(fig4)

        wc_numerical = numerical_cutoff(w, mag)
        cutoff_col1, cutoff_col2 = st.columns(2)
        if wc_numerical is not None:
            cutoff_col1.metric("Numerical -3 dB cutoff (rad/time)", f"{wc_numerical:.6f}")
        else:
            cutoff_col1.warning("Cutoff not reached within the selected frequency range.")

        if is_two_step:
            wc_analytical = analytical_cutoff_2step(k1, k2, kb, kd)
        else:
            wc_analytical = analytical_cutoff_1step(km, kb, kd)

        if wc_analytical is not None:
            cutoff_col2.metric("Analytical -3 dB cutoff (rad/time)", f"{wc_analytical:.6f}")
        else:
            cutoff_col2.warning("No positive real cutoff root found.")
    else:
        st.info("Set the frequency range and click **Plot Bode Response**.")
