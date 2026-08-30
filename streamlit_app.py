from datetime import datetime

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
from gaussian_noise import (
    simulate_1step_noisy,
    simulate_2step_noisy,
    add_measurement_noise,
)
from multi_start_fit import run_multi_start
from multi_start_plots import plot_histograms
from history_store import load_history, append_entry, clear_history, delete_entry
from profile_likelihood import (
    profile_raw_parameter,
    profile_derived_quantity,
    DERIVED_QUANTITY_FORMULAS,
    compute_true_values,
)


def render_multi_start_results(results_df, param_names, derived_names, true_values, include_nonconverged):
    """Render the histograms + summary table for one multi-start fit result.

    Shared by the Least Squares Fitting tab (right after a run) and the
    History tab (replaying a stored past run).
    """
    n_converged = int(results_df["converged"].sum())
    n_total = len(results_df)
    st.markdown(f"**Multi-start fit complete: {n_converged} / {n_total} runs converged**")

    plot_df = results_df if include_nonconverged else results_df[results_df["converged"]]

    if len(plot_df) == 0:
        st.warning(
            "No runs to display (no converged runs, and non-converged "
            "runs are excluded)."
        )
        return

    st.markdown("**Raw fitted parameters across runs**")
    fig_raw = plot_histograms(plot_df, param_names, true_values, color="tab:blue")
    st.pyplot(fig_raw)

    st.markdown("**Derived quantities across runs**")
    fig_der = plot_histograms(plot_df, derived_names, true_values, color="tab:purple")
    st.pyplot(fig_der)

    st.markdown("**Summary statistics across runs (mean, std, coefficient of variation)**")
    summary_rows = []
    for name in param_names + derived_names:
        vals = plot_df[name].to_numpy(dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = std / mean if mean != 0 else float("nan")
        summary_rows.append({
            "Parameter": name,
            "Synthetic input": true_values.get(name, float("nan")),
            "Mean": mean,
            "Std dev": std,
            "CV": cv,
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df.set_index("Parameter"))

    with st.expander("All multi-start fit results (raw table)"):
        st.markdown(
            "Each run shows two rows: its final **Fitted** values, and the "
            "**Initial guess** it started from directly below."
        )
        detail_cols = ["run", "Type"] + param_names + ["cost", "converged", "message", "nfev"]
        detail_rows = []
        for _, r in results_df.iterrows():
            detail_rows.append({
                "run": int(r["run"]),
                "Type": "Fitted",
                **{name: r[name] for name in param_names},
                "cost": f"{r['cost']:.6g}",
                "converged": str(bool(r["converged"])),
                "message": str(r["message"]),
                "nfev": str(int(r["nfev"])),
            })
            detail_rows.append({
                "run": int(r["run"]),
                "Type": "Initial guess",
                **{name: r[f"{name}_init"] for name in param_names},
                "cost": "n/a",
                "converged": "n/a",
                "message": "n/a",
                "nfev": "n/a",
            })
        # cost/converged/message/nfev are formatted as strings above (rather than
        # left as their native numeric/bool dtype) so that mixing them with the
        # "n/a" placeholder on Initial guess rows doesn't create a column with
        # inconsistent types, which Streamlit/PyArrow cannot serialize.
        detail_df = pd.DataFrame(detail_rows, columns=detail_cols)
        st.dataframe(detail_df, hide_index=True)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(page_title="Fluorescent Protein Maturation", layout="wide")
st.title("Fluorescent Protein Maturation Delay Model")

sim_tab, data_tab, bode_tab, history_tab, profile_tab = st.tabs(
    ["Simulation", "Least Squares Fitting", "Bode Plot", "History", "Profile Likelihood"]
)

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
        help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.20.",
    )
    k2 = st.sidebar.number_input(
        "k2 - rate X -> M", min_value=0.0, value=0.10, step=0.01, format="%.3f",
        help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.10.",
    )
else:
    km = st.sidebar.number_input(
        "km - rate I -> M", min_value=0.0, value=0.15, step=0.01, format="%.3f",
        help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.15.",
    )

kd = st.sidebar.number_input(
    "kd - degradation / dilution rate", min_value=0.0, value=0.01, step=0.005, format="%.4f",
    help="Suggested range: 0.001-0.05 /sec. Applies to all species. Illustrative default: 0.01.",
)
kb = st.sidebar.number_input(
    "kb - photobleaching rate (M -> B)", min_value=0.0, value=0.02, step=0.005, format="%.4f",
    help="Suggested range: 0.0-0.1 /sec. Illustrative default: 0.02.",
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
t_end = st.sidebar.number_input("End time (sec)", min_value=1.0, value=60.0, step=10.0)
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
            with st.spinner("Running simulation..."):
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
            with st.spinner("Running simulation..."):
                sol = solve_ivp(model_1step, (t_eval[0], t_eval[-1]), y0,
                                 t_eval=t_eval, args=(params,), method="RK45")
            I, M, B = sol.y
            F = alpha * M

            ax.plot(sol.t, I, label="I (immature)")
            ax.plot(sol.t, M, label="M (mature)")
            ax.plot(sol.t, B, label="B (bleached)")
            ax.plot(sol.t, F, "--", label="F = alpha * M (fluorescence)", linewidth=2)
            ax.set_title("1-step maturation model")

        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Amount/ Mean Intensity")
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)
    else:
        st.info("Set your parameters in the sidebar and click **Run Simulation**.")

with data_tab:
    st.markdown(
        "Fit maturation model parameters to a fluorescence trace using "
        "least-squares optimization. Either upload real experimental data "
        "from a trench intensity CSV, or generate synthetic noisy data by "
        "perturbing the simulation's rate constants with Gaussian noise and "
        "re-integrating the model, then fit against it."
    )

    data_source = st.radio(
        "Data source",
        options=["Upload experimental CSV", "Generate synthetic data from simulation"],
    )

    data = None
    data_label = ""

    if data_source == "Upload experimental CSV":
        st.markdown(
            "Upload a trench intensity CSV (columns: `Slice, Mean, StdDev, Min, Max`) "
            "to plot **Mean intensity vs. Slice**."
        )

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                candidate = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read CSV file: {e}")
            else:
                if "Slice" not in candidate.columns or "Mean" not in candidate.columns:
                    st.error(
                        "CSV must contain 'Slice' and 'Mean' columns. "
                        f"Found columns: {list(candidate.columns)}"
                    )
                else:
                    seconds_per_slice = st.number_input(
                        "Time per slice (seconds)",
                        min_value=0.0, value=60.0, step=1.0, format="%.3f",
                        help="Conversion factor from imaging slice/frame number to time. "
                             "Suggested: 60 seconds per slice (adjust to match your "
                             "acquisition interval).",
                    )
                    candidate["Time"] = candidate["Slice"] * seconds_per_slice
                    data = candidate
                    data_label = uploaded_file.name
        else:
            st.info("Upload a CSV file to see the Mean intensity vs. Slice plot.")

    else:
        st.markdown(
            "Generates a synthetic fluorescence trace using the model and "
            "rate constants currently set in the sidebar. Two independent "
            "noise sources can be applied: Gaussian noise on the rate "
            "constant(s) themselves (drawn fresh at every simulated time "
            "step, before the model is integrated forward), and/or Gaussian "
            "measurement noise added directly to the resulting Mean "
            "intensity trace (e.g. camera/shot noise). The resulting trace "
            "is then treated like experimental data below."
        )

        if is_two_step:
            noise_col1, noise_col2, noise_col3 = st.columns(3)
            with noise_col1:
                k1_noise_std = st.number_input(
                    "k1 noise std dev", min_value=0.0, value=0.02, step=0.01, format="%.4f",
                    help="Standard deviation of the Gaussian noise added to k1 at each time step.",
                )
            with noise_col2:
                k2_noise_std = st.number_input(
                    "k2 noise std dev", min_value=0.0, value=0.02, step=0.01, format="%.4f",
                    help="Standard deviation of the Gaussian noise added to k2 at each time step.",
                )
            with noise_col3:
                kb_noise_std = st.number_input(
                    "kb noise std dev", min_value=0.0, value=0.005, step=0.001, format="%.4f",
                    help="Standard deviation of the Gaussian noise added to kb at each time step.",
                )
        else:
            noise_col1, noise_col2 = st.columns(2)
            with noise_col1:
                km_noise_std = st.number_input(
                    "km noise std dev", min_value=0.0, value=0.02, step=0.01, format="%.4f",
                    help="Standard deviation of the Gaussian noise added to km at each time step.",
                )
            with noise_col2:
                kb_noise_std = st.number_input(
                    "kb noise std dev", min_value=0.0, value=0.005, step=0.001, format="%.4f",
                    help="Standard deviation of the Gaussian noise added to kb at each time step.",
                )

        measurement_noise_std = st.number_input(
            "Measurement noise std dev (Mean intensity)",
            min_value=0.0, value=0.0, step=0.5,
            help="Standard deviation of independent Gaussian noise added directly "
                 "to the simulated Mean intensity trace (e.g. camera/shot noise), "
                 "on top of any rate-constant noise above. Set to 0 to disable.",
        )

        use_seed = st.checkbox("Fix random seed (reproducible noise)", value=False)
        seed = None
        measurement_seed = None
        if use_seed:
            seed = int(st.number_input("Random seed", min_value=0, value=0, step=1))
            measurement_seed = seed + 1

        generate_button = st.button("Generate Synthetic Data", type="primary")

        if generate_button:
            t_syn = np.linspace(0, t_end, int(n_points))
            with st.spinner("Generating synthetic data..."):
                if is_two_step:
                    params_syn = {"u": u, "k1": k1, "k2": k2, "kb": kb, "kd": kd, "alpha": alpha}
                    _, _, _, _, _, F_syn = simulate_2step_noisy(
                        t_syn, params_syn, I0=I0, X0=X0, M0=M0, B0=B0,
                        k1_std=k1_noise_std, k2_std=k2_noise_std, kb_std=kb_noise_std,
                        seed=seed,
                    )
                else:
                    params_syn = {"u": u, "km": km, "kb": kb, "kd": kd, "alpha": alpha}
                    _, _, _, _, F_syn = simulate_1step_noisy(
                        t_syn, params_syn, I0=I0, M0=M0, B0=B0,
                        km_std=km_noise_std, kb_std=kb_noise_std,
                        seed=seed,
                    )

                F_syn = add_measurement_noise(F_syn, measurement_noise_std, seed=measurement_seed)

            st.session_state["synthetic_data_df"] = pd.DataFrame(
                {"Slice": t_syn, "Mean": F_syn, "Time": t_syn}
            )
            st.session_state["synthetic_params"] = {
                "is_two_step": is_two_step,
                "I0": I0,
                "km": None if is_two_step else km,
                "k1": k1 if is_two_step else None,
                "k2": k2 if is_two_step else None,
                "kb": kb,
                "kd": kd,
                "alpha": alpha,
            }

        if "synthetic_data_df" in st.session_state:
            data = st.session_state["synthetic_data_df"]
            data_label = "Synthetic data"
        else:
            st.info("Click **Generate Synthetic Data** to create a synthetic trace.")

    if data is not None:
                fig2, ax2 = plt.subplots(figsize=(9, 5))
                ax2.plot(data["Time"], data["Mean"], marker="o", markersize=3)
                ax2.set_xlabel("Time (sec)")
                ax2.set_ylabel("Mean intensity")
                ax2.set_title(data_label)
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

                time_min = float(data["Time"].min())
                time_max = float(data["Time"].max())

                col1, col2 = st.columns(2)
                with col1:
                    time_start, time_end = st.slider(
                        "Fitting region (Time range, sec)",
                        min_value=time_min, max_value=time_max,
                        value=(time_min, time_max),
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

                time_all = data["Time"].to_numpy(dtype=float)
                F_raw_all = data["Mean"].to_numpy(dtype=float)
                mask = (time_all >= time_start) & (time_all <= time_end)
                t_raw = time_all[mask]
                F_raw = F_raw_all[mask]

                if len(t_raw) < 5:
                    st.warning(
                        "Selected time region contains too few data points "
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

                    if fit_is_two_step:
                        current_param_names = ["I0", "k1", "k2", "kb", "kd", "alpha"]
                        current_centers = [I0_guess, k1_guess, k2_guess, kb_guess, kd_guess, alpha_guess]
                        current_bounds = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 5.0, 5.0, 10.0])
                    else:
                        current_param_names = ["I0", "km", "kb", "kd", "alpha"]
                        current_centers = [I0_guess, km_guess, kb_guess, kd_guess, alpha_guess]
                        current_bounds = ([0.0, 0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 5.0, 10.0])
                    st.session_state["current_fit_data"] = {
                        "t_fit": t_fit,
                        "F_meas": F_meas,
                        "fit_is_two_step": fit_is_two_step,
                        "u_step": u_step,
                        "param_names": current_param_names,
                        "centers": current_centers,
                        "bounds": current_bounds,
                        "data_source": data_source,
                    }

                    fit_button = st.button("Fit Parameters", type="primary")

                    st.divider()
                    st.markdown(
                        "**Multi-start fit**: repeat the fit above from many "
                        "independently randomized initial guesses (log-uniform "
                        "around the guesses in the expander above), to check "
                        "convergence robustness and parameter identifiability "
                        "on the *same* data selected above."
                    )
                    multi_col1, multi_col2 = st.columns(2)
                    with multi_col1:
                        n_multi_runs = st.number_input(
                            "Number of runs (N)", min_value=2, value=30, step=1,
                        )
                    with multi_col2:
                        include_nonconverged = st.checkbox(
                            "Include non-converged runs in plots/stats", value=False,
                        )
                    multi_seed_fix = st.checkbox(
                        "Fix random seed for initial guesses", value=False, key="multi_seed_fix",
                    )
                    multi_seed = None
                    if multi_seed_fix:
                        multi_seed = int(st.number_input(
                            "Initial-guess random seed", min_value=0, value=0, step=1, key="multi_seed_val",
                        ))

                    multi_fit_button = st.button("Run Multi-Start Fit (N runs)")

                    if fit_button:
                        fixed = {"u": u_step}

                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        ax3.plot(t_raw, F_raw, "k--", linewidth=1.5, label="Raw selected data")
                        ax3.plot(t_raw, F_meas, "o", markersize=4, label="Baseline-corrected data")

                        if fit_is_two_step:
                            x0 = np.array([I0_guess, k1_guess, k2_guess, kb_guess, kd_guess, alpha_guess])
                            bounds = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 5.0, 5.0, 10.0])

                            with st.spinner("Fitting parameters..."):
                                fit = least_squares(
                                    residuals_2step, x0, bounds=bounds, args=(t_fit, F_meas, fixed),
                                    max_nfev=1000,
                                )
                            I0_hat, k1_hat, k2_hat, kb_hat, kd_hat, alpha_hat = fit.x

                            fitted_params = {
                                "u": u_step, "k1": k1_hat, "k2": k2_hat,
                                "kb": kb_hat, "kd": kd_hat, "alpha": alpha_hat,
                            }
                            st.session_state["fit_result"] = {
                                "is_two_step": True,
                                "km": None,
                                "k1": k1_hat,
                                "k2": k2_hat,
                                "kb": kb_hat,
                                "kd": kd_hat,
                                "alpha": alpha_hat,
                                "source_label": (
                                    "synthetic data"
                                    if data_source == "Generate synthetic data from simulation"
                                    else "experimental data"
                                ),
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
                            ax3.set_title("2-step fit on selected time region")
                        else:
                            x0 = np.array([I0_guess, km_guess, kb_guess, kd_guess, alpha_guess])
                            bounds = ([0.0, 0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 5.0, 10.0])

                            with st.spinner("Fitting parameters..."):
                                fit = least_squares(
                                    residuals_1step, x0, bounds=bounds, args=(t_fit, F_meas, fixed),
                                    max_nfev=1000,
                                )
                            I0_hat, km_hat, kb_hat, kd_hat, alpha_hat = fit.x

                            fitted_params = {
                                "u": u_step, "km": km_hat,
                                "kb": kb_hat, "kd": kd_hat, "alpha": alpha_hat,
                            }
                            st.session_state["fit_result"] = {
                                "is_two_step": False,
                                "km": km_hat,
                                "k1": None,
                                "k2": None,
                                "kb": kb_hat,
                                "kd": kd_hat,
                                "alpha": alpha_hat,
                                "source_label": (
                                    "synthetic data"
                                    if data_source == "Generate synthetic data from simulation"
                                    else "experimental data"
                                ),
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
                            ax3.set_title("1-step fit on selected time region")

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

                        ax3.set_xlabel("Time (sec)")
                        ax3.set_ylabel("Mean intensity")
                        ax3.legend()
                        fig3.tight_layout()
                        st.pyplot(fig3)

                        synthetic_params = st.session_state.get("synthetic_params")
                        if (
                            data_source == "Generate synthetic data from simulation"
                            and not fit_is_two_step
                            and synthetic_params is not None
                            and not synthetic_params["is_two_step"]
                        ):
                            a_true = synthetic_params["km"] + synthetic_params["kd"]
                            b_true = synthetic_params["kb"] + synthetic_params["kd"]
                            G_true = synthetic_params["alpha"] * synthetic_params["km"]

                            a_fit = km_hat + kd_hat
                            b_fit = kb_hat + kd_hat
                            G_fit = alpha_hat * km_hat

                            st.markdown("**Synthetic data: input vs. fitted derived quantities**")
                            comparison_df = pd.DataFrame(
                                {
                                    "Quantity": ["a = km + kd", "b = kb + kd", "G = alpha * km"],
                                    "Synthetic input": [a_true, b_true, G_true],
                                    "Least-squares output": [a_fit, b_fit, G_fit],
                                }
                            )
                            st.table(comparison_df.set_index("Quantity"))

                        if (
                            data_source == "Generate synthetic data from simulation"
                            and fit_is_two_step
                            and synthetic_params is not None
                            and synthetic_params["is_two_step"]
                        ):
                            a_true = synthetic_params["k1"] + synthetic_params["kd"]
                            c_true = synthetic_params["k2"] + synthetic_params["kd"]
                            b_true = synthetic_params["kb"] + synthetic_params["kd"]
                            G3_true = synthetic_params["alpha"] * synthetic_params["k1"] * synthetic_params["k2"]

                            a_fit = k1_hat + kd_hat
                            c_fit = k2_hat + kd_hat
                            b_fit = kb_hat + kd_hat
                            G3_fit = alpha_hat * k1_hat * k2_hat

                            st.markdown("**Synthetic data: input vs. fitted derived quantities**")
                            comparison_df = pd.DataFrame(
                                {
                                    "Quantity": ["a = k1 + kd", "c = k2 + kd", "b = kb + kd", "G3 = alpha * k1 * k2"],
                                    "Synthetic input": [a_true, c_true, b_true, G3_true],
                                    "Least-squares output": [a_fit, c_fit, b_fit, G3_fit],
                                }
                            )
                            st.table(comparison_df.set_index("Quantity"))

                    if multi_fit_button:
                        fixed_multi = {"u": u_step}

                        synthetic_params_centers = st.session_state.get("synthetic_params")
                        use_synthetic_centers = (
                            data_source == "Generate synthetic data from simulation"
                            and synthetic_params_centers is not None
                            and synthetic_params_centers["is_two_step"] == fit_is_two_step
                            and synthetic_params_centers.get("I0") is not None
                        )

                        if fit_is_two_step:
                            if use_synthetic_centers:
                                centers = [
                                    synthetic_params_centers["I0"], synthetic_params_centers["k1"],
                                    synthetic_params_centers["k2"], synthetic_params_centers["kb"],
                                    synthetic_params_centers["kd"], synthetic_params_centers["alpha"],
                                ]
                            else:
                                centers = [I0_guess, k1_guess, k2_guess, kb_guess, kd_guess, alpha_guess]
                            param_names = ["I0", "k1", "k2", "kb", "kd", "alpha"]
                            bounds_multi = ([0.0, 0.0, 0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 5.0, 5.0, 10.0])
                            residual_fn = residuals_2step
                        else:
                            if use_synthetic_centers:
                                centers = [
                                    synthetic_params_centers["I0"], synthetic_params_centers["km"],
                                    synthetic_params_centers["kb"], synthetic_params_centers["kd"],
                                    synthetic_params_centers["alpha"],
                                ]
                            else:
                                centers = [I0_guess, km_guess, kb_guess, kd_guess, alpha_guess]
                            param_names = ["I0", "km", "kb", "kd", "alpha"]
                            bounds_multi = ([0.0, 0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 5.0, 10.0])
                            residual_fn = residuals_1step

                        st.caption(
                            "Multi-start initial guesses centered on: "
                            + (
                                "synthetic input parameters (sidebar values used to generate this data)"
                                if use_synthetic_centers
                                else "advanced initial guesses above (no matching synthetic ground truth available)"
                            )
                        )

                        with st.spinner(f"Running multi-start fit ({int(n_multi_runs)} runs)..."):
                            results_df = run_multi_start(
                                residual_fn, param_names, centers, bounds_multi,
                                args=(t_fit, F_meas, fixed_multi), n_runs=int(n_multi_runs),
                                seed=multi_seed,
                            )

                        if fit_is_two_step:
                            results_df["a"] = results_df["k1"] + results_df["kd"]
                            results_df["c"] = results_df["k2"] + results_df["kd"]
                            results_df["b"] = results_df["kb"] + results_df["kd"]
                            results_df["G3"] = results_df["alpha"] * results_df["k1"] * results_df["k2"]
                            results_df["G3*I0"] = results_df["G3"] * results_df["I0"]
                            derived_names = ["a", "c", "b", "G3", "G3*I0"]
                        else:
                            results_df["a"] = results_df["km"] + results_df["kd"]
                            results_df["b"] = results_df["kb"] + results_df["kd"]
                            results_df["G"] = results_df["alpha"] * results_df["km"]
                            results_df["G*I0"] = results_df["G"] * results_df["I0"]
                            derived_names = ["a", "b", "G", "G*I0"]

                        synthetic_params_multi = st.session_state.get("synthetic_params")
                        true_values = {}
                        if (
                            data_source == "Generate synthetic data from simulation"
                            and synthetic_params_multi is not None
                            and synthetic_params_multi["is_two_step"] == fit_is_two_step
                            and synthetic_params_multi.get("I0") is not None
                        ):
                            true_values["I0"] = synthetic_params_multi["I0"]
                            true_values["kb"] = synthetic_params_multi["kb"]
                            true_values["kd"] = synthetic_params_multi["kd"]
                            true_values["alpha"] = synthetic_params_multi["alpha"]
                            if fit_is_two_step:
                                true_values["k1"] = synthetic_params_multi["k1"]
                                true_values["k2"] = synthetic_params_multi["k2"]
                                true_values["a"] = true_values["k1"] + true_values["kd"]
                                true_values["c"] = true_values["k2"] + true_values["kd"]
                                true_values["b"] = true_values["kb"] + true_values["kd"]
                                true_values["G3"] = (
                                    true_values["alpha"] * true_values["k1"] * true_values["k2"]
                                )
                                true_values["G3*I0"] = true_values["G3"] * true_values["I0"]
                            else:
                                true_values["km"] = synthetic_params_multi["km"]
                                true_values["a"] = true_values["km"] + true_values["kd"]
                                true_values["b"] = true_values["kb"] + true_values["kd"]
                                true_values["G"] = true_values["alpha"] * true_values["km"]
                                true_values["G*I0"] = true_values["G"] * true_values["I0"]

                        append_entry({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "fit_is_two_step": fit_is_two_step,
                            "data_source": data_source,
                            "n_total": len(results_df),
                            "n_converged": int(results_df["converged"].sum()),
                            "results_df": results_df,
                            "param_names": param_names,
                            "derived_names": derived_names,
                            "true_values": true_values,
                        })

                        render_multi_start_results(
                            results_df, param_names, derived_names, true_values, include_nonconverged,
                        )

with bode_tab:
    st.markdown(
        "Frequency response of the maturation model, showing how strongly the "
        "reporter attenuates and delays gene expression fluctuations at each "
        "frequency, and the cutoff frequency above which dynamics are lost to "
        "maturation/bleaching. Plots the synthetic-data ground truth (if "
        "generated) and the most recent least-squares fit (if run) for "
        "comparison; the -3 dB cutoff metrics below still use the sidebar's "
        "current rate constants."
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
                title_suffix = "2-step maturation model"
            else:
                w, mag, phase = bode_1step(alpha, km, kb, kd, w)
                title_suffix = "1-step maturation model"

            fig4, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6))

            ax_mag.set_ylabel("Magnitude (dB)")
            ax_mag.set_xlabel("Frequency (rad/sec)")
            ax_mag.set_title("Bode Plot")
            ax_mag.grid(True, which="both", linestyle="--", alpha=0.5)

            ax_phase.set_ylabel("Phase (degrees)")
            ax_phase.set_xlabel("Frequency (rad/sec)")
            ax_phase.grid(True, which="both", linestyle="--", alpha=0.5)

            synthetic_params = st.session_state.get("synthetic_params")
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
                ax_mag.semilogx(w, mag_syn, color="tab:green", linestyle="--", label=syn_label)
                ax_phase.semilogx(w, phase_syn, color="tab:green", linestyle="--", label=syn_label)

            fit_result = st.session_state.get("fit_result")
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
                ax_mag.semilogx(w, mag_fit, color="tab:red", linestyle=":", label=fit_label)
                ax_phase.semilogx(w, phase_fit, color="tab:red", linestyle=":", label=fit_label)

            has_overlay = synthetic_params is not None or fit_result is not None
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

        wc_numerical = numerical_cutoff(w, mag)
        cutoff_col1, cutoff_col2 = st.columns(2)
        if wc_numerical is not None:
            cutoff_col1.metric("Numerical -3 dB cutoff (rad/sec)", f"{wc_numerical:.6f}")
        else:
            cutoff_col1.warning("Cutoff not reached within the selected frequency range.")

        if is_two_step:
            wc_analytical = analytical_cutoff_2step(k1, k2, kb, kd)
        else:
            wc_analytical = analytical_cutoff_1step(km, kb, kd)

        if wc_analytical is not None:
            cutoff_col2.metric("Analytical -3 dB cutoff (rad/sec)", f"{wc_analytical:.6f}")
        else:
            cutoff_col2.warning("No positive real cutoff root found.")
    else:
        st.info("Set the frequency range and click **Plot Bode Response**.")

with history_tab:
    st.markdown(
        "Every multi-start fit run, saved to disk (`multi_start_history.json`) "
        "so it survives app restarts — most recent first. Each entry keeps its "
        "fitted results and the synthetic-data ground truth (if any) exactly "
        "as they were at the time it ran — click **Display** to view its "
        "histograms and summary statistics again, even after changing "
        "sidebar parameters or running other fits since."
    )

    history = load_history()

    if not history:
        st.info(
            "No multi-start fit runs saved yet. Run one from the "
            "**Run Multi-Start Fit (N runs)** button in the Least Squares "
            "Fitting tab."
        )
    else:
        clear_col1, clear_col2 = st.columns([3, 1])
        with clear_col2:
            if st.button("Clear history"):
                clear_history()
                st.rerun()

        for i in range(len(history) - 1, -1, -1):
            entry = history[i]
            model_label = "2-step" if entry["fit_is_two_step"] else "1-step"
            source_label = (
                "synthetic data"
                if entry["data_source"] == "Generate synthetic data from simulation"
                else "experimental data"
            )
            truth_label = "available" if entry["true_values"] else "not available"

            with st.container(border=True):
                st.markdown(
                    f"**Run {i + 1}** — {entry['timestamp']} — {model_label} model, "
                    f"fit to {source_label}, {entry['n_converged']}/{entry['n_total']} "
                    f"converged, ground truth {truth_label}"
                )
                hist_col1, hist_col2, hist_col3 = st.columns([2, 1, 1])
                with hist_col1:
                    hist_include_nonconverged = st.checkbox(
                        "Include non-converged runs", value=False, key=f"history_nonconv_{i}",
                    )
                with hist_col2:
                    display_clicked = st.button("Display", key=f"history_display_{i}")
                with hist_col3:
                    delete_clicked = st.button("Delete", key=f"history_delete_{i}")

                if delete_clicked:
                    delete_entry(i)
                    st.rerun()

                if display_clicked:
                    render_multi_start_results(
                        entry["results_df"], entry["param_names"], entry["derived_names"],
                        entry["true_values"], hist_include_nonconverged,
                    )

with profile_tab:
    st.markdown(
        "Profile likelihood: sweep one raw parameter or derived quantity across "
        "a grid of fixed values. At each grid point, everything else is "
        "re-optimized (with multiple random restarts) to fit as well as it can "
        "around that fixed value, and the best achieved cost is recorded. A "
        "sharply rising cost away from the best-fit value means that "
        "parameter/quantity is well-identified by the data; a flat profile "
        "means it isn't — it can trade off against other parameters without "
        "hurting the fit. This is a stronger identifiability check than the "
        "multi-start scatter alone, since every other parameter is actively "
        "re-optimized at each point rather than left wherever a single fit "
        "happened to land. **This can be slow** — each grid point re-runs the "
        "full multi-start fitting machinery."
    )

    current_fit_data = st.session_state.get("current_fit_data")
    if current_fit_data is None:
        st.info(
            "Select a data source and a fitting region with at least 5 points "
            "in the **Least Squares Fitting** tab first (no need to click a "
            "Fit button — just get past the region-selection step)."
        )
    else:
        fit_is_two_step_p = current_fit_data["fit_is_two_step"]
        param_names_p = current_fit_data["param_names"]
        centers_p = current_fit_data["centers"]
        bounds_p = current_fit_data["bounds"]
        data_source_p = current_fit_data["data_source"]
        derived_names_p = list(DERIVED_QUANTITY_FORMULAS[fit_is_two_step_p].keys())

        st.caption(
            f"Profiling against the {'2-step' if fit_is_two_step_p else '1-step'} "
            "model and data/region currently selected in the Least Squares "
            "Fitting tab."
        )

        synthetic_params_p = st.session_state.get("synthetic_params")
        true_values_p = compute_true_values(synthetic_params_p, fit_is_two_step_p)

        profile_target = st.selectbox(
            "Parameter or derived quantity to profile",
            options=param_names_p + derived_names_p,
        )
        is_derived = profile_target in derived_names_p

        center_lookup = dict(zip(param_names_p, centers_p))
        if profile_target in true_values_p:
            default_center = true_values_p[profile_target]
        elif profile_target in center_lookup:
            default_center = center_lookup[profile_target]
        else:
            default_center = DERIVED_QUANTITY_FORMULAS[fit_is_two_step_p][profile_target](center_lookup)
        default_center = max(float(default_center), 1e-6)

        grid_col1, grid_col2, grid_col3 = st.columns(3)
        with grid_col1:
            grid_min = st.number_input(
                "Grid min", value=default_center * 0.3, format="%.6f", key="profile_grid_min",
            )
        with grid_col2:
            grid_max = st.number_input(
                "Grid max", value=default_center * 3.0, format="%.6f", key="profile_grid_max",
            )
        with grid_col3:
            grid_points = st.number_input(
                "Grid points", min_value=3, value=8, step=1, key="profile_grid_points",
            )

        restart_col1, restart_col2, restart_col3 = st.columns(3)
        with restart_col1:
            n_restarts_p = st.number_input(
                "Random restarts per grid point", min_value=1, value=4, step=1,
                key="profile_n_restarts",
            )
        with restart_col2:
            profile_max_nfev = st.number_input(
                "Max evaluations per restart", min_value=100, value=400, step=100,
                key="profile_max_nfev",
            )
        with restart_col3:
            profile_seed_fix = st.checkbox(
                "Fix random seed", value=False, key="profile_seed_fix",
            )
        profile_seed = None
        if profile_seed_fix:
            profile_seed = int(st.number_input(
                "Profile random seed", min_value=0, value=0, step=1, key="profile_seed_val",
            ))

        st.caption(
            f"Worst case: {int(grid_points)} grid points x {int(n_restarts_p)} "
            "restarts = "
            f"{int(grid_points) * int(n_restarts_p)} optimization runs."
        )

        run_profile_button = st.button("Run Profile Likelihood", type="primary")

        if run_profile_button:
            if grid_max <= grid_min:
                st.error("Grid max must be greater than grid min.")
            else:
                grid_values = np.linspace(grid_min, grid_max, int(grid_points))
                residual_fn_p = residuals_2step if fit_is_two_step_p else residuals_1step
                fixed_p = {"u": current_fit_data["u_step"]}
                args_p = (current_fit_data["t_fit"], current_fit_data["F_meas"], fixed_p)

                with st.spinner(
                    f"Computing profile likelihood for {profile_target} "
                    f"({int(grid_points)} grid points x {int(n_restarts_p)} restarts)..."
                ):
                    if is_derived:
                        profile_df = profile_derived_quantity(
                            residual_fn_p, param_names_p, profile_target, fit_is_two_step_p,
                            grid_values, centers_p, bounds_p, args_p,
                            n_restarts=int(n_restarts_p), seed=profile_seed,
                            max_nfev=int(profile_max_nfev),
                        )
                    else:
                        profile_df = profile_raw_parameter(
                            residual_fn_p, param_names_p, profile_target,
                            grid_values, centers_p, bounds_p, args_p,
                            n_restarts=int(n_restarts_p), seed=profile_seed,
                            max_nfev=int(profile_max_nfev),
                        )

                best_row = profile_df.loc[profile_df["cost"].idxmin()]
                st.success(
                    f"Best cost {best_row['cost']:.6g} at "
                    f"{profile_target} = {best_row['value']:.6g}"
                )

                fig_p, ax_p = plt.subplots(figsize=(8, 5))
                ax_p.plot(profile_df["value"], profile_df["cost"], "o-", color="tab:blue")
                ax_p.axvline(
                    best_row["value"], color="tab:green", linestyle="--",
                    label="Best-fit value (this profile)",
                )
                if profile_target in true_values_p:
                    ax_p.axvline(
                        true_values_p[profile_target], color="tab:red", linestyle=":",
                        label="Synthetic input (true value)",
                    )
                ax_p.set_xlabel(profile_target)
                ax_p.set_ylabel("Min. sum-of-squared residuals")
                ax_p.set_title(f"Profile likelihood: {profile_target}")
                ax_p.legend(fontsize=8)
                fig_p.tight_layout()
                st.pyplot(fig_p)

                with st.expander("Profile likelihood raw data"):
                    st.dataframe(profile_df, hide_index=True)
