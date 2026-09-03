# --- Least Squares Fitting tab: fit the full maturation model to data ----
# Fits I0/km(or k1,k2)/kb/kd/alpha to whatever trace is loaded in the Data
# tab, over a user-selected time region. Two ways to fit: a single
# `scipy.optimize.least_squares` call from one initial guess ("Fit
# Parameters"), or a multi-start sweep from many randomized initial guesses
# ("Run Multi-Start Fit") to check convergence robustness/identifiability.

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import least_squares

from Maturation_Models import (
    simulate_1step,
    simulate_2step,
    residuals_1step,
    residuals_2step,
)
from multi_start_fit import run_multi_start
from history_store import append_multi_start_entry

from app.shared import render_multi_start_results


def render_fitting_tab():
    data = st.session_state.get("current_data")
    data_label = st.session_state.get("current_data_label", "")
    data_source = st.session_state.get("current_data_source")

    if data is None:
        st.info(
            "No data loaded yet. Upload a CSV or generate synthetic data in "
            "the **Data** tab first."
        )
    else:
                st.subheader("Fit maturation model parameters to data")
                st.caption(f"Using: {data_label}")
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
                    dataset_key = st.session_state.get("current_dataset_key")
                    run_key = "|".join(str(v) for v in (
                        dataset_key, fit_is_two_step, u_step, time_start, time_end, baseline,
                    ))
                    st.session_state["current_fit_data"] = {
                        "t_fit": t_fit,
                        "F_meas": F_meas,
                        "fit_is_two_step": fit_is_two_step,
                        "u_step": u_step,
                        "param_names": current_param_names,
                        "centers": current_centers,
                        "bounds": current_bounds,
                        "data_source": data_source,
                        "dataset_key": dataset_key,
                        "dataset_label": data_label,
                        "run_key": run_key,
                        "time_start": time_start,
                        "time_end": time_end,
                        "baseline": baseline,
                    }

                    # Single fit: one least_squares call from the initial guess above.
                    fit_button = st.button("Fit Parameters", type="primary")

                    st.divider()
                    # Multi-start fit: the same fit repeated from many randomized
                    # initial guesses (see multi_start_fit.run_multi_start).
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
                            fit_title = "2-step fit on selected time region"
                            metrics_cols = {
                                "col1": [("I0", I0_hat), ("k1", k1_hat)],
                                "col2": [("k2", k2_hat), ("kb", kb_hat)],
                                "col3": [("kd", kd_hat), ("alpha", alpha_hat)],
                            }
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
                            fit_title = "1-step fit on selected time region"
                            metrics_cols = {
                                "col1": [("I0", I0_hat), ("km", km_hat)],
                                "col2": [("kb", kb_hat), ("kd", kd_hat)],
                                "col3": [("alpha", alpha_hat)],
                            }

                        residuals = fit.fun
                        sse = float(np.sum(residuals ** 2))
                        rmse = float(np.sqrt(np.mean(residuals ** 2)))
                        ss_tot = float(np.sum((F_meas - np.mean(F_meas)) ** 2))
                        r_squared = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")

                        comparison_df = None
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

                            comparison_df = pd.DataFrame(
                                {
                                    "Quantity": ["a = km + kd", "b = kb + kd", "G = alpha * km"],
                                    "Synthetic input": [a_true, b_true, G_true],
                                    "Least-squares output": [a_fit, b_fit, G_fit],
                                }
                            )
                        elif (
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

                            comparison_df = pd.DataFrame(
                                {
                                    "Quantity": ["a = k1 + kd", "c = k2 + kd", "b = kb + kd", "G3 = alpha * k1 * k2"],
                                    "Synthetic input": [a_true, c_true, b_true, G3_true],
                                    "Least-squares output": [a_fit, c_fit, b_fit, G3_fit],
                                }
                            )

                        st.session_state["fit_single_result"] = {
                            "is_two_step": fit_is_two_step,
                            "t_raw": t_raw,
                            "F_raw": F_raw,
                            "F_meas": F_meas,
                            "F_fit": F_fit,
                            "title": fit_title,
                            "metrics_cols": metrics_cols,
                            "baseline": baseline,
                            "sse": sse,
                            "rmse": rmse,
                            "r_squared": r_squared,
                            "success": fit.success,
                            "message": fit.message,
                            "comparison_df": comparison_df,
                        }

                    fit_single_result = st.session_state.get("fit_single_result")
                    if fit_single_result is None:
                        st.info("Click **Fit Parameters** to fit the model to the selected data above.")
                    else:
                        fsr = fit_single_result

                        st.success(
                            f"Fit complete ({'2-step' if fsr['is_two_step'] else '1-step'} model)"
                        )
                        res_col1, res_col2, res_col3 = st.columns(3)
                        for label, val in fsr["metrics_cols"]["col1"]:
                            res_col1.metric(label, f"{val:.4f}")
                        for label, val in fsr["metrics_cols"]["col2"]:
                            res_col2.metric(label, f"{val:.4f}")
                        for label, val in fsr["metrics_cols"]["col3"]:
                            res_col3.metric(label, f"{val:.4f}")

                        st.caption(f"Baseline subtracted = {fsr['baseline']:.4f}")

                        st.markdown("**Goodness of fit**")
                        err_col1, err_col2, err_col3 = st.columns(3)
                        err_col1.metric("SSE (least-squares cost)", f"{fsr['sse']:.4f}")
                        err_col2.metric("RMSE", f"{fsr['rmse']:.4f}")
                        err_col3.metric("R²", f"{fsr['r_squared']:.4f}")
                        if not fsr["success"]:
                            st.warning(f"Solver did not fully converge: {fsr['message']}")

                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        ax3.plot(fsr["t_raw"], fsr["F_raw"], "k--", linewidth=1.5, label="Raw selected data")
                        ax3.plot(fsr["t_raw"], fsr["F_meas"], "o", markersize=4, label="Baseline-corrected data")
                        ax3.plot(
                            fsr["t_raw"], fsr["F_fit"], "-", linewidth=2,
                            label=("2-step model fit" if fsr["is_two_step"] else "1-step model fit"),
                        )
                        ax3.set_title(fsr["title"])
                        ax3.set_xlabel("Time (sec)")
                        ax3.set_ylabel("Mean intensity")
                        ax3.legend()
                        fig3.tight_layout()
                        st.pyplot(fig3)

                        if fsr["comparison_df"] is not None:
                            st.markdown("**Synthetic data: input vs. fitted derived quantities**")
                            st.table(fsr["comparison_df"].set_index("Quantity"))

                    st.divider()

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

                        append_multi_start_entry({
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

                        st.session_state["fit_multi_result"] = {
                            "results_df": results_df,
                            "param_names": param_names,
                            "derived_names": derived_names,
                            "true_values": true_values,
                            "use_synthetic_centers": use_synthetic_centers,
                        }

                    fit_multi_result = st.session_state.get("fit_multi_result")
                    if fit_multi_result is None:
                        st.info("Click **Run Multi-Start Fit (N runs)** to run the fit.")
                    else:
                        st.caption(
                            "Multi-start initial guesses centered on: "
                            + (
                                "synthetic input parameters (Simulation tab values used to generate this data)"
                                if fit_multi_result["use_synthetic_centers"]
                                else "advanced initial guesses above (no matching synthetic ground truth available)"
                            )
                        )
                        render_multi_start_results(
                            fit_multi_result["results_df"], fit_multi_result["param_names"],
                            fit_multi_result["derived_names"], fit_multi_result["true_values"],
                            include_nonconverged,
                        )
