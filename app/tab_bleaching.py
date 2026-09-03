# --- Bleaching Only Simulation tab -----------------------------------
# Two things, in order: (1) simulate/fit a pure photobleaching-decay model
# (M(t) = M0*exp(-b*t), no maturation) to estimate b = kb+kd directly; (2)
# reuse that b as a *fixed* value in a fit of the full maturation model
# ("known bleaching pole fit"), removing one free parameter. Both fits are
# saved together to the Bleaching Fit History tab when the known-b fit's
# "Run Multi-Start Fit" button is clicked (the last step in the tab).

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from Bleaching_Only_Model import simulate_bleach, residuals_bleach
from Maturation_Model_Known_Bleaching_Pole import (
    residuals_1step_known_b,
    residuals_2step_known_b,
)
from gaussian_noise import simulate_bleach_noisy, add_measurement_noise
from multi_start_fit import run_multi_start
from multi_start_plots import plot_histograms
from history_store import append_bleach_entry

from app.shared import render_multi_start_results


def _multi_start_part(result):
    """Build a bleach-history fit-result part from a stored multi-start result dict, or None.

    Shared by the bleach-only and known-b saves in the Bleaching Only tab so
    each save bundles both fits' latest results into one history entry.
    """
    if result is None:
        return None
    results_df = result["results_df"]
    return {
        "n_total": len(results_df),
        "n_converged": int(results_df["converged"].sum()),
        "param_names": result["param_names"],
        "derived_names": result["derived_names"],
        "true_values": result["true_values"],
        "extra_info": result.get("extra_info", {}),
        "results_df": results_df,
    }


def _save_bleach_tab_history_entry():
    """Save the current Bleaching Only tab state as one combined history entry.

    Called when the known bleaching pole fit's "Run Multi-Start Fit" button
    is clicked (the last step in the tab), bundling everything currently run
    in the tab this session: both fits. Any piece not yet run is saved as
    None.
    """
    append_bleach_entry({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bleach_only": _multi_start_part(st.session_state.get("bleach_multi_result")),
        "known_b": _multi_start_part(st.session_state.get("known_b_multi_result")),
    })


def render_bleaching_tab():
    st.markdown(
        "Models cells whose FP is already fully matured when imaging starts, "
        "so I(t) ~ 0 throughout and no new mature protein is produced. The "
        "mature pool only decays, via photobleaching and dilution:\n\n"
        "dM/dt = -kb*M - kd*M = -b*M,  b = kb + kd\n\n"
        "with M(0) = M0. This has the closed form M(t) = M0 * exp(-b*t), "
        "so F(t) = alpha*M(t) = A*exp(-b*t) with lumped amplitude A = alpha*M0. "
        "Adjust the initial condition and rate constants below, then click "
        "**Run Simulation**."
    )

    bleach_ic_cols = st.columns(4)
    with bleach_ic_cols[0]:
        M0_bleach = st.number_input(
            "M0 - mature (fluorescent) protein at t=0", min_value=0.0, value=100.0, step=10.0,
            help="Suggested: 100. Fluorescence already present in the cell when imaging starts.",
            key="bleach_M0",
        )
    with bleach_ic_cols[1]:
        B0_bleach = st.number_input(
            "B0 - bleached protein at t=0", min_value=0.0, value=0.0, step=10.0,
            help="Suggested: 0. No photobleaching has occurred yet.",
            key="bleach_B0",
        )

    st.session_state.setdefault("bleach_kb", 0.02)
    st.session_state.setdefault("bleach_kd", 0.0)
    st.session_state.setdefault("bleach_alpha", 1.0)

    sim_run_version = st.session_state.get("sim_run_version")
    if sim_run_version is not None and st.session_state.get("bleach_synced_version") != sim_run_version:
        st.session_state["bleach_kb"] = st.session_state["sim_kb"]
        st.session_state["bleach_kd"] = st.session_state["sim_kd"]
        st.session_state["bleach_alpha"] = st.session_state["sim_alpha"]
        st.session_state["bleach_synced_version"] = sim_run_version

    if sim_run_version is not None:
        st.caption(
            "kb, kd, and alpha below are synced from the last **Run Simulation** in "
            "the **Simulation** tab (kb="
            f"{st.session_state['sim_kb']:.4f}, kd={st.session_state['sim_kd']:.4f}, "
            f"alpha={st.session_state['sim_alpha']:.4f}"
            "); edit them here to override."
        )

    bleach_rc_cols = st.columns(4)
    with bleach_rc_cols[0]:
        kb_bleach = st.number_input(
            "kb - photobleaching rate (M -> B)", min_value=0.0, step=0.005, format="%.4f",
            help="Suggested range: 0.0-0.1 /sec. Illustrative default: 0.02. Synced from the "
                 "Simulation tab's kb after a run, but can be overridden here.",
            key="bleach_kb",
        )
    with bleach_rc_cols[1]:
        kd_bleach = st.number_input(
            "kd - degradation / dilution rate", min_value=0.0, step=0.005, format="%.4f",
            help="Suggested range: 0.001-0.05 /sec. Default: 0 (growth halted). Synced from the "
                 "Simulation tab's kd after a run, but can be overridden here.",
            key="bleach_kd",
        )
    with bleach_rc_cols[2]:
        alpha_bleach = st.number_input(
            "alpha - fluorescence scaling factor", min_value=0.0, step=0.1,
            help="Suggested: 1.0. Brightness per unit mature protein. Synced from the "
                 "Simulation tab's alpha after a run, but can be overridden here.",
            key="bleach_alpha",
        )

    st.subheader("Derived parameters")
    b_val_bleach = kb_bleach + kd_bleach
    A_val_bleach = alpha_bleach * M0_bleach
    dp_cols_bleach = st.columns(2)
    dp_cols_bleach[0].metric("b = kb + kd", f"{b_val_bleach:.4f}")
    dp_cols_bleach[1].metric("A = alpha * M0", f"{A_val_bleach:.4f}")

    bleach_st_cols = st.columns(3)
    with bleach_st_cols[0]:
        t_end_bleach = st.number_input("End time (sec)", min_value=1.0, value=60.0, step=10.0, key="bleach_t_end")
    with bleach_st_cols[1]:
        n_points_bleach = st.number_input(
            "Number of time points", min_value=10, value=300, step=10, key="bleach_n_points"
        )
    with bleach_st_cols[2]:
        st.write("")
        run_bleach = st.button("Run Simulation", type="primary", key="bleach_run")

    st.divider()

    if run_bleach:
        t_eval_bleach = np.linspace(0, t_end_bleach, int(n_points_bleach))
        params_bleach = {"kb": kb_bleach, "kd": kd_bleach, "alpha": alpha_bleach}

        with st.spinner("Running simulation..."):
            t_b, M_b, B_b, F_b = simulate_bleach(t_eval_bleach, params_bleach, M0=M0_bleach, B0=B0_bleach)

        st.session_state["bleach_sim"] = {
            "t": t_b, "M": M_b, "B": B_b, "F": F_b,
            "M0": M0_bleach, "B0": B0_bleach,
            "kb": kb_bleach, "kd": kd_bleach, "alpha": alpha_bleach,
        }

    bleach_sim = st.session_state.get("bleach_sim")

    if bleach_sim is None:
        st.info("Set your parameters above and click **Run Simulation**.")
    else:
        fig_b, ax_b = plt.subplots(figsize=(9, 5))
        ax_b.plot(bleach_sim["t"], bleach_sim["M"], label="M (mature)")
        ax_b.plot(bleach_sim["t"], bleach_sim["B"], label="B (bleached)")
        ax_b.plot(bleach_sim["t"], bleach_sim["F"], "--", label="F = alpha * M (fluorescence)", linewidth=2)
        ax_b.set_xlabel("Time (sec)")
        ax_b.set_ylabel("Amount / Mean Intensity")
        ax_b.set_title("Bleaching-only model (I(t) ~ 0)")
        ax_b.legend()
        fig_b.tight_layout()

        st.pyplot(fig_b)

        st.divider()
        # Perturbs kb/kd/alpha with noise and adds measurement noise, to give
        # the fits below something to recover the true params from.
        st.subheader("Generate synthetic data")
        st.markdown(
            "Generates a synthetic fluorescence trace using the model and "
            "rate constants set above. Two independent noise sources can be "
            "applied: Gaussian noise on kb (drawn fresh at every simulated "
            "time step, before the model is integrated forward), and/or "
            "Gaussian measurement noise added directly to the resulting Mean "
            "intensity trace (e.g. camera/shot noise). The resulting trace "
            "is then used as the data to fit against below."
        )

        bleach_noise_cols = st.columns(2)
        with bleach_noise_cols[0]:
            kb_noise_std_bleach = st.number_input(
                "kb noise std dev", min_value=0.0, value=0.005, step=0.001, format="%.4f",
                help="Standard deviation of the Gaussian noise added to kb at each time step.",
                key="bleach_kb_noise_std",
            )
        with bleach_noise_cols[1]:
            measurement_noise_std_bleach = st.number_input(
                "Measurement noise std dev (Mean intensity)",
                min_value=0.0, value=0.0, step=0.5,
                help="Standard deviation of independent Gaussian noise added directly "
                     "to the simulated Mean intensity trace, on top of any kb noise above.",
                key="bleach_measurement_noise_std",
            )

        use_seed_bleach = st.checkbox(
            "Fix random seed (reproducible noise)", value=False, key="bleach_use_seed"
        )
        seed_bleach = None
        measurement_seed_bleach = None
        if use_seed_bleach:
            seed_bleach = int(st.number_input(
                "Random seed", min_value=0, value=0, step=1, key="bleach_seed_val"
            ))
            measurement_seed_bleach = seed_bleach + 1

        generate_button_bleach = st.button(
            "Generate Synthetic Data", type="primary", key="bleach_generate_data"
        )

        if generate_button_bleach:
            with st.spinner("Generating synthetic data..."):
                params_syn_bleach = {"kb": kb_bleach, "kd": kd_bleach, "alpha": alpha_bleach}
                t_syn_bleach, _, _, F_syn_bleach = simulate_bleach_noisy(
                    bleach_sim["t"], params_syn_bleach, M0=M0_bleach, B0=B0_bleach,
                    kb_std=kb_noise_std_bleach, seed=seed_bleach,
                )
                F_syn_bleach = add_measurement_noise(
                    F_syn_bleach, measurement_noise_std_bleach, seed=measurement_seed_bleach
                )

            st.session_state["bleach_synthetic_data"] = {"t": t_syn_bleach, "F": F_syn_bleach}
            st.session_state["bleach_synthetic_params"] = {
                "M0": M0_bleach, "kb": kb_bleach, "kd": kd_bleach, "alpha": alpha_bleach,
            }
            st.session_state["bleach_noise_params"] = {
                "kb_noise_std": kb_noise_std_bleach,
                "measurement_noise_std": measurement_noise_std_bleach,
                "seed": seed_bleach,
                "measurement_seed": measurement_seed_bleach,
            }

        bleach_synth = st.session_state.get("bleach_synthetic_data")
        bleach_synth_params = st.session_state.get("bleach_synthetic_params")

        if bleach_synth is None:
            st.info("Click **Generate Synthetic Data** to create a synthetic trace.")
        else:
            fig_syn_b, ax_syn_b = plt.subplots(figsize=(9, 5))
            ax_syn_b.plot(bleach_synth["t"], bleach_synth["F"], marker="o", markersize=3)
            ax_syn_b.set_xlabel("Time (sec)")
            ax_syn_b.set_ylabel("Mean intensity")
            ax_syn_b.set_title("Synthetic data")
            fig_syn_b.tight_layout()
            st.pyplot(fig_syn_b)

        st.divider()
        # Fits M0, kb, alpha (Bleaching_Only_Model.residuals_bleach) from
        # many randomized initial guesses against the synthetic data above.
        # kd is fixed at 0 (growth halted) rather than fitted.
        st.subheader("Multi-start least-squares fit")

        if bleach_synth is None:
            st.info("Generate synthetic data above to fit against.")
        else:
            st.markdown(
                "Repeats a least-squares fit of M0, kb, and alpha to the "
                "synthetic data above from many independently randomized "
                "initial guesses, log-uniform over one decade centered on the "
                "synthetic input parameters, to check convergence robustness "
                "and parameter identifiability. kd is fixed at 0 (growth "
                "halted) rather than fitted, so kb alone is the decay rate "
                "b = kb + kd. alpha and M0 only enter as A = alpha * M0."
            )

            fit_col1_bleach, fit_col2_bleach = st.columns(2)
            with fit_col1_bleach:
                n_multi_runs_bleach = st.number_input(
                    "Number of runs (N)", min_value=2, value=30, step=1, key="bleach_n_multi_runs",
                )
            with fit_col2_bleach:
                include_nonconverged_bleach = st.checkbox(
                    "Include non-converged runs in plots/stats", value=False,
                    key="bleach_include_nonconverged",
                )
            multi_seed_fix_bleach = st.checkbox(
                "Fix random seed for initial guesses", value=False, key="bleach_multi_seed_fix",
            )
            multi_seed_bleach = None
            if multi_seed_fix_bleach:
                multi_seed_bleach = int(st.number_input(
                    "Initial-guess random seed", min_value=0, value=0, step=1, key="bleach_multi_seed_val",
                ))

            multi_fit_button_bleach = st.button("Run Multi-Start Fit (N runs)", key="bleach_multi_fit_button")

            if multi_fit_button_bleach:
                # kd is fixed at 0 in the fit itself (residuals_bleach); it's
                # still reported below as a (constant) derived value so b and
                # the summary table keep their usual shape.
                param_names_bleach = ["M0", "kb", "alpha"]
                centers_bleach = [
                    bleach_synth_params["M0"], bleach_synth_params["kb"],
                    bleach_synth_params["alpha"],
                ]
                bounds_bleach = ([0.0, 0.0, 0.1], [1e6, 5.0, 10.0])

                with st.spinner(f"Running multi-start fit ({int(n_multi_runs_bleach)} runs)..."):
                    results_df_bleach = run_multi_start(
                        residuals_bleach, param_names_bleach, centers_bleach, bounds_bleach,
                        args=(bleach_synth["t"], bleach_synth["F"]), n_runs=int(n_multi_runs_bleach),
                        seed=multi_seed_bleach,
                    )

                results_df_bleach["kd"] = 0.0
                results_df_bleach["b"] = results_df_bleach["kb"] + results_df_bleach["kd"]
                results_df_bleach["A"] = results_df_bleach["alpha"] * results_df_bleach["M0"]
                derived_names_bleach = ["kd", "b", "A"]

                true_values_bleach = {
                    "M0": bleach_synth_params["M0"],
                    "kb": bleach_synth_params["kb"],
                    "alpha": bleach_synth_params["alpha"],
                    "kd": bleach_synth_params["kd"],
                    "b": bleach_synth_params["kb"] + bleach_synth_params["kd"],
                    "A": bleach_synth_params["alpha"] * bleach_synth_params["M0"],
                }

                st.session_state["bleach_multi_result"] = {
                    "results_df": results_df_bleach,
                    "param_names": param_names_bleach,
                    "derived_names": derived_names_bleach,
                    "true_values": true_values_bleach,
                }

            bleach_multi_result = st.session_state.get("bleach_multi_result")
            if bleach_multi_result is None:
                st.info("Click **Run Multi-Start Fit (N runs)** to fit against the synthetic data above.")
            else:
                st.caption(
                    "Multi-start initial guesses centered on: synthetic input "
                    "parameters (log-uniform, 1 decade span)"
                )
                render_multi_start_results(
                    bleach_multi_result["results_df"], bleach_multi_result["param_names"],
                    bleach_multi_result["derived_names"], bleach_multi_result["true_values"],
                    include_nonconverged_bleach,
                )

    st.divider()
    # Fits the full model (Maturation_Model_Known_Bleaching_Pole.py) with
    # b = kb+kd fixed to the value entered below, using the synthetic data
    # from the Data tab (not the pure-decay data generated above).
    st.subheader("Known bleaching pole fit (full maturation model)")
    st.markdown(
        "Fits the full maturation model to the synthetic fluorescence trace "
        "generated in the **Data** tab, treating b = kb + kd as fixed and "
        "known (e.g. estimated above from the bleaching-only multi-start "
        "fit) rather than estimated. kd is fixed at 0 (growth halted) "
        "rather than fitted, so kb = b directly. Multi-start initial "
        "guesses are still centered on the synthetic input parameters "
        "(log-uniform, one decade span); since several of the raw fitted "
        "parameters (I0, km/k1/k2, alpha) are not individually "
        "identifiable on their own, only the identifiable derived "
        "quantities are reported below, compared to their synthetic input "
        "values."
    )

    data_kb = st.session_state.get("current_data")
    data_source_kb = st.session_state.get("current_data_source")
    synthetic_params_kb = st.session_state.get("synthetic_params")

    if (
        data_kb is None
        or data_source_kb != "Generate synthetic data from simulation"
        or synthetic_params_kb is None
    ):
        st.info(
            "This fit requires synthetic data. In the **Data** tab, select "
            "'Generate synthetic data from simulation' and click "
            "**Generate Synthetic Data** first."
        )
    else:
        fit_is_two_step_kb = synthetic_params_kb["is_two_step"]
        st.caption(
            "Using synthetic data generated from the "
            + ("2-step (I -> X -> M)" if fit_is_two_step_kb else "1-step (I -> M)")
            + " model (set in the Simulation tab)."
        )

        time_all_kb = data_kb["Time"].to_numpy(dtype=float)
        F_meas_kb = data_kb["Mean"].to_numpy(dtype=float)
        t_fit_kb = time_all_kb - time_all_kb[0]

        u_step_kb = st.number_input(
            "u - production rate (fixed during fit)",
            min_value=0.0, value=0.0, step=0.1, key="known_b_u_step",
            help="Suggested: match the Simulation tab's u (0 if translation was blocked).",
        )

        true_b_kb = synthetic_params_kb["kb"] + synthetic_params_kb["kd"]
        b_known = st.number_input(
            "b = kb + kd (known, fixed during fit)",
            min_value=0.0, value=float(true_b_kb), step=0.005, format="%.4f",
            key="known_b_value",
            help="Suggested: the b value estimated from the multi-start fit "
                 "above. Defaults to this synthetic dataset's true b for "
                 "validation.",
        )

        multi_col1_kb, multi_col2_kb = st.columns(2)
        with multi_col1_kb:
            n_multi_runs_kb = st.number_input(
                "Number of runs (N)", min_value=2, value=30, step=1, key="known_b_n_multi_runs",
            )
        with multi_col2_kb:
            include_nonconverged_kb = st.checkbox(
                "Include non-converged runs in plots/stats", value=False,
                key="known_b_include_nonconverged",
            )
        multi_seed_fix_kb = st.checkbox(
            "Fix random seed for initial guesses", value=False, key="known_b_multi_seed_fix",
        )
        multi_seed_kb = None
        if multi_seed_fix_kb:
            multi_seed_kb = int(st.number_input(
                "Initial-guess random seed", min_value=0, value=0, step=1, key="known_b_multi_seed_val",
            ))

        multi_fit_button_kb = st.button("Run Multi-Start Fit (N runs)", key="known_b_multi_fit_button")

        if multi_fit_button_kb:
            # kd is fixed at 0 in the fit itself (residuals_*_known_b); it's
            # still reported below as a (constant) derived value.
            fixed_kb = {"u": u_step_kb, "b": b_known}

            if fit_is_two_step_kb:
                centers_kb = [
                    synthetic_params_kb["I0"], synthetic_params_kb["k1"],
                    synthetic_params_kb["k2"], synthetic_params_kb["alpha"],
                ]
                param_names_kb = ["I0", "k1", "k2", "alpha"]
                bounds_kb = ([0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 10.0])
                residual_fn_kb = residuals_2step_known_b
            else:
                centers_kb = [
                    synthetic_params_kb["I0"], synthetic_params_kb["km"],
                    synthetic_params_kb["alpha"],
                ]
                param_names_kb = ["I0", "km", "alpha"]
                bounds_kb = ([0.0, 0.0, 0.1], [1e6, 5.0, 10.0])
                residual_fn_kb = residuals_1step_known_b

            with st.spinner(f"Running multi-start fit ({int(n_multi_runs_kb)} runs)..."):
                results_df_kb = run_multi_start(
                    residual_fn_kb, param_names_kb, centers_kb, bounds_kb,
                    args=(t_fit_kb, F_meas_kb, fixed_kb), n_runs=int(n_multi_runs_kb),
                    seed=multi_seed_kb,
                )

            results_df_kb["kd"] = 0.0
            if fit_is_two_step_kb:
                results_df_kb["a"] = results_df_kb["k1"] + results_df_kb["kd"]
                results_df_kb["c"] = results_df_kb["k2"] + results_df_kb["kd"]
                results_df_kb["G3"] = results_df_kb["alpha"] * results_df_kb["k1"] * results_df_kb["k2"]
                results_df_kb["G3*I0"] = results_df_kb["G3"] * results_df_kb["I0"]
                derived_names_kb = ["kd", "a", "c", "G3", "G3*I0"]
            else:
                results_df_kb["a"] = results_df_kb["km"] + results_df_kb["kd"]
                results_df_kb["G"] = results_df_kb["alpha"] * results_df_kb["km"]
                results_df_kb["G*I0"] = results_df_kb["G"] * results_df_kb["I0"]
                derived_names_kb = ["kd", "a", "G", "G*I0"]

            true_values_kb = {"kd": synthetic_params_kb["kd"]}
            if fit_is_two_step_kb:
                true_values_kb["a"] = synthetic_params_kb["k1"] + synthetic_params_kb["kd"]
                true_values_kb["c"] = synthetic_params_kb["k2"] + synthetic_params_kb["kd"]
                true_values_kb["G3"] = (
                    synthetic_params_kb["alpha"] * synthetic_params_kb["k1"] * synthetic_params_kb["k2"]
                )
                true_values_kb["G3*I0"] = true_values_kb["G3"] * synthetic_params_kb["I0"]
            else:
                true_values_kb["a"] = synthetic_params_kb["km"] + synthetic_params_kb["kd"]
                true_values_kb["G"] = synthetic_params_kb["alpha"] * synthetic_params_kb["km"]
                true_values_kb["G*I0"] = true_values_kb["G"] * synthetic_params_kb["I0"]

            st.session_state["known_b_multi_result"] = {
                "results_df": results_df_kb,
                "param_names": param_names_kb,
                "derived_names": derived_names_kb,
                "true_values": true_values_kb,
                "extra_info": {
                    "fit_is_two_step": bool(fit_is_two_step_kb),
                    "b_known": float(b_known),
                    "u_step": float(u_step_kb),
                },
            }

            _save_bleach_tab_history_entry()

        known_b_multi_result = st.session_state.get("known_b_multi_result")
        if known_b_multi_result is None:
            st.info("Click **Run Multi-Start Fit (N runs)** to fit against the synthetic data.")
        else:
            st.caption(
                "Multi-start initial guesses centered on: synthetic input "
                "parameters (Simulation tab values used to generate this "
                "data); b is fixed at the value entered above, not estimated. "
                "Running this fit saves everything currently run in this tab "
                "— this fit and the bleach-only fit above, if run — as one "
                "entry in the **Bleaching tab fit history** below."
            )

            results_df_kb = known_b_multi_result["results_df"]
            param_names_kb = known_b_multi_result["param_names"]
            derived_names_kb = known_b_multi_result["derived_names"]
            true_values_kb = known_b_multi_result["true_values"]

            n_converged_kb = int(results_df_kb["converged"].sum())
            st.markdown(
                f"**Multi-start fit complete: {n_converged_kb} / {len(results_df_kb)} runs converged**"
            )

            plot_df_kb = results_df_kb if include_nonconverged_kb else results_df_kb[results_df_kb["converged"]]

            if len(plot_df_kb) == 0:
                st.warning(
                    "No runs to display (no converged runs, and non-converged "
                    "runs are excluded)."
                )
            else:
                st.markdown("**Derived quantities across runs**")
                fig_kb = plot_histograms(plot_df_kb, derived_names_kb, true_values_kb, color="tab:purple")
                st.pyplot(fig_kb)

                st.markdown("**Synthetic input vs. fitted derived quantities**")
                summary_rows_kb = []
                for name in derived_names_kb:
                    vals = plot_df_kb[name].to_numpy(dtype=float)
                    mean = float(np.mean(vals))
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    cv = std / mean if mean != 0 else float("nan")
                    summary_rows_kb.append({
                        "Quantity": name,
                        "Synthetic input": true_values_kb[name],
                        "Fitted (mean across runs)": mean,
                        "Std dev": std,
                        "CV": cv,
                    })
                summary_df_kb = pd.DataFrame(summary_rows_kb)
                st.table(summary_df_kb.set_index("Quantity"))

                with st.expander("All multi-start fit results (raw table)"):
                    st.markdown(
                        "Each run shows two rows: its final **Fitted** values, and the "
                        "**Initial guess** it started from directly below. Raw fitted "
                        "parameters (I0, km/k1/k2, alpha) are included here for "
                        "inspection even though they are not individually identifiable "
                        "on their own; b is fixed at the value entered above, and kd is "
                        "fixed at 0 (growth halted) — neither is fitted."
                    )
                    detail_cols_kb = (
                        ["run", "Type"] + param_names_kb + derived_names_kb
                        + ["cost", "converged", "message", "nfev"]
                    )
                    detail_rows_kb = []
                    for _, r in results_df_kb.iterrows():
                        detail_rows_kb.append({
                            "run": int(r["run"]),
                            "Type": "Fitted",
                            **{name: r[name] for name in param_names_kb},
                            **{name: f"{r[name]:.6g}" for name in derived_names_kb},
                            "cost": f"{r['cost']:.6g}",
                            "converged": str(bool(r["converged"])),
                            "message": str(r["message"]),
                            "nfev": str(int(r["nfev"])),
                        })
                        detail_rows_kb.append({
                            "run": int(r["run"]),
                            "Type": "Initial guess",
                            **{name: r[f"{name}_init"] for name in param_names_kb},
                            **{name: "n/a" for name in derived_names_kb},
                            "cost": "n/a",
                            "converged": "n/a",
                            "message": "n/a",
                            "nfev": "n/a",
                        })
                    detail_df_kb = pd.DataFrame(detail_rows_kb, columns=detail_cols_kb)
                    st.dataframe(detail_df_kb, hide_index=True)
