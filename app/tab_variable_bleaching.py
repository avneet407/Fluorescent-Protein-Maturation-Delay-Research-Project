# --- Variable Bleaching tab -------------------------------------------
# Fixes kd at 0 (growth halted) and lets the user add multiple parameter
# sets -- each its own kb (photobleaching rate) and initial conditions
# (I0, X0, M0, B0), with its own label/key -- then overlays the resulting
# fluorescence F(t) curve for every set on one graph. Only km/k1,k2, u, and
# alpha are held fixed across every set.

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from Maturation_Models import simulate_1step, simulate_2step
from gaussian_noise import simulate_1step_noisy, simulate_2step_noisy, add_measurement_noise
from variable_bleaching_fit import residuals_1step_shared_km, residuals_2step_shared_k
from multi_start_fit import run_multi_start
from multi_start_plots import plot_histograms
from history_store import append_variable_bleaching_entry


def render_variable_bleaching_tab():
    st.markdown(
        "Simulates the full maturation model with kd fixed at 0 (growth "
        "halted), overlaying the fluorescence F(t) curve for several "
        "parameter sets on one graph. Each set has its own kb "
        "(photobleaching rate) and initial conditions (I0, X0, M0, B0); "
        "only km (or k1/k2), u, and alpha below are shared across every set."
    )

    st.session_state.setdefault("vb_kb_sets", [])
    kb_sets = st.session_state["vb_kb_sets"]

    model_choice = st.radio(
        "Maturation model",
        options=["1-step (I -> M)", "2-step (I -> X -> M)"],
        horizontal=True, key="vb_model",
    )
    is_two_step = model_choice.startswith("2")

    st.subheader("Fixed rate constants (kd = 0, growth halted)")
    rc_cols = st.columns(3)
    with rc_cols[0]:
        if is_two_step:
            k1 = st.number_input(
                "k1 - rate I -> X", min_value=0.0, value=0.20, step=0.01, format="%.3f",
                help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.20.",
                key="vb_k1",
            )
            k2 = st.number_input(
                "k2 - rate X -> M", min_value=0.0, value=0.10, step=0.01, format="%.3f",
                help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.10.",
                key="vb_k2",
            )
        else:
            km = st.number_input(
                "km - rate I -> M", min_value=0.0, value=0.15, step=0.01, format="%.3f",
                help="Suggested range: 0.05-0.5 /sec. Illustrative default: 0.15.",
                key="vb_km",
            )
    with rc_cols[1]:
        u = st.number_input(
            "u - production rate", min_value=0.0, value=0.0, step=0.1, format="%.3f",
            help="Suggested: 0 if translation is blocked (e.g. chloramphenicol chase). "
                 "Use a positive value to model ongoing translation.",
            key="vb_u",
        )
    with rc_cols[2]:
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
    st.subheader("Add a parameter set")
    st.markdown("kb and the initial conditions below vary per set; label it to tell it apart on the graph.")

    new_label = st.text_input(
        "Label (optional)", key="vb_new_label",
        placeholder=f"Set {len(kb_sets) + 1}",
    )

    if is_two_step:
        set_cols = st.columns(5)
    else:
        set_cols = st.columns(4)
    with set_cols[0]:
        new_kb = st.number_input(
            "kb - photobleaching rate (M -> B)", min_value=0.0, value=0.02, step=0.005, format="%.4f",
            key="vb_new_kb",
        )
    with set_cols[1]:
        new_I0 = st.number_input(
            "I0 - immature protein at t=0", min_value=0.0, value=100.0, step=10.0,
            help="Suggested: 100. Pool of just-translated protein present when translation is halted.",
            key="vb_new_I0",
        )
    new_X0 = 0.0
    if is_two_step:
        with set_cols[2]:
            new_X0 = st.number_input(
                "X0 - intermediate protein at t=0", min_value=0.0, value=0.0, step=10.0,
                help="Suggested: 0. Usually no protein has reached the intermediate stage yet.",
                key="vb_new_X0",
            )
    m0_col = set_cols[3] if is_two_step else set_cols[2]
    b0_col = set_cols[4] if is_two_step else set_cols[3]
    with m0_col:
        new_M0 = st.number_input(
            "M0 - mature (fluorescent) protein at t=0", min_value=0.0, value=0.0, step=10.0,
            help="Suggested: 0. No protein has finished maturing at t=0.",
            key="vb_new_M0",
        )
    with b0_col:
        new_B0 = st.number_input(
            "B0 - bleached protein at t=0", min_value=0.0, value=0.0, step=10.0,
            help="Suggested: 0. No photobleaching has occurred yet.",
            key="vb_new_B0",
        )

    if st.button("Run (add to graph)", type="primary", key="vb_add_kb"):
        label = new_label.strip() or f"Set {len(kb_sets) + 1}"
        kb_sets.append({
            "label": label, "kb": new_kb,
            "I0": new_I0, "X0": new_X0, "M0": new_M0, "B0": new_B0,
        })

    if kb_sets:
        with st.expander(f"Parameter sets on graph ({len(kb_sets)})", expanded=False):
            for i, ks in enumerate(kb_sets):
                row_cols = st.columns([5, 1])
                with row_cols[0]:
                    details = (
                        f"kb={ks['kb']:.4f}, I0={ks['I0']:.4g}, X0={ks['X0']:.4g}, "
                        f"M0={ks['M0']:.4g}, B0={ks['B0']:.4g}"
                        if is_two_step else
                        f"kb={ks['kb']:.4f}, I0={ks['I0']:.4g}, M0={ks['M0']:.4g}, B0={ks['B0']:.4g}"
                    )
                    st.write(f"**{ks['label']}** — {details}")
                with row_cols[1]:
                    if st.button("Remove", key=f"vb_remove_{i}"):
                        kb_sets.pop(i)
                        st.rerun()
            if st.button("Clear all", key="vb_clear_all"):
                st.session_state["vb_kb_sets"] = []
                st.rerun()

    if not kb_sets:
        st.info("Add at least one parameter set above and click **Run** to see the fluorescence curves.")
        return

    t_eval = np.linspace(0, t_end, int(n_points))

    fig, ax = plt.subplots(figsize=(9, 5))
    for ks in kb_sets:
        if is_two_step:
            params = {"u": u, "k1": k1, "k2": k2, "kb": ks["kb"], "kd": kd, "alpha": alpha}
            _, _, _, _, _, F = simulate_2step(
                t_eval, params, I0=ks["I0"], X0=ks["X0"], M0=ks["M0"], B0=ks["B0"],
            )
        else:
            params = {"u": u, "km": km, "kb": ks["kb"], "kd": kd, "alpha": alpha}
            _, _, _, _, F = simulate_1step(t_eval, params, I0=ks["I0"], M0=ks["M0"], B0=ks["B0"])
        ax.plot(t_eval, F, label=ks["label"])

    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("F = alpha * M (fluorescence)")
    ax.set_title(
        ("2-step" if is_two_step else "1-step")
        + " maturation model — fluorescence across parameter sets (kd = 0)"
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
        "Mean intensity trace, applied to every parameter set's "
        "fluorescence curve above."
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
                        t_eval, params_syn, I0=ks["I0"], X0=ks["X0"], M0=ks["M0"], B0=ks["B0"],
                        k1_std=k1_noise_std, k2_std=k2_noise_std, kb_std=kb_noise_std,
                        seed=curve_seed,
                    )
                else:
                    params_syn = {"u": u, "km": km, "kb": ks["kb"], "kd": kd, "alpha": alpha}
                    _, _, _, _, F_syn = simulate_1step_noisy(
                        t_eval, params_syn, I0=ks["I0"], M0=ks["M0"], B0=ks["B0"],
                        km_std=km_noise_std, kb_std=kb_noise_std,
                        seed=curve_seed,
                    )
                F_syn = add_measurement_noise(F_syn, measurement_noise_std, seed=measurement_seed)
                synthetic_results.append({
                    "label": ks["label"], "kb": ks["kb"],
                    "I0": ks["I0"], "X0": ks["X0"], "M0": ks["M0"], "B0": ks["B0"],
                    "t": t_eval, "F": F_syn,
                })

        st.session_state["vb_synthetic_results"] = synthetic_results
        st.session_state["vb_synthetic_meta"] = {
            "is_two_step": is_two_step,
            "km": None if is_two_step else km,
            "k1": k1 if is_two_step else None,
            "k2": k2 if is_two_step else None,
            "u": u, "alpha": alpha,
            "noise_params": {
                "km_noise_std": None if is_two_step else km_noise_std,
                "k1_noise_std": k1_noise_std if is_two_step else None,
                "k2_noise_std": k2_noise_std if is_two_step else None,
                "kb_noise_std": kb_noise_std,
                "measurement_noise_std": measurement_noise_std,
                "seed": seed,
            },
        }

    synthetic_results = st.session_state.get("vb_synthetic_results")
    if synthetic_results is None:
        st.info(
            "Click **Generate Synthetic Data** to create noisy fluorescence "
            "traces for the parameter sets above."
        )
    else:
        fig_syn, ax_syn = plt.subplots(figsize=(9, 5))
        for sr in synthetic_results:
            ax_syn.plot(sr["t"], sr["F"], marker="o", markersize=2, linewidth=1, label=sr["label"])
        ax_syn.set_xlabel("Time (sec)")
        ax_syn.set_ylabel("Mean intensity (synthetic)")
        ax_syn.set_title("Synthetic data across parameter sets")
        ax_syn.legend(fontsize=8)
        fig_syn.tight_layout()
        st.pyplot(fig_syn)

        combined_rows = [
            {"Set": sr["label"], "kb": sr["kb"], "Time": t_val, "Mean": f_val}
            for sr in synthetic_results
            for t_val, f_val in zip(sr["t"], sr["F"])
        ]
        st.dataframe(pd.DataFrame(combined_rows))

    st.divider()
    st.subheader("Least squares fit across traces (shared km)")
    st.markdown(
        "Jointly fits every trace generated above against its own synthetic "
        "data: one km (or k1, k2) shared across all traces, with kb and "
        "K estimated independently per trace. Repeats the fit from many "
        "independently randomized initial guesses (log-uniform, one decade "
        "span, centered on the synthetic input parameters) to check "
        "whether the optimizer converges to the same km regardless of "
        "starting point -- same multi-start convention used elsewhere in "
        "this app. u is held fixed and kd is fixed at 0, matching the rest "
        "of this tab. I0 and alpha are not individually identifiable from "
        "F(t) alone (only their product with km is), so instead of fitting "
        "them separately, each trace's amplitude is fit as K = G\\*I0 = "
        "alpha\\*km\\*I0 (G3\\*I0 for the 2-step model) -- the transfer "
        "function's gain numerator times I0, same convention used "
        "elsewhere in this app. The fit itself always assumes X0 = M0 = "
        "B0 = 0 (only I0 is nonzero at t=0), matching every other "
        "least-squares fit in this app; a trace generated above with "
        "nonzero M0/X0/B0 will still fit kb well but its recovered K will "
        "be biased, since that starting fluorescence gets absorbed into "
        "the amplitude term instead."
    )

    if synthetic_results is None:
        st.info("Generate synthetic data above to fit against.")
        return

    synth_meta = st.session_state["vb_synthetic_meta"]
    fit_is_two_step = synth_meta["is_two_step"]
    st.caption(
        "Fitting against the "
        + ("2-step (I -> X -> M)" if fit_is_two_step else "1-step (I -> M)")
        + " synthetic data generated above."
    )

    fit_col1, fit_col2 = st.columns(2)
    with fit_col1:
        n_multi_runs_vb = st.number_input(
            "Number of runs (N)", min_value=1, value=10, step=1, key="vb_n_multi_runs",
            help="Fewer runs than other multi-start fits in this app by default, since "
                 "each run here jointly simulates every trace above.",
        )
    with fit_col2:
        include_nonconverged_vb = st.checkbox(
            "Include non-converged runs in plots/stats", value=False, key="vb_include_nonconverged",
        )

    fit_seed_fix = st.checkbox(
        "Fix random seed for initial guesses", value=False, key="vb_fit_seed_fix",
    )
    fit_seed = None
    if fit_seed_fix:
        fit_seed = int(st.number_input(
            "Initial-guess random seed", min_value=0, value=0, step=1, key="vb_fit_seed_val",
        ))

    fit_button = st.button("Run Multi-Start Fit (N runs)", type="primary", key="vb_fit_button")

    if fit_button:
        t_fit = synthetic_results[0]["t"]
        F_meas_list = [sr["F"] for sr in synthetic_results]
        n_traces = len(synthetic_results)
        fixed = {"u": synth_meta["u"]}
        trace_labels = [sr["label"] for sr in synthetic_results]

        trace_param_names = []
        for i in range(n_traces):
            trace_param_names.append(f"kb_{i + 1}")
            trace_param_names.append(f"K_{i + 1}")

        if fit_is_two_step:
            shared_names = ["k1", "k2"]
            shared_true = [synth_meta["k1"], synth_meta["k2"]]
            gain_true = synth_meta["alpha"] * synth_meta["k1"] * synth_meta["k2"]
            residual_fn = residuals_2step_shared_k
        else:
            shared_names = ["km"]
            shared_true = [synth_meta["km"]]
            gain_true = synth_meta["alpha"] * synth_meta["km"]
            residual_fn = residuals_1step_shared_km

        param_names_vb = shared_names + trace_param_names
        centers_vb = shared_true + [v for sr in synthetic_results for v in (sr["kb"], gain_true * sr["I0"])]
        n_shared = len(shared_names)
        lower = [0.0] * n_shared + [0.0, 0.0] * n_traces
        upper = [5.0] * n_shared + [5.0, 1e8] * n_traces

        true_values_vb = dict(zip(shared_names, shared_true))
        for i, sr in enumerate(synthetic_results):
            true_values_vb[f"kb_{i + 1}"] = sr["kb"]
            true_values_vb[f"K_{i + 1}"] = gain_true * sr["I0"]

        with st.spinner(f"Running multi-start fit ({int(n_multi_runs_vb)} runs)..."):
            results_df_vb = run_multi_start(
                residual_fn, param_names_vb, centers_vb, (lower, upper),
                args=(t_fit, F_meas_list, fixed), n_runs=int(n_multi_runs_vb),
                max_nfev=2000,
            )

        vb_fit_result = {
            "is_two_step": fit_is_two_step,
            "shared_names": shared_names,
            "trace_labels": trace_labels,
            "param_names": param_names_vb,
            "true_values": true_values_vb,
            "results_df": results_df_vb,
        }
        st.session_state["vb_fit_result"] = vb_fit_result

        append_variable_bleaching_entry({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_two_step": fit_is_two_step,
            "n_traces": n_traces,
            "n_runs": int(n_multi_runs_vb),
            "shared_names": shared_names,
            "trace_labels": trace_labels,
            "param_names": param_names_vb,
            "true_values": true_values_vb,
            "results_df": results_df_vb,
            "u": synth_meta["u"],
            "alpha": synth_meta["alpha"],
            "noise_params": synth_meta["noise_params"],
            "fit_seed": fit_seed,
        })

    fit_result = st.session_state.get("vb_fit_result")
    if fit_result is None:
        st.info("Click **Run Multi-Start Fit (N runs)** to jointly fit all traces above.")
    else:
        results_df_vb = fit_result["results_df"]
        n_converged_vb = int(results_df_vb["converged"].sum())
        st.markdown(
            f"**Multi-start fit complete: {n_converged_vb} / {len(results_df_vb)} runs converged**"
        )

        plot_df_vb = results_df_vb if include_nonconverged_vb else results_df_vb[results_df_vb["converged"]]

        if len(plot_df_vb) == 0:
            st.warning(
                "No runs to display (no converged runs, and non-converged "
                "runs are excluded)."
            )
        else:
            shared_names = fit_result["shared_names"]
            shared_label = "k1, k2" if fit_result["is_two_step"] else "km"
            st.markdown(f"**{shared_label} across runs** (shared across all traces)")
            fig_shared = plot_histograms(plot_df_vb, shared_names, fit_result["true_values"], color="tab:blue")
            st.pyplot(fig_shared)

            summary_rows_vb = []
            for name in shared_names:
                vals = plot_df_vb[name].to_numpy(dtype=float)
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                summary_rows_vb.append({
                    "Quantity": name,
                    "Synthetic input": fit_result["true_values"][name],
                    "Mean (across runs)": mean,
                    "Std dev": std,
                    "CV": std / mean if mean != 0 else float("nan"),
                })
            st.markdown("**Synthetic input vs. jointly fitted shared parameters**")
            st.dataframe(pd.DataFrame(summary_rows_vb).set_index("Quantity"))

            trace_labels = fit_result["trace_labels"]
            kb_display_names = [f"kb ({label})" for label in trace_labels]
            K_display_names = [f"K ({label})" for label in trace_labels]
            display_df_vb = plot_df_vb.copy()
            display_true_values = dict(fit_result["true_values"])
            for i, label in enumerate(trace_labels):
                kb_name, K_name = f"kb_{i + 1}", f"K_{i + 1}"
                display_df_vb[kb_display_names[i]] = display_df_vb[kb_name]
                display_df_vb[K_display_names[i]] = display_df_vb[K_name]
                display_true_values[kb_display_names[i]] = fit_result["true_values"][kb_name]
                display_true_values[K_display_names[i]] = fit_result["true_values"][K_name]

            st.markdown("**kb across runs, per trace**")
            fig_kb = plot_histograms(display_df_vb, kb_display_names, display_true_values, color="tab:purple")
            st.pyplot(fig_kb)

            st.markdown("**K across runs, per trace** (K = G\\*I0 / G3\\*I0)")
            fig_K = plot_histograms(display_df_vb, K_display_names, display_true_values, color="tab:green")
            st.pyplot(fig_K)

            st.markdown(
                "**Per-trace fitted parameters** (kb, and K = G\\*I0 / G3\\*I0 — I0 and "
                "alpha aren't fit individually), mean ± std across runs"
            )
            trace_rows_vb = []
            for i, label in enumerate(fit_result["trace_labels"]):
                kb_name, K_name = f"kb_{i + 1}", f"K_{i + 1}"
                kb_vals = plot_df_vb[kb_name].to_numpy(dtype=float)
                K_vals = plot_df_vb[K_name].to_numpy(dtype=float)
                trace_rows_vb.append({
                    "Label": label,
                    "kb (true)": fit_result["true_values"][kb_name],
                    "kb (mean fitted)": float(np.mean(kb_vals)),
                    "kb (std)": float(np.std(kb_vals, ddof=1)) if len(kb_vals) > 1 else 0.0,
                    "K (true)": fit_result["true_values"][K_name],
                    "K (mean fitted)": float(np.mean(K_vals)),
                    "K (std)": float(np.std(K_vals, ddof=1)) if len(K_vals) > 1 else 0.0,
                })
            st.dataframe(pd.DataFrame(trace_rows_vb).set_index("Label"))

            with st.expander(f"All {len(results_df_vb)} runs (raw table)"):
                st.markdown(
                    "One row per run: each parameter's initial guess "
                    "(`{name}_init`) and final fitted value, plus fit "
                    "diagnostics. Consistent fitted values for "
                    f"{shared_label} across runs (regardless of the "
                    "`_init` starting point) indicate the fit is robust, "
                    "not stuck in different local optima."
                )
                st.dataframe(results_df_vb, hide_index=True)
