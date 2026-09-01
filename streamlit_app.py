import hashlib
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
from Bleaching_Only_Model import (
    model_bleach,
    simulate_bleach,
    residuals_bleach,
)
from Maturation_Model_Known_Bleaching_Pole import (
    residuals_1step_known_b,
    residuals_2step_known_b,
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
    simulate_bleach_noisy,
    add_measurement_noise,
)
from multi_start_fit import run_multi_start
from multi_start_plots import plot_histograms
from history_store import (
    load_multi_start_history,
    append_multi_start_entry,
    clear_multi_start_history,
    delete_multi_start_entry,
    load_profile_history,
    append_profile_entry,
    clear_profile_history,
    delete_profile_entry,
)
from profile_likelihood import (
    profile_raw_parameter,
    profile_derived_quantity,
    DERIVED_QUANTITY_FORMULAS,
    compute_true_values,
)
from profile_likelihood_2D import profile_ab_2d, plot_profile_2d


def compute_dataset_key(data_label, data_df):
    """Stable content-based identity for a loaded dataset.

    Used to group Profile Likelihood History entries by the underlying data
    they were run against: regenerating/re-uploading the same trace yields
    the same key, while any change to the actual Mean values yields a new one.
    """
    payload = data_label + "|" + ",".join(f"{v:.10g}" for v in data_df["Mean"].to_numpy(dtype=float))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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


def render_profile_likelihood_result(profile_df, profile_target, true_value=None):
    """Render the SSE-vs-value plot + raw data table for one profile likelihood run.

    Shared by the Profile Likelihood tab (right after a run) and the Profile
    Likelihood History tab (replaying a stored past run).
    """
    best_row = profile_df.loc[profile_df["sse"].idxmin()]
    st.success(
        f"Lowest SSE {best_row['sse']:.6g} at {profile_target} = {best_row['value']:.6g}"
    )

    fig_p, ax_p = plt.subplots(figsize=(8, 5))
    ax_p.plot(profile_df["value"], profile_df["sse"], "o-", color="tab:blue")
    ax_p.axvline(
        best_row["value"], color="tab:green", linestyle="--",
        label="Best-fit value (this profile)",
    )
    if true_value is not None:
        ax_p.axvline(
            true_value, color="tab:red", linestyle=":",
            label="Synthetic input (true value)",
        )
    ax_p.set_xlabel(profile_target)
    ax_p.set_ylabel("SSE (sum of squared residuals)")
    ax_p.set_title(f"Profile likelihood: {profile_target}")
    ax_p.legend(fontsize=8)
    fig_p.tight_layout()
    st.pyplot(fig_p)

    with st.expander("Profile likelihood raw data"):
        st.dataframe(profile_df, hide_index=True)


def render_profile_2d_result(profile_df):
    """Render the 2D (a, b) SSE contour plot + raw data table for one 2D profile likelihood run.

    Shared by the Profile Likelihood tab (right after a run) and the Profile
    Likelihood History tab (replaying a stored past run).
    """
    best_row = profile_df.loc[profile_df["sse"].idxmin()]
    st.success(
        f"Lowest SSE {best_row['sse']:.6g} at a = {best_row['a']:.6g}, b = {best_row['b']:.6g}"
    )

    fig = plot_profile_2d(profile_df)
    st.pyplot(fig)

    with st.expander("2D profile likelihood raw data"):
        st.dataframe(profile_df, hide_index=True)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(page_title="Fluorescent Protein Maturation", layout="wide")
st.title("Fluorescent Protein Maturation Delay Model")

sim_tab, upload_tab, bleach_tab, fit_tab, profile_tab, bode_tab, ms_history_tab, pl_history_tab = st.tabs(
    [
        "Simulation", "Data", "Bleaching Only Simulation",
        "Least Squares Fitting", "Profile Likelihood", "Bode Plot", "Multi-Start History",
        "Profile Likelihood History",
    ]
)

with sim_tab:
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
            "kd - degradation / dilution rate", min_value=0.0, value=0.01, step=0.005, format="%.4f",
            help="Suggested range: 0.001-0.05 /sec. Applies to all species. Illustrative default: 0.01.",
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

with upload_tab:
    st.markdown(
        "Provide the fluorescence trace to fit against in the **Least Squares "
        "Fitting** tab. Either upload real experimental data from a trench "
        "intensity CSV, or generate synthetic noisy data by perturbing the "
        "simulation's rate constants with Gaussian noise and re-integrating "
        "the model."
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
            "rate constants currently set in the **Simulation** tab. Two independent "
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
            st.session_state["noise_params"] = {
                "km_noise_std": None if is_two_step else km_noise_std,
                "k1_noise_std": k1_noise_std if is_two_step else None,
                "k2_noise_std": k2_noise_std if is_two_step else None,
                "kb_noise_std": kb_noise_std,
                "measurement_noise_std": measurement_noise_std,
                "seed": seed,
                "measurement_seed": measurement_seed,
            }

        if "synthetic_data_df" in st.session_state:
            data = st.session_state["synthetic_data_df"]
            data_label = "Synthetic data"
        else:
            st.info("Click **Generate Synthetic Data** to create a synthetic trace.")

    dataset_key = None
    if data is not None:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.plot(data["Time"], data["Mean"], marker="o", markersize=3)
        ax2.set_xlabel("Time (sec)")
        ax2.set_ylabel("Mean intensity")
        ax2.set_title(data_label)
        fig2.tight_layout()

        st.pyplot(fig2)
        st.dataframe(data)

        dataset_key = compute_dataset_key(data_label, data)
        is_synthetic_source = data_source == "Generate synthetic data from simulation"
        st.session_state.setdefault("dataset_info", {})[dataset_key] = {
            "label": data_label,
            "synthetic_params": st.session_state.get("synthetic_params") if is_synthetic_source else None,
            "noise_params": st.session_state.get("noise_params") if is_synthetic_source else None,
        }

    st.session_state["current_data"] = data
    st.session_state["current_data_label"] = data_label
    st.session_state["current_data_source"] = data_source
    st.session_state["current_dataset_key"] = dataset_key

with bleach_tab:
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
    st.session_state.setdefault("bleach_kd", 0.01)

    sim_run_version = st.session_state.get("sim_run_version")
    if sim_run_version is not None and st.session_state.get("bleach_synced_version") != sim_run_version:
        st.session_state["bleach_kb"] = st.session_state["sim_kb"]
        st.session_state["bleach_kd"] = st.session_state["sim_kd"]
        st.session_state["bleach_synced_version"] = sim_run_version

    if sim_run_version is not None:
        st.caption(
            "kb and kd below are synced from the last **Run Simulation** in "
            "the **Simulation** tab (kb="
            f"{st.session_state['sim_kb']:.4f}, kd={st.session_state['sim_kd']:.4f}"
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
            help="Suggested range: 0.001-0.05 /sec. Illustrative default: 0.01. Synced from the "
                 "Simulation tab's kd after a run, but can be overridden here.",
            key="bleach_kd",
        )
    with bleach_rc_cols[2]:
        alpha_bleach = st.number_input(
            "alpha - fluorescence scaling factor", min_value=0.0, value=1.0, step=0.1,
            help="Suggested: 1.0. Brightness per unit mature protein.",
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
        st.subheader("Multi-start least-squares fit")

        if bleach_synth is None:
            st.info("Generate synthetic data above to fit against.")
        else:
            st.markdown(
                "Repeats a least-squares fit of M0, kb, kd, and alpha to the "
                "synthetic data above from many independently randomized "
                "initial guesses, log-uniform over one decade centered on the "
                "synthetic input parameters, to check convergence robustness "
                "and parameter identifiability. Note that kb and kd trade off "
                "against each other since only b = kb + kd is identifiable "
                "from F(t) alone, and similarly alpha and M0 only enter as "
                "A = alpha * M0."
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
                param_names_bleach = ["M0", "kb", "kd", "alpha"]
                centers_bleach = [
                    bleach_synth_params["M0"], bleach_synth_params["kb"],
                    bleach_synth_params["kd"], bleach_synth_params["alpha"],
                ]
                bounds_bleach = ([0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 10.0])

                with st.spinner(f"Running multi-start fit ({int(n_multi_runs_bleach)} runs)..."):
                    results_df_bleach = run_multi_start(
                        residuals_bleach, param_names_bleach, centers_bleach, bounds_bleach,
                        args=(bleach_synth["t"], bleach_synth["F"]), n_runs=int(n_multi_runs_bleach),
                        seed=multi_seed_bleach,
                    )

                results_df_bleach["b"] = results_df_bleach["kb"] + results_df_bleach["kd"]
                results_df_bleach["A"] = results_df_bleach["alpha"] * results_df_bleach["M0"]
                derived_names_bleach = ["b", "A"]

                true_values_bleach = {
                    "M0": bleach_synth_params["M0"],
                    "kb": bleach_synth_params["kb"],
                    "kd": bleach_synth_params["kd"],
                    "alpha": bleach_synth_params["alpha"],
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
    st.subheader("Known bleaching pole fit (full maturation model)")
    st.markdown(
        "Fits the full maturation model to the synthetic fluorescence trace "
        "generated in the **Data** tab, treating b = kb + kd as fixed and "
        "known (e.g. estimated above from the bleaching-only multi-start "
        "fit) rather than estimated. Multi-start initial guesses are still "
        "centered on the synthetic input parameters (log-uniform, one "
        "decade span); since several of the raw fitted parameters (I0, "
        "km/k1/k2, kd, alpha) are not individually identifiable on their "
        "own, only the identifiable derived quantities are reported below, "
        "compared to their synthetic input values."
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
            fixed_kb = {"u": u_step_kb, "b": b_known}

            if fit_is_two_step_kb:
                centers_kb = [
                    synthetic_params_kb["I0"], synthetic_params_kb["k1"],
                    synthetic_params_kb["k2"], synthetic_params_kb["kd"], synthetic_params_kb["alpha"],
                ]
                param_names_kb = ["I0", "k1", "k2", "kd", "alpha"]
                bounds_kb = ([0.0, 0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 5.0, 10.0])
                residual_fn_kb = residuals_2step_known_b
            else:
                centers_kb = [
                    synthetic_params_kb["I0"], synthetic_params_kb["km"],
                    synthetic_params_kb["kd"], synthetic_params_kb["alpha"],
                ]
                param_names_kb = ["I0", "km", "kd", "alpha"]
                bounds_kb = ([0.0, 0.0, 0.0, 0.1], [1e6, 5.0, 5.0, 10.0])
                residual_fn_kb = residuals_1step_known_b

            with st.spinner(f"Running multi-start fit ({int(n_multi_runs_kb)} runs)..."):
                results_df_kb = run_multi_start(
                    residual_fn_kb, param_names_kb, centers_kb, bounds_kb,
                    args=(t_fit_kb, F_meas_kb, fixed_kb), n_runs=int(n_multi_runs_kb),
                    seed=multi_seed_kb,
                )

            if fit_is_two_step_kb:
                results_df_kb["a"] = results_df_kb["k1"] + results_df_kb["kd"]
                results_df_kb["G3"] = results_df_kb["alpha"] * results_df_kb["k1"] * results_df_kb["k2"]
                results_df_kb["G3*I0"] = results_df_kb["G3"] * results_df_kb["I0"]
                derived_names_kb = ["a", "G3", "G3*I0"]
            else:
                results_df_kb["a"] = results_df_kb["km"] + results_df_kb["kd"]
                results_df_kb["G"] = results_df_kb["alpha"] * results_df_kb["km"]
                results_df_kb["G*I0"] = results_df_kb["G"] * results_df_kb["I0"]
                derived_names_kb = ["a", "G", "G*I0"]

            true_values_kb = {}
            if fit_is_two_step_kb:
                true_values_kb["a"] = synthetic_params_kb["k1"] + synthetic_params_kb["kd"]
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
                "derived_names": derived_names_kb,
                "true_values": true_values_kb,
            }

        known_b_multi_result = st.session_state.get("known_b_multi_result")
        if known_b_multi_result is None:
            st.info("Click **Run Multi-Start Fit (N runs)** to fit against the synthetic data.")
        else:
            st.caption(
                "Multi-start initial guesses centered on: synthetic input "
                "parameters (Simulation tab values used to generate this "
                "data); b is fixed at the value entered above, not estimated."
            )

            results_df_kb = known_b_multi_result["results_df"]
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

with fit_tab:
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

with profile_tab:
    st.markdown(
        "Profile likelihood: sweep one raw parameter or derived quantity across "
        "a grid of fixed values. At each grid point, everything else is "
        "re-optimized (a single least-squares fit, starting from an initial "
        "guess drawn the same way as the multi-start fit) to fit as well as it "
        "can around that fixed value, and the resulting SSE is recorded. A "
        "sharply rising SSE away from the best-fit value means that "
        "parameter/quantity is well-identified by the data; a flat profile "
        "means it isn't — it can trade off against other parameters without "
        "hurting the fit."
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

        def _profile_default_value(name):
            if name in true_values_p:
                val = true_values_p[name]
            elif name in center_lookup:
                val = center_lookup[name]
            else:
                val = DERIVED_QUANTITY_FORMULAS[fit_is_two_step_p][name](center_lookup)
            return max(float(val), 1e-6)

        default_center = _profile_default_value(profile_target)

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

        fit_col1, fit_col2 = st.columns(2)
        with fit_col1:
            profile_max_nfev = st.number_input(
                "Max evaluations per fit", min_value=100, value=1000, step=100,
                key="profile_max_nfev",
            )
        with fit_col2:
            profile_seed_fix = st.checkbox(
                "Fix random seed", value=False, key="profile_seed_fix",
            )
        profile_seed = None
        if profile_seed_fix:
            profile_seed = int(st.number_input(
                "Profile random seed", min_value=0, value=0, step=1, key="profile_seed_val",
            ))

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
                    f"({int(grid_points)} grid points)..."
                ):
                    if is_derived:
                        profile_df = profile_derived_quantity(
                            residual_fn_p, param_names_p, profile_target, fit_is_two_step_p,
                            grid_values, centers_p, bounds_p, args_p,
                            seed=profile_seed, max_nfev=int(profile_max_nfev),
                        )
                    else:
                        profile_df = profile_raw_parameter(
                            residual_fn_p, param_names_p, profile_target,
                            grid_values, centers_p, bounds_p, args_p,
                            seed=profile_seed, max_nfev=int(profile_max_nfev),
                        )

                dataset_key_p = current_fit_data.get("dataset_key")
                dataset_info_p = st.session_state.get("dataset_info", {}).get(dataset_key_p, {})
                true_value_p = true_values_p.get(profile_target)

                append_profile_entry({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_key": dataset_key_p,
                    "dataset_label": current_fit_data.get("dataset_label", dataset_info_p.get("label", "")),
                    "synthetic_params": dataset_info_p.get("synthetic_params"),
                    "noise_params": dataset_info_p.get("noise_params"),
                    "run_key": current_fit_data.get("run_key"),
                    "fit_is_two_step": fit_is_two_step_p,
                    "u_step": current_fit_data["u_step"],
                    "time_start": current_fit_data.get("time_start"),
                    "time_end": current_fit_data.get("time_end"),
                    "baseline": current_fit_data.get("baseline"),
                    "profile_type": "1d",
                    "profile_target": profile_target,
                    "is_derived": is_derived,
                    "true_value": true_value_p,
                    "profile_df": profile_df,
                })

                st.session_state["profile_1d_result"] = {
                    "profile_df": profile_df,
                    "profile_target": profile_target,
                    "true_value": true_value_p,
                }

        profile_1d_result = st.session_state.get("profile_1d_result")
        if profile_1d_result is None:
            st.info("Click **Run Profile Likelihood** to compute a profile.")
        else:
            render_profile_likelihood_result(
                profile_1d_result["profile_df"], profile_1d_result["profile_target"],
                profile_1d_result["true_value"],
            )

        st.divider()
        st.subheader("2D Profile Likelihood: a vs b")
        st.markdown(
            "Jointly sweeps `a = km+kd` (or `k1+kd`) and `b = kb+kd` over the "
            "same grid, re-optimizing the remaining free parameters at each "
            "(a, b) point. This is the direct visual test for the a/b "
            "exchange degeneracy: a second low-SSE region near the mirror "
            "point (b_true, a_true), in addition to the true "
            "(a_true, b_true), means the fit can equally well explain the "
            "data with the two decay rates' physical roles swapped."
        )

        default_a = _profile_default_value("a")
        default_b = _profile_default_value("b")
        # Default both axes to span the same range, wide enough to cover both
        # a's and b's own default value -- so the mirror point (b_true, a_true)
        # is visible by default. Each axis can still be narrowed/widened
        # independently below.
        default_ab_lo = min(default_a, default_b) * 0.3
        default_ab_hi = max(default_a, default_b) * 3.0

        ab_bounds_col1, ab_bounds_col2 = st.columns(2)
        with ab_bounds_col1:
            st.markdown("**a bounds**")
            a_grid_min = st.number_input(
                "a grid min", value=default_ab_lo, format="%.6f", key="profile_a_grid_min",
            )
            a_grid_max = st.number_input(
                "a grid max", value=default_ab_hi, format="%.6f", key="profile_a_grid_max",
            )
        with ab_bounds_col2:
            st.markdown("**b bounds**")
            b_grid_min = st.number_input(
                "b grid min", value=default_ab_lo, format="%.6f", key="profile_b_grid_min",
            )
            b_grid_max = st.number_input(
                "b grid max", value=default_ab_hi, format="%.6f", key="profile_b_grid_max",
            )

        ab_grid_points = st.number_input(
            "Grid points per axis", min_value=3, value=8, step=1, key="profile_ab_grid_points",
        )

        st.caption(
            f"Worst case: {int(ab_grid_points)}² = "
            f"{int(ab_grid_points) ** 2} optimization runs. Reuses the "
            "\"Max evaluations per fit\" and random seed settings above."
        )

        run_2d_button = st.button("Run 2D Profile Likelihood (a, b)")

        if run_2d_button:
            if a_grid_max <= a_grid_min:
                st.error("a grid max must be greater than a grid min.")
            elif b_grid_max <= b_grid_min:
                st.error("b grid max must be greater than b grid min.")
            else:
                a_values = np.linspace(a_grid_min, a_grid_max, int(ab_grid_points))
                b_values = np.linspace(b_grid_min, b_grid_max, int(ab_grid_points))
                residual_fn_p = residuals_2step if fit_is_two_step_p else residuals_1step
                fixed_p = {"u": current_fit_data["u_step"]}
                args_p = (current_fit_data["t_fit"], current_fit_data["F_meas"], fixed_p)

                with st.spinner(
                    f"Computing 2D profile likelihood over a, b "
                    f"({int(ab_grid_points)}² grid points)..."
                ):
                    profile_2d_df = profile_ab_2d(
                        residual_fn_p, param_names_p, fit_is_two_step_p,
                        a_values, b_values, centers_p, bounds_p, args_p,
                        seed=profile_seed, max_nfev=int(profile_max_nfev),
                    )

                dataset_key_p2 = current_fit_data.get("dataset_key")
                dataset_info_p2 = st.session_state.get("dataset_info", {}).get(dataset_key_p2, {})
                true_a_p = true_values_p.get("a")
                true_b_p = true_values_p.get("b")

                append_profile_entry({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_key": dataset_key_p2,
                    "dataset_label": current_fit_data.get("dataset_label", dataset_info_p2.get("label", "")),
                    "synthetic_params": dataset_info_p2.get("synthetic_params"),
                    "noise_params": dataset_info_p2.get("noise_params"),
                    "run_key": current_fit_data.get("run_key"),
                    "fit_is_two_step": fit_is_two_step_p,
                    "u_step": current_fit_data["u_step"],
                    "time_start": current_fit_data.get("time_start"),
                    "time_end": current_fit_data.get("time_end"),
                    "baseline": current_fit_data.get("baseline"),
                    "profile_type": "2d",
                    "profile_target": "2D a, b",
                    "is_derived": True,
                    "true_value": None,
                    "true_a": true_a_p,
                    "true_b": true_b_p,
                    "profile_df": profile_2d_df,
                })

                st.session_state["profile_2d_result"] = {"profile_df": profile_2d_df}

        profile_2d_result = st.session_state.get("profile_2d_result")
        if profile_2d_result is None:
            st.info("Click **Run 2D Profile Likelihood (a, b)** to compute a profile.")
        else:
            render_profile_2d_result(profile_2d_result["profile_df"])

with bode_tab:
    st.markdown(
        "Frequency response of the maturation model, showing how strongly the "
        "reporter attenuates and delays gene expression fluctuations at each "
        "frequency, and the cutoff frequency above which dynamics are lost to "
        "maturation/bleaching. Plots the synthetic-data ground truth (if "
        "generated) and the most recent least-squares fit (if run) for "
        "comparison, using the **Simulation** tab's rate constants at the "
        "time **Plot Bode Response** is clicked."
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
            else:
                w, mag, phase = bode_1step(alpha, km, kb, kd, w)

            synthetic_params = st.session_state.get("synthetic_params")
            mag_syn = phase_syn = syn_label = None
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

            fit_result = st.session_state.get("fit_result")
            mag_fit = phase_fit = fit_label = None
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

            wc_numerical = numerical_cutoff(w, mag)
            if is_two_step:
                wc_analytical = analytical_cutoff_2step(k1, k2, kb, kd)
            else:
                wc_analytical = analytical_cutoff_1step(km, kb, kd)

        st.session_state["bode_result"] = {
            "w": w, "mag": mag, "phase": phase,
            "mag_syn": mag_syn, "phase_syn": phase_syn, "syn_label": syn_label,
            "mag_fit": mag_fit, "phase_fit": phase_fit, "fit_label": fit_label,
            "wc_numerical": wc_numerical, "wc_analytical": wc_analytical,
        }

    bode_result = st.session_state.get("bode_result")
    if bode_result is None:
        st.info("Set the frequency range and click **Plot Bode Response**.")
    else:
        br = bode_result
        fig4, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6))

        ax_mag.semilogx(br["w"], br["mag"])
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_xlabel("Frequency (rad/sec)")
        ax_mag.set_title("Bode Plot")
        ax_mag.grid(True, which="both", linestyle="--", alpha=0.5)

        ax_phase.semilogx(br["w"], br["phase"])
        ax_phase.set_ylabel("Phase (degrees)")
        ax_phase.set_xlabel("Frequency (rad/sec)")
        ax_phase.grid(True, which="both", linestyle="--", alpha=0.5)

        has_overlay = br["mag_syn"] is not None or br["mag_fit"] is not None
        if br["mag_syn"] is not None:
            ax_mag.semilogx(br["w"], br["mag_syn"], color="tab:green", linestyle="--", label=br["syn_label"])
            ax_phase.semilogx(br["w"], br["phase_syn"], color="tab:green", linestyle="--", label=br["syn_label"])
        if br["mag_fit"] is not None:
            ax_mag.semilogx(br["w"], br["mag_fit"], color="tab:red", linestyle=":", label=br["fit_label"])
            ax_phase.semilogx(br["w"], br["phase_fit"], color="tab:red", linestyle=":", label=br["fit_label"])

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

        cutoff_col1, cutoff_col2 = st.columns(2)
        if br["wc_numerical"] is not None:
            cutoff_col1.metric("Numerical -3 dB cutoff (rad/sec)", f"{br['wc_numerical']:.6f}")
        else:
            cutoff_col1.warning("Cutoff not reached within the selected frequency range.")

        if br["wc_analytical"] is not None:
            cutoff_col2.metric("Analytical -3 dB cutoff (rad/sec)", f"{br['wc_analytical']:.6f}")
        else:
            cutoff_col2.warning("No positive real cutoff root found.")

with ms_history_tab:
    st.markdown(
        "Every multi-start fit run, saved to disk (`multi_start_history.json`) "
        "so it survives app restarts — most recent first. Each entry keeps its "
        "fitted results and the synthetic-data ground truth (if any) exactly "
        "as they were at the time it ran — click **Display** to view its "
        "histograms and summary statistics again, even after changing "
        "Simulation tab parameters or running other fits since."
    )

    history = load_multi_start_history()

    if not history:
        st.info(
            "No multi-start fit runs saved yet. Run one from the "
            "**Run Multi-Start Fit (N runs)** button in the Least Squares "
            "Fitting tab."
        )
    else:
        displayed_ms = st.session_state.setdefault("ms_history_displayed", set())

        clear_col1, clear_col2 = st.columns([3, 1])
        with clear_col2:
            if st.button("Clear history", key="clear_ms_history"):
                clear_multi_start_history()
                st.session_state["ms_history_displayed"] = set()
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
                    delete_multi_start_entry(i)
                    st.session_state["ms_history_displayed"] = set()
                    st.rerun()

                if display_clicked:
                    displayed_ms.add(i)

                if i in displayed_ms:
                    render_multi_start_results(
                        entry["results_df"], entry["param_names"], entry["derived_names"],
                        entry["true_values"], hist_include_nonconverged,
                    )

with pl_history_tab:
    st.markdown(
        "Every profile likelihood run, saved to disk "
        "(`profile_likelihood_history.json`) so it survives app restarts. "
        "Runs are grouped by the **dataset** they were run against "
        "(regenerating/re-uploading the same trace groups new runs alongside "
        "past ones), then by **run** — one specific fitting region + model "
        "choice, since multiple parameters/quantities can each be profiled "
        "against the same run (e.g. Run 1: km, then Run 1: kb). Each dataset "
        "group records the synthetic ground truth and noise settings used to "
        "generate it, if applicable."
    )

    pl_history = load_profile_history()

    if not pl_history:
        st.info(
            "No profile likelihood runs saved yet. Run one from the "
            "**Profile Likelihood** tab."
        )
    else:
        displayed_pl = st.session_state.setdefault("pl_history_displayed", set())

        clear_col1, clear_col2 = st.columns([3, 1])
        with clear_col2:
            if st.button("Clear history", key="clear_pl_history"):
                clear_profile_history()
                st.session_state["pl_history_displayed"] = set()
                st.rerun()

        # Group entries by dataset, preserving first-seen (oldest-first) order.
        datasets = {}
        dataset_order = []
        for idx, entry in enumerate(pl_history):
            dk = entry["dataset_key"]
            if dk not in datasets:
                datasets[dk] = {
                    "label": entry["dataset_label"],
                    "synthetic_params": entry["synthetic_params"],
                    "noise_params": entry["noise_params"],
                    "entries": [],
                }
                dataset_order.append(dk)
            datasets[dk]["entries"].append((idx, entry))

        for dk in reversed(dataset_order):  # most recently seen dataset first
            group = datasets[dk]

            # Group this dataset's entries by run, numbering runs in first-seen order.
            runs = {}
            run_order = []
            for idx, entry in group["entries"]:
                rk = entry["run_key"]
                if rk not in runs:
                    runs[rk] = []
                    run_order.append(rk)
                runs[rk].append((idx, entry))

            with st.container(border=True):
                st.markdown(f"**Dataset: {group['label']}**")

                synthetic_params = group["synthetic_params"]
                noise_params = group["noise_params"]
                if synthetic_params is not None:
                    truth_bits = []
                    if synthetic_params.get("is_two_step"):
                        truth_bits.append(f"k1={synthetic_params['k1']:.4g}")
                        truth_bits.append(f"k2={synthetic_params['k2']:.4g}")
                    else:
                        truth_bits.append(f"km={synthetic_params['km']:.4g}")
                    truth_bits.append(f"I0={synthetic_params['I0']:.4g}")
                    truth_bits.append(f"kb={synthetic_params['kb']:.4g}")
                    truth_bits.append(f"kd={synthetic_params['kd']:.4g}")
                    truth_bits.append(f"alpha={synthetic_params['alpha']:.4g}")
                    st.caption("Synthetic input: " + ", ".join(truth_bits))

                    if noise_params is not None:
                        noise_bits = []
                        for key in ("km_noise_std", "k1_noise_std", "k2_noise_std", "kb_noise_std"):
                            val = noise_params.get(key)
                            if val is not None:
                                noise_bits.append(f"{key}={val:.4g}")
                        if noise_params.get("measurement_noise_std") is not None:
                            noise_bits.append(
                                f"measurement_noise_std={noise_params['measurement_noise_std']:.4g}"
                            )
                        seed_val = noise_params.get("seed")
                        noise_bits.append(f"seed={seed_val if seed_val is not None else 'random'}")
                        st.caption("Noise added: " + ", ".join(noise_bits))
                else:
                    st.caption("Experimental data (no synthetic ground truth).")

                for n, rk in enumerate(run_order, start=1):
                    run_entries = runs[rk]
                    first_entry = run_entries[0][1]
                    model_label = "2-step" if first_entry["fit_is_two_step"] else "1-step"
                    st.markdown(
                        f"Run {n} — {model_label} model, "
                        f"t=[{first_entry['time_start']:.4g}, {first_entry['time_end']:.4g}] sec, "
                        f"baseline={first_entry['baseline']:.4g}, u={first_entry['u_step']:.4g}"
                    )

                    for idx, entry in run_entries:
                        profile_type = entry.get("profile_type", "1d")
                        if profile_type == "2d":
                            label_line = (
                                f"&nbsp;&nbsp;**{entry['profile_target']}** "
                                f"— {entry['timestamp']}"
                            )
                        else:
                            kind_label = "derived quantity" if entry["is_derived"] else "raw parameter"
                            label_line = (
                                f"&nbsp;&nbsp;**{entry['profile_target']}** ({kind_label}) "
                                f"— {entry['timestamp']}"
                            )

                        row_col1, row_col2, row_col3 = st.columns([3, 1, 1])
                        with row_col1:
                            st.markdown(label_line)
                        with row_col2:
                            display_clicked_p = st.button("Display", key=f"pl_history_display_{idx}")
                        with row_col3:
                            delete_clicked_p = st.button("Delete", key=f"pl_history_delete_{idx}")

                        if delete_clicked_p:
                            delete_profile_entry(idx)
                            st.session_state["pl_history_displayed"] = set()
                            st.rerun()

                        if display_clicked_p:
                            displayed_pl.add(idx)

                        if idx in displayed_pl:
                            if profile_type == "2d":
                                render_profile_2d_result(entry["profile_df"])
                            else:
                                render_profile_likelihood_result(
                                    entry["profile_df"], entry["profile_target"], entry["true_value"],
                                )
