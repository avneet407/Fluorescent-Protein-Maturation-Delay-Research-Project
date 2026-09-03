# --- Profile Likelihood tab: parameter identifiability sweeps ------------
# For a fit already run in Least Squares Fitting, sweeps one raw parameter
# or derived quantity (1D, below) or a joint (a, b) grid (2D, further down)
# across a range of fixed values, refitting everything else at each point,
# to see how sharply the data constrains it (a flat SSE curve/surface means
# poor identifiability).

from datetime import datetime

import numpy as np
import streamlit as st

from Maturation_Models import residuals_1step, residuals_2step
from history_store import append_profile_entry
from profile_likelihood import (
    profile_raw_parameter,
    profile_derived_quantity,
    DERIVED_QUANTITY_FORMULAS,
    compute_true_values,
)
from profile_likelihood_2D import profile_ab_2d

from app.shared import render_profile_likelihood_result, render_profile_2d_result


def render_profile_likelihood_tab():
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
        # Joint sweep over the two lumped rate constants a = km+kd (or
        # k1+kd) and b = kb+kd, producing an SSE contour instead of a curve.
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
