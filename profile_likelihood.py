"""Profile likelihood analysis for the maturation model least-squares fits.

For a chosen raw parameter (e.g. `km`) or derived quantity (e.g. `a = km + kd`),
sweeps it across a grid of fixed values. At each grid point, everything else is
re-optimized as best it can to compensate (with multiple random restarts, reusing
the multi-start fitting machinery), and the best (minimum) achieved
sum-of-squared-residuals cost is recorded. The resulting cost-vs-value curve is
the profile likelihood: a sharp rise away from the best-fit value means that
parameter/quantity is well-identified by the data; a flat profile means it isn't
(it can trade off against other parameters without hurting the fit) — this is a
more rigorous identifiability check than looking at multi-start scatter alone,
since here every other parameter is actively re-optimized at each grid point
rather than left wherever a single fit happened to land.

Both profiling functions reuse the exact same `residual_fn`, `bounds`, and
`args = (t, F_meas, fixed)` the rest of the app's fitting already uses, and the
same fixed dataset is passed through unchanged for every grid point and restart
— nothing here regenerates data.

Derived quantities are profiled by reparametrization rather than a general
constrained optimizer (e.g. SLSQP): one raw parameter is treated as "computed"
from the others plus the fixed target value (e.g. `kd = a_target - km`), and
`scipy.optimize.least_squares` optimizes over the remaining free raw
parameters, exactly like raw-parameter profiling. This was chosen over SLSQP
after testing showed SLSQP (with finite-difference gradients on this ODE-based,
multi-scale objective) badly violated the constraint and returned inflated
costs — reusing the same trust-region least-squares approach that already
works well elsewhere in this app is far more reliable.
"""

import numpy as np
import pandas as pd

from multi_start_fit import run_multi_start

# Derived-quantity formulas, keyed by `fit_is_two_step`, then by quantity name.
# Each formula takes a dict of {raw_param_name: value} and returns the derived value.
DERIVED_QUANTITY_FORMULAS = {
    False: {
        "a": lambda p: p["km"] + p["kd"],
        "b": lambda p: p["kb"] + p["kd"],
        "G": lambda p: p["alpha"] * p["km"],
        "G*I0": lambda p: p["alpha"] * p["km"] * p["I0"],
    },
    True: {
        "a": lambda p: p["k1"] + p["kd"],
        "c": lambda p: p["k2"] + p["kd"],
        "b": lambda p: p["kb"] + p["kd"],
        "G3": lambda p: p["alpha"] * p["k1"] * p["k2"],
        "G3*I0": lambda p: p["alpha"] * p["k1"] * p["k2"] * p["I0"],
    },
}

# Reparametrization spec for each derived quantity: which raw parameter is
# treated as "computed" from the others, and how to compute it from the free
# parameters (as a dict) plus the fixed target value for this grid point.
DERIVED_QUANTITY_REPARAM = {
    False: {
        "a": ("kd", lambda free, target: target - free["km"]),
        "b": ("kd", lambda free, target: target - free["kb"]),
        "G": ("km", lambda free, target: target / free["alpha"]),
        "G*I0": ("I0", lambda free, target: target / (free["alpha"] * free["km"])),
    },
    True: {
        "a": ("kd", lambda free, target: target - free["k1"]),
        "c": ("kd", lambda free, target: target - free["k2"]),
        "b": ("kd", lambda free, target: target - free["kb"]),
        "G3": ("k2", lambda free, target: target / (free["alpha"] * free["k1"])),
        "G3*I0": ("I0", lambda free, target: target / (free["alpha"] * free["k1"] * free["k2"])),
    },
}


def compute_true_values(synthetic_params, fit_is_two_step):
    """Ground-truth raw parameters + derived quantities for synthetic data, if available.

    Returns {} if `synthetic_params` is None or was generated with a different
    model (1-step vs 2-step) than `fit_is_two_step`.
    """
    true_values = {}
    if (
        synthetic_params is None
        or synthetic_params.get("is_two_step") != fit_is_two_step
        or synthetic_params.get("I0") is None
    ):
        return true_values

    true_values["I0"] = synthetic_params["I0"]
    true_values["kb"] = synthetic_params["kb"]
    true_values["kd"] = synthetic_params["kd"]
    true_values["alpha"] = synthetic_params["alpha"]
    if fit_is_two_step:
        true_values["k1"] = synthetic_params["k1"]
        true_values["k2"] = synthetic_params["k2"]
    else:
        true_values["km"] = synthetic_params["km"]

    for name, formula in DERIVED_QUANTITY_FORMULAS[fit_is_two_step].items():
        true_values[name] = formula(true_values)

    return true_values


def profile_raw_parameter(
    residual_fn, param_names, fix_name, grid_values, centers, bounds, args,
    n_restarts, seed=None, decade_span=1.0, max_nfev=1000,
):
    """Profile likelihood for one raw parameter.

    At each grid value, fixes `fix_name` to that value and re-runs multi-start
    least-squares (`multi_start_fit.run_multi_start`) over the remaining free
    parameters. Returns a DataFrame with one row per grid point: `value`
    (the fixed value), `cost` (minimum achieved across restarts),
    `converged` (whether that best run converged), and the best-fit value of
    every other free parameter.
    """
    fix_idx = param_names.index(fix_name)
    free_names = [n for n in param_names if n != fix_name]

    lower, upper = bounds
    free_centers = [c for n, c in zip(param_names, centers) if n != fix_name]
    free_lower = [b for n, b in zip(param_names, lower) if n != fix_name]
    free_upper = [b for n, b in zip(param_names, upper) if n != fix_name]
    free_bounds = (free_lower, free_upper)

    rows = []
    for i, value in enumerate(grid_values):
        def wrapped_residual_fn(x_free, *rargs, _value=float(value), _fix_idx=fix_idx):
            x_full = np.insert(np.asarray(x_free, dtype=float), _fix_idx, _value)
            return residual_fn(x_full, *rargs)

        run_seed = None if seed is None else seed + i
        results_df = run_multi_start(
            wrapped_residual_fn, free_names, free_centers, free_bounds, args,
            n_runs=n_restarts, seed=run_seed, decade_span=decade_span, max_nfev=max_nfev,
        )
        best = results_df.loc[results_df["cost"].idxmin()]

        row = {"value": float(value), "cost": float(best["cost"]), "converged": bool(best["converged"])}
        for name in free_names:
            row[name] = float(best[name])
        rows.append(row)

    return pd.DataFrame(rows)


def profile_derived_quantity(
    residual_fn, param_names, quantity_name, fit_is_two_step, grid_values, centers, bounds, args,
    n_restarts, seed=None, decade_span=1.0, max_nfev=1000,
):
    """Profile likelihood for one derived quantity, via reparametrization.

    One raw parameter (`computed_param`, per `DERIVED_QUANTITY_REPARAM`) is
    computed at every residual evaluation from the free parameters and the
    fixed grid value (e.g. `kd = a_target - km`), and least-squares optimizes
    over the remaining free raw parameters — exactly like `profile_raw_parameter`,
    just with a dynamically-computed "fixed" value instead of a constant one.

    Values of `computed_param` outside its own natural range (e.g. negative
    kd) are not rejected — the resulting poor fit residuals penalize them
    naturally, which keeps the optimization landscape smooth. Only genuine
    numerical singularities (e.g. dividing by a near-zero rate constant for
    `G`/`G*I0`/`G3`/`G3*I0`) are guarded against explicitly, since those would
    otherwise produce NaN/Inf that break the ODE solver.

    Returns a DataFrame with one row per grid point: `value`, `cost` (minimum
    achieved across restarts), `converged`, and the best-fit value of every
    raw parameter (including the computed one) at that point.
    """
    computed_param, solve_computed = DERIVED_QUANTITY_REPARAM[fit_is_two_step][quantity_name]
    free_names = [n for n in param_names if n != computed_param]

    lower, upper = bounds
    bounds_by_name = dict(zip(param_names, zip(lower, upper)))
    free_centers = [c for n, c in zip(param_names, centers) if n != computed_param]
    free_lower = [bounds_by_name[n][0] for n in free_names]
    free_upper = [bounds_by_name[n][1] for n in free_names]
    free_bounds = (free_lower, free_upper)

    rows = []
    for i, value in enumerate(grid_values):
        value = float(value)

        def wrapped_residual_fn(x_free, *rargs, _value=value):
            free = dict(zip(free_names, x_free))
            try:
                computed_value = solve_computed(free, _value)
            except ZeroDivisionError:
                computed_value = np.nan
            if not np.isfinite(computed_value):
                t_arr = np.asarray(rargs[0], dtype=float)
                return np.full_like(t_arr, 1e6)
            full = dict(free)
            full[computed_param] = computed_value
            x_full = np.array([full[n] for n in param_names])
            return residual_fn(x_full, *rargs)

        run_seed = None if seed is None else seed + i
        results_df = run_multi_start(
            wrapped_residual_fn, free_names, free_centers, free_bounds, args,
            n_runs=n_restarts, seed=run_seed, decade_span=decade_span, max_nfev=max_nfev,
        )
        best = results_df.loc[results_df["cost"].idxmin()]

        free_best = {name: float(best[name]) for name in free_names}
        computed_value = solve_computed(free_best, value)

        row = {"value": value, "cost": float(best["cost"]), "converged": bool(best["converged"])}
        row.update(free_best)
        row[computed_param] = float(computed_value)
        rows.append(row)

    return pd.DataFrame(rows)
