"""Multi-start least-squares fitting.

Runs the same `scipy.optimize.least_squares` fit many times from
independently, randomly sampled initial guesses, to check whether the
optimizer converges to a consistent solution or gets stuck in different
local optima / degenerate parameter combinations that trade off against
each other (e.g. km and kd both increasing while km + kd stays fixed).
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def sample_log_uniform(center, rng, decade_span=1.0):
    """Draw one positive value, log-uniform over `decade_span` decades centered on `center`."""
    center = max(float(center), 1e-12)
    log_c = np.log10(center)
    half = decade_span / 2.0
    return 10.0 ** rng.uniform(log_c - half, log_c + half)


def sample_initial_guess(centers, rng, decade_span=1.0):
    """Draw a full parameter vector, each entry independently log-uniform around its center."""
    return np.array([sample_log_uniform(c, rng, decade_span) for c in centers])


def _clip_to_bounds(x0, bounds, eps=1e-9):
    lower, upper = bounds
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.clip(x0, lower + eps, upper - eps)


def run_multi_start(
    residual_fn, param_names, centers, bounds, args, n_runs, seed=None, decade_span=1.0, max_nfev=1000,
):
    """Run least_squares `n_runs` times from independent random initial guesses.

    Returns a DataFrame with one row per run: for each parameter in
    `param_names`, both its initial guess (`"{name}_init"`, the actual
    bounds-clipped x0 fed into the optimizer) and its final fitted value
    (`name`), plus `cost`, `converged`, `message`, `nfev`, `run`.
    Non-converged runs are kept (flagged via `converged=False`), not dropped.

    `max_nfev` raises the per-run evaluation budget above scipy's default
    (100 * n_params) — since initial guesses here are deliberately scattered
    widely across a decade, some starting points are far from a solution and
    would otherwise hit the default cap before converging.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_runs):
        x0 = sample_initial_guess(centers, rng, decade_span=decade_span)
        x0 = _clip_to_bounds(x0, bounds)

        fit = least_squares(residual_fn, x0, bounds=bounds, args=args, max_nfev=max_nfev)

        row = {}
        for name, guess, fitted in zip(param_names, x0, fit.x):
            row[f"{name}_init"] = float(guess)
            row[name] = float(fitted)
        row["cost"] = float(fit.cost)
        row["converged"] = bool(fit.success)
        row["message"] = fit.message
        row["nfev"] = int(fit.nfev)
        row["run"] = i
        rows.append(row)

    return pd.DataFrame(rows)
