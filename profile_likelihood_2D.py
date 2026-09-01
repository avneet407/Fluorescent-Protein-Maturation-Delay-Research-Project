"""2D profile likelihood for the (a, b) decay-rate pair.

`a` (`km + kd`, or `k1 + kd` for the 2-step model) and `b` (`kb + kd`) are the
two decay-rate combinations that appear as eigenvalues of the linear part of
the ODE system -- see the a/b exchange-degeneracy discussion in
`profile_likelihood.py`'s docstring. Individually profiling `a` or `b` can
show a spurious-looking second minimum where the fit has swapped which
physical rate constant (`km`/`k1` vs `kb`) is playing which decay role; this
module makes that degeneracy directly visible by sweeping both jointly.

Jointly fixing `a` and `b` eliminates `km` (or `k1`) and `kb` entirely
(`km = a - kd`, `kb = b - kd`), which leaves `kd` itself as a genuinely free
parameter alongside `I0`, `alpha` (and `k2` for the 2-step model) -- unlike
single-quantity profiling in `profile_likelihood.py`, which reparametrizes by
solving for one *computed* raw parameter and treats the others as free. At
each (a, b) grid point, a single `least_squares` run (initial guess drawn the
same log-uniform way `multi_start_fit.run_multi_start` draws one) re-optimizes
the remaining free parameters, and the SSE is recorded -- matching the
single-run-per-grid-point convention used throughout `profile_likelihood.py`.

If the fit can equally well explain the data with the two decay rates'
physical roles swapped, the resulting SSE grid shows a second low-SSE region
near the mirror point (b_true, a_true), in addition to the true
(a_true, b_true).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from multi_start_fit import sample_initial_guess


def _clip_to_bounds(x0, bounds, eps=1e-9):
    lower, upper = bounds
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.clip(x0, lower + eps, upper - eps)


def profile_ab_2d(
    residual_fn, param_names, fit_is_two_step, a_values, b_values, centers, bounds, args,
    seed=None, decade_span=1.0, max_nfev=1000,
):
    """Sweep `a` and `b` jointly over a grid, re-optimizing the remaining free parameters.

    `param_names`/`centers`/`bounds` are the full raw-parameter set as used
    elsewhere in the app (e.g. `["I0", "km", "kb", "kd", "alpha"]` for
    1-step, `["I0", "k1", "k2", "kb", "kd", "alpha"]` for 2-step). The two
    parameters `a`/`b` pin down (`km` or `k1`, and `kb`) are removed from the
    free set; `kd` becomes the free parameter that absorbs the (a, b)
    constraint.

    Returns a DataFrame with one row per (a, b) grid point: `a`, `b`, `sse`
    (sum of squared residuals at the fitted solution), `converged`, and the
    fitted value of every remaining free parameter (including the
    back-computed `km`/`k1`, `kb`, and `kd`).
    """
    decay_param = "k1" if fit_is_two_step else "km"
    fixed_names = {decay_param, "kb"}
    free_names = [n for n in param_names if n not in fixed_names]
    kd_idx = free_names.index("kd")

    lower, upper = bounds
    bounds_by_name = dict(zip(param_names, zip(lower, upper)))
    free_centers = [c for n, c in zip(param_names, centers) if n not in fixed_names]
    free_lower = [bounds_by_name[n][0] for n in free_names]
    free_upper = [bounds_by_name[n][1] for n in free_names]
    free_bounds = (free_lower, free_upper)

    rng = np.random.default_rng(seed)
    rows = []
    for a_val in a_values:
        for b_val in b_values:
            a_val = float(a_val)
            b_val = float(b_val)

            def wrapped_residual_fn(x_free, *rargs, _a=a_val, _b=b_val, _kd_idx=kd_idx):
                kd_val = x_free[_kd_idx]
                full = dict(zip(free_names, x_free))
                full[decay_param] = _a - kd_val
                full["kb"] = _b - kd_val
                x_full = np.array([full[n] for n in param_names])
                return residual_fn(x_full, *rargs)

            x0 = sample_initial_guess(free_centers, rng, decade_span=decade_span)
            x0 = _clip_to_bounds(x0, free_bounds)

            fit = least_squares(
                wrapped_residual_fn, x0, bounds=free_bounds, args=args, max_nfev=max_nfev,
            )

            free_best = dict(zip(free_names, fit.x))
            kd_best = free_best["kd"]

            row = {
                "a": a_val,
                "b": b_val,
                "sse": float(np.sum(fit.fun ** 2)),
                "converged": bool(fit.success),
            }
            row.update(free_best)
            row[decay_param] = float(a_val - kd_best)
            row["kb"] = float(b_val - kd_best)
            rows.append(row)

    return pd.DataFrame(rows)


def plot_profile_2d(profile_df):
    """Plain contour plot of SSE over the (a, b) grid: a and b varied, SSE as the third axis."""
    a_vals = np.sort(profile_df["a"].unique())
    b_vals = np.sort(profile_df["b"].unique())
    sse_grid = (
        profile_df.pivot(index="b", columns="a", values="sse")
        .reindex(index=b_vals, columns=a_vals)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    filled = ax.contourf(a_vals, b_vals, sse_grid, levels=15, cmap="viridis_r")
    fig.colorbar(filled, ax=ax, label="SSE")

    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title("2D profile likelihood: SSE over (a, b)")
    fig.tight_layout()
    return fig
