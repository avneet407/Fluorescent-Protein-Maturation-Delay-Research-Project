"""ODE definition for the bleaching-only model (fully matured pool, I(t) ~ 0).

Only photobleaching and dilution act on the mature pool:

    dM/dt = -kb*M - kd*M = -b*M,   b = kb + kd
    dB/dt =  kb*M - kd*B

with M(0) = M0. This has a closed-form solution M(t) = M0 * exp(-b*t).
"""

import numpy as np
from scipy.integrate import solve_ivp


def model_bleach(t, y, params):
    M, B = y
    kb = params["kb"]
    kd = params["kd"]

    dMdt = -kb * M - kd * M
    dBdt = kb * M - kd * B
    return [dMdt, dBdt]


def simulate_bleach(t, params, M0, B0):
    y0 = [M0, B0]
    sol = solve_ivp(model_bleach, (t[0], t[-1]), y0, t_eval=t, args=(params,), method="RK45")
    M, B = sol.y
    F = params["alpha"] * M
    return sol.t, M, B, F


def analytical_M(t, M0, kb, kd):
    """Closed-form M(t) = M0 * exp(-b*t), b = kb + kd."""
    b = kb + kd
    return M0 * np.exp(-b * t)


def analytical_F(t, A, kb, kd):
    """Closed-form F(t) = A * exp(-b*t), A = alpha * M0, b = kb + kd."""
    b = kb + kd
    return A * np.exp(-b * t)


# ---------------------------------------------------------
# Residual function for least-squares fitting
# ---------------------------------------------------------

def residuals_bleach(x, t, F_meas):
    M0, kb, kd, alpha = x
    params = {"kb": kb, "kd": kd, "alpha": alpha}
    _, M, B, F = simulate_bleach(t, params, M0=M0, B0=0.0)
    return F - F_meas
