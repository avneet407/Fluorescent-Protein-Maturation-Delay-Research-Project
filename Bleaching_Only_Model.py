"""ODE definition for the bleaching-only model (fully matured pool, I(t) ~ 0).

Only photobleaching and dilution act on the mature pool:

    dM/dt = -kb*M - kd*M = -b*M,   b = kb + kd
    dB/dt =  kb*M - kd*B

with M(0) = M0. This has a closed-form solution M(t) = M0 * exp(-b*t), so
F(t) = alpha*M(t) = A*exp(-b*t) with A = alpha*M0.

M0 and alpha are not individually identifiable from F(t) alone (only their
product A = alpha*M0 is), so the least-squares fit below estimates b and A
directly rather than kb/M0/alpha separately; M0 is fixed at a reference
value of 1 and alpha is derived from A at evaluation time -- alpha itself is
never reported, since it isn't identifiable on its own. Same convention as
the Variable Bleaching tab's K = G*I0 parametrization.
"""

import numpy as np
from scipy.integrate import solve_ivp

M0_REF = 1.0


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
    """x = [b, A] where b = kb + kd (kd fixed at 0) and A = alpha*M0 (the
    only amplitude combination identifiable from F(t) alone)."""
    b, A = x
    alpha = A / M0_REF
    params = {"kb": b, "kd": 0.0, "alpha": alpha}
    _, M, B, F = simulate_bleach(t, params, M0=M0_REF, B0=0.0)
    return F - F_meas
