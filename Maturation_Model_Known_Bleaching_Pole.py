"""Maturation model with a known bleaching pole b = kb + kd.

F(t) = alpha*M(t) depends on kb and kd only through the lumped combination
b = kb + kd (the same b directly measurable from a bleaching-only
experiment, see Bleaching_Only_Model.py). If b has been separately measured
that way, it can be fixed rather than estimated, removing one free
parameter from the full least-squares fit: only I0 (and X0), km (or
k1/k2), kd, and alpha remain free; u and b are held fixed.

    dI/dt = u - km*I - kd*I            (1-step)
    dM/dt = km*I - b*M

    dI/dt = u - k1*I - kd*I            (2-step)
    dX/dt = k1*I - k2*X - kd*X
    dM/dt = k2*X - b*M
"""

from scipy.integrate import solve_ivp


def model_1step_known_b(t, y, params):
    I, M = y
    u = params["u"]
    km = params["km"]
    kd = params["kd"]
    b = params["b"]

    dIdt = u - km * I - kd * I
    dMdt = km * I - b * M
    return [dIdt, dMdt]


def model_2step_known_b(t, y, params):
    I, X, M = y
    u = params["u"]
    k1 = params["k1"]
    k2 = params["k2"]
    kd = params["kd"]
    b = params["b"]

    dIdt = u - k1 * I - kd * I
    dXdt = k1 * I - k2 * X - kd * X
    dMdt = k2 * X - b * M
    return [dIdt, dXdt, dMdt]


def simulate_1step_known_b(t, params, I0, M0):
    y0 = [I0, M0]
    sol = solve_ivp(model_1step_known_b, (t[0], t[-1]), y0, t_eval=t, args=(params,), method="RK45")
    I, M = sol.y
    F = params["alpha"] * M
    return sol.t, I, M, F


def simulate_2step_known_b(t, params, I0, X0, M0):
    y0 = [I0, X0, M0]
    sol = solve_ivp(model_2step_known_b, (t[0], t[-1]), y0, t_eval=t, args=(params,), method="RK45")
    I, X, M = sol.y
    F = params["alpha"] * M
    return sol.t, I, X, M, F


# ---------------------------------------------------------
# Residual functions for least-squares fitting (b held fixed via `fixed`)
# ---------------------------------------------------------

def residuals_1step_known_b(x, t, F_meas, fixed):
    I0, km, kd, alpha = x
    params = {"u": fixed["u"], "km": km, "kd": kd, "alpha": alpha, "b": fixed["b"]}
    _, I, M, F = simulate_1step_known_b(t, params, I0=I0, M0=0.0)
    return F - F_meas


def residuals_2step_known_b(x, t, F_meas, fixed):
    I0, k1, k2, kd, alpha = x
    params = {"u": fixed["u"], "k1": k1, "k2": k2, "kd": kd, "alpha": alpha, "b": fixed["b"]}
    _, I, X, M, F = simulate_2step_known_b(t, params, I0=I0, X0=0.0, M0=0.0)
    return F - F_meas
