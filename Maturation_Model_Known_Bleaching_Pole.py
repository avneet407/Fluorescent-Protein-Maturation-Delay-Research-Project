"""Maturation model with a known bleaching pole b = kb + kd.

F(t) = alpha*M(t) depends on kb and kd only through the lumped combination
b = kb + kd (the same b directly measurable from a bleaching-only
experiment, see Bleaching_Only_Model.py). If b has been separately measured
that way, it can be fixed rather than estimated, removing one free
parameter from the full least-squares fit; u and b are held fixed, and kd
is fixed at 0 (growth halted).

I0 and alpha are not individually identifiable from F(t) alone (only their
product with km/k1*k2 is -- the transfer function's gain numerator G times
I0, i.e. K = G*I0, G3*I0 for the 2-step model), so instead of fitting them
separately, the free parameters are km (or k1/k2) and K directly; I0 is
fixed at a reference value of 1 and alpha is derived from K at evaluation
time -- alpha itself is never reported, since it isn't identifiable on its
own. Same convention as the Variable Bleaching tab's joint fit
(variable_bleaching_fit.py).

    dI/dt = u - km*I - kd*I            (1-step)
    dM/dt = km*I - b*M

    dI/dt = u - k1*I - kd*I            (2-step)
    dX/dt = k1*I - k2*X - kd*X
    dM/dt = k2*X - b*M
"""

from scipy.integrate import solve_ivp

I0_REF = 1.0


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
    """x = [km, K] where K = G*I0 = alpha*km*I0 (I0 and alpha aren't fit
    individually since only their product with km is identifiable). kd is
    fixed at 0 (growth halted)."""
    km, K = x
    alpha = K / km if km > 0 else 0.0
    params = {"u": fixed["u"], "km": km, "kd": 0.0, "alpha": alpha, "b": fixed["b"]}
    _, I, M, F = simulate_1step_known_b(t, params, I0=I0_REF, M0=0.0)
    return F - F_meas


def residuals_2step_known_b(x, t, F_meas, fixed):
    """x = [k1, k2, K] where K = G3*I0 = alpha*k1*k2*I0. kd is fixed at 0
    (growth halted)."""
    k1, k2, K = x
    denom = k1 * k2
    alpha = K / denom if denom > 0 else 0.0
    params = {"u": fixed["u"], "k1": k1, "k2": k2, "kd": 0.0, "alpha": alpha, "b": fixed["b"]}
    _, I, X, M, F = simulate_2step_known_b(t, params, I0=I0_REF, X0=0.0, M0=0.0)
    return F - F_meas
