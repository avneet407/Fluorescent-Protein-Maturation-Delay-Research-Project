"""ODE definitions for fluorescent protein maturation models."""

from scipy.integrate import solve_ivp


def model_1step(t, y, params):
    I, M, B = y
    u = params["u"]
    km = params["km"]
    kb = params["kb"]
    kd = params["kd"]

    dIdt = u - km * I - kd * I
    dMdt = km * I - kb * M - kd * M
    dBdt = kb * M - kd * B
    return [dIdt, dMdt, dBdt]


def model_2step(t, y, params):
    I, X, M, B = y
    u = params["u"]
    k1 = params["k1"]
    k2 = params["k2"]
    kb = params["kb"]
    kd = params["kd"]

    dIdt = u - k1 * I - kd * I
    dXdt = k1 * I - k2 * X - kd * X
    dMdt = k2 * X - kb * M - kd * M
    dBdt = kb * M - kd * B
    return [dIdt, dXdt, dMdt, dBdt]


# ---------------------------------------------------------
# Simulation helpers (used for fitting to data)
# ---------------------------------------------------------

def simulate_1step(t, params, I0, M0, B0):
    y0 = [I0, M0, B0]
    sol = solve_ivp(model_1step, (t[0], t[-1]), y0, t_eval=t, args=(params,), method="RK45")
    I, M, B = sol.y
    F = params["alpha"] * M
    return sol.t, I, M, B, F


def simulate_2step(t, params, I0, X0, M0, B0):
    y0 = [I0, X0, M0, B0]
    sol = solve_ivp(model_2step, (t[0], t[-1]), y0, t_eval=t, args=(params,), method="RK45")
    I, X, M, B = sol.y
    F = params["alpha"] * M
    return sol.t, I, X, M, B, F


# ---------------------------------------------------------
# Residual functions for least-squares fitting
# ---------------------------------------------------------

def residuals_1step(x, t, F_meas, fixed):
    I0, km, kb, kd, alpha = x
    params = {"u": fixed["u"], "km": km, "kb": kb, "kd": kd, "alpha": alpha}
    _, I, M, B, F = simulate_1step(t, params, I0=I0, M0=0.0, B0=0.0)
    return F - F_meas


def residuals_2step(x, t, F_meas, fixed):
    I0, k1, k2, kb, kd, alpha = x
    params = {"u": fixed["u"], "k1": k1, "k2": k2, "kb": kb, "kd": kd, "alpha": alpha}
    _, I, X, M, B, F = simulate_2step(t, params, I0=I0, X0=0.0, M0=0.0, B0=0.0)
    return F - F_meas
