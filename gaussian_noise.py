"""Synthetic noisy data generation for the fluorescent protein maturation models.

Rather than adding measurement noise directly to the fluorescence trace, these
helpers perturb the maturation rate constant(s) (`km`, or `k1`/`k2`) and the
bleaching rate `kb` with independent Gaussian draws at every integration step,
then Euler-integrate the ODE system forward. This produces a synthetic trace
whose noise reflects rate-constant variability rather than pure readout noise.
"""

import numpy as np


def _noisy_rate(nominal, std, rng):
    """Draw a single Gaussian-perturbed rate constant, clipped at zero."""
    if std <= 0:
        return nominal
    return max(0.0, nominal + rng.normal(0.0, std))


def add_measurement_noise(F, std, seed=None):
    """Add Gaussian measurement noise directly to an intensity trace.

    Unlike the rate-constant noise above (which perturbs the underlying
    dynamics), this models readout/measurement noise (e.g. camera/shot
    noise) applied on top of the simulated Mean intensity values.
    """
    F = np.asarray(F, dtype=float)
    if std <= 0:
        return F
    rng = np.random.default_rng(seed)
    return F + rng.normal(0.0, std, size=F.shape)


def simulate_1step_noisy(t, params, I0, M0, B0, km_std=0.0, kb_std=0.0, seed=None):
    """Euler-integrate the 1-step model (I -> M -> B) with Gaussian noise on km and kb.

    params must contain: u, km, kb, kd, alpha.
    Returns (t, I, M, B, F).
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    n = len(t)

    I = np.empty(n)
    M = np.empty(n)
    B = np.empty(n)
    I[0], M[0], B[0] = I0, M0, B0

    u = params["u"]
    km = params["km"]
    kb = params["kb"]
    kd = params["kd"]
    alpha = params["alpha"]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        km_i = _noisy_rate(km, km_std, rng)
        kb_i = _noisy_rate(kb, kb_std, rng)

        dIdt = u - km_i * I[i - 1] - kd * I[i - 1]
        dMdt = km_i * I[i - 1] - kb_i * M[i - 1] - kd * M[i - 1]
        dBdt = kb_i * M[i - 1] - kd * B[i - 1]

        I[i] = I[i - 1] + dt * dIdt
        M[i] = M[i - 1] + dt * dMdt
        B[i] = B[i - 1] + dt * dBdt

    F = alpha * M
    return t, I, M, B, F


def simulate_2step_noisy(t, params, I0, X0, M0, B0, k1_std=0.0, k2_std=0.0, kb_std=0.0, seed=None):
    """Euler-integrate the 2-step model (I -> X -> M -> B) with Gaussian noise on k1, k2, and kb.

    params must contain: u, k1, k2, kb, kd, alpha.
    Returns (t, I, X, M, B, F).
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    n = len(t)

    I = np.empty(n)
    X = np.empty(n)
    M = np.empty(n)
    B = np.empty(n)
    I[0], X[0], M[0], B[0] = I0, X0, M0, B0

    u = params["u"]
    k1 = params["k1"]
    k2 = params["k2"]
    kb = params["kb"]
    kd = params["kd"]
    alpha = params["alpha"]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        k1_i = _noisy_rate(k1, k1_std, rng)
        k2_i = _noisy_rate(k2, k2_std, rng)
        kb_i = _noisy_rate(kb, kb_std, rng)

        dIdt = u - k1_i * I[i - 1] - kd * I[i - 1]
        dXdt = k1_i * I[i - 1] - k2_i * X[i - 1] - kd * X[i - 1]
        dMdt = k2_i * X[i - 1] - kb_i * M[i - 1] - kd * M[i - 1]
        dBdt = kb_i * M[i - 1] - kd * B[i - 1]

        I[i] = I[i - 1] + dt * dIdt
        X[i] = X[i - 1] + dt * dXdt
        M[i] = M[i - 1] + dt * dMdt
        B[i] = B[i - 1] + dt * dBdt

    F = alpha * M
    return t, I, X, M, B, F
