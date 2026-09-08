"""Joint least-squares fitting across multiple fluorescence traces that share
one maturation rate (km, or k1/k2) but otherwise have independent parameters.

Used by the Variable Bleaching tab. Every trace there is generated with its
own kb and initial conditions but the same km (or k1/k2), u, and alpha, so
fitting them jointly with km shared recovers one maturation rate estimate
instead of a separate (and less identifiable) one per trace.

I0 and alpha are not individually identifiable from F(t) alone when u = 0
(the model is linear in I0, so only the combination alpha * km * I0 -- the
transfer function's gain numerator G times I0, i.e. K = G*I0 -- sets the
amplitude; same "G"/"G3" convention used elsewhere in this app, e.g.
tab_fitting.py's comparison tables). So each trace's free parameters are kb
and K directly (K = G*I0 for 1-step, G3*I0 for 2-step); I0 is fixed at a
reference value of 1 and alpha is derived from K at evaluation time -- alpha
itself is never reported, since it isn't identifiable on its own.

kd is fixed at 0 (growth halted), and X0/M0/B0 are fixed at 0, matching
Maturation_Models.py's single-trace residuals_1step/residuals_2step.
"""

import numpy as np

from Maturation_Models import simulate_1step, simulate_2step

I0_REF = 1.0


def residuals_1step_shared_km(x, t, F_meas_list, fixed):
    """Joint residuals for the 1-step model across multiple traces sharing one km.

    x = [km, kb_1, K_1, kb_2, K_2, ...] (2 free params per trace: kb and
    K = G*I0 = alpha*km*I0, the only amplitude combination identifiable
    from F(t) alone when u = 0).
    fixed must contain: u (shared across all traces).
    F_meas_list: one 1D array per trace, all evaluated at the same `t`.
    """
    km = x[0]
    u = fixed["u"]
    resid_parts = []
    for i in range(len(F_meas_list)):
        kb_i, K_i = x[1 + 2 * i: 3 + 2 * i]
        alpha_i = K_i / km if km > 0 else 0.0
        params = {"u": u, "km": km, "kb": kb_i, "kd": 0.0, "alpha": alpha_i}
        _, I, M, B, F = simulate_1step(t, params, I0=I0_REF, M0=0.0, B0=0.0)
        resid_parts.append(F - F_meas_list[i])
    return np.concatenate(resid_parts)


def residuals_2step_shared_k(x, t, F_meas_list, fixed):
    """Joint residuals for the 2-step model across multiple traces sharing one k1, k2.

    x = [k1, k2, kb_1, K_1, kb_2, K_2, ...] (2 free params per trace: kb and
    K = G3*I0 = alpha*k1*k2*I0).
    fixed must contain: u (shared across all traces).
    """
    k1, k2 = x[0], x[1]
    u = fixed["u"]
    denom = k1 * k2
    resid_parts = []
    for i in range(len(F_meas_list)):
        kb_i, K_i = x[2 + 2 * i: 4 + 2 * i]
        alpha_i = K_i / denom if denom > 0 else 0.0
        params = {"u": u, "k1": k1, "k2": k2, "kb": kb_i, "kd": 0.0, "alpha": alpha_i}
        _, I, X, M, B, F = simulate_2step(t, params, I0=I0_REF, X0=0.0, M0=0.0, B0=0.0)
        resid_parts.append(F - F_meas_list[i])
    return np.concatenate(resid_parts)
