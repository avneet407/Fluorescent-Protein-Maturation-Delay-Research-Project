"""Kalman filter for fluorescent protein maturation delay (1-step and 2-step).

Ports the closed-form discretized state-space model from
Kalman_Filter/1-step_Kalman_Filter.py and Kalman_Filter/2-step_Kalman_Filter.py
into reusable functions, so the Streamlit Kalman Filter tab (and its history
replay) can run the same math with user-supplied parameters instead of the
hard-coded constants at the top of those scripts.

This module is filter-only: it does not generate ground truth or noisy
measurements itself. The noisy fluorescence trace it filters comes from the
Synthetic Gene Expression tab (synthetic_expression.py), generated from a
user-typed u(t) formula and its own "true" rate constants -- which may
differ from the (e.g. least-squares-calibrated) rate constants passed to
`run_kalman_1step`/`run_kalman_2step` here, mirroring how a real experiment's
ground truth is never exactly what a fit recovers.

State vector (u modelled as a random walk, driven only by process noise qu):
    1-step: x = [I, M, u]^T,       a = km + kd,               b = kb + kd
    2-step: x = [I, X, M, u]^T,    a1 = km1 + kd, a2 = km2 + kd, b = kb + kd
"""

import numpy as np
from scipy.linalg import expm

EPS = 1e-10  # guards against removable singularities when two poles coincide


def _van_loan_Q(A, Qc, F, dt):
    n = A.shape[0]
    block = np.block([[-A, Qc], [np.zeros((n, n)), A.T]]) * dt
    exp_block = expm(block)
    return F @ exp_block[:n, n:]


def build_1step_system(km, kb, kd, alpha, dt):
    """Returns (F, H, A) for the 1-step model's [I, M, u] state space."""
    a = km + kd
    b = kb + kd

    if abs(a - b) < EPS:
        F11 = np.exp(-a * dt)
        F13 = (1 - np.exp(-a * dt)) / a
        F21 = km * dt * np.exp(-a * dt)
        F22 = np.exp(-b * dt)
        F23 = (km / a) * ((1 - np.exp(-b * dt)) / b - dt * np.exp(-a * dt))
    else:
        F11 = np.exp(-a * dt)
        F13 = (1 - np.exp(-a * dt)) / a
        F21 = km * (np.exp(-a * dt) - np.exp(-b * dt)) / (b - a)
        F22 = np.exp(-b * dt)
        F23 = (km / a) * ((1 - np.exp(-b * dt)) / b - (np.exp(-a * dt) - np.exp(-b * dt)) / (b - a))

    F = np.array([
        [F11, 0, F13],
        [F21, F22, F23],
        [0, 0, 1],
    ])
    H = np.array([[0, alpha, 0]])
    A = np.array([
        [-a, 0, 1],
        [km, -b, 0],
        [0, 0, 0],
    ])
    return F, H, A


def build_2step_system(km1, km2, kb, kd, alpha, dt):
    """Returns (F, H, A) for the 2-step model's [I, X, M, u] state space."""
    a1 = km1 + kd
    a2 = km2 + kd
    b = kb + kd

    A = np.array([
        [-a1, 0, 0, 1],
        [km1, -a2, 0, 0],
        [0, km2, -b, 0],
        [0, 0, 0, 0],
    ])

    degenerate = abs(a1 - a2) < EPS or abs(a2 - b) < EPS or abs(a1 - b) < EPS
    if degenerate:
        # Several removable singularities can coincide in the 3-pole cascade;
        # rather than deriving every limiting case, fall back to the
        # numerically exact matrix exponential (same approach as the script).
        F = expm(A * dt)
    else:
        F11 = np.exp(-a1 * dt)
        F14 = (1 - np.exp(-a1 * dt)) / a1

        F21 = km1 * (np.exp(-a1 * dt) - np.exp(-a2 * dt)) / (a2 - a1)
        F22 = np.exp(-a2 * dt)
        F24 = km1 * (1 / (a1 * a2)
                     - np.exp(-a1 * dt) / (a1 * (a2 - a1))
                     + np.exp(-a2 * dt) / (a2 * (a2 - a1)))

        F31 = km1 * km2 * (np.exp(-a1 * dt) / ((a2 - a1) * (b - a1))
                            + np.exp(-a2 * dt) / ((a1 - a2) * (b - a2))
                            + np.exp(-b * dt) / ((a1 - b) * (a2 - b)))
        F32 = km2 * (np.exp(-a2 * dt) - np.exp(-b * dt)) / (b - a2)
        F33 = np.exp(-b * dt)
        F34 = km1 * km2 * (1 / (a1 * a2 * b)
                            - np.exp(-a1 * dt) / (a1 * (a2 - a1) * (b - a1))
                            - np.exp(-a2 * dt) / (a2 * (a1 - a2) * (b - a2))
                            - np.exp(-b * dt) / (b * (a1 - b) * (a2 - b)))

        F = np.array([
            [F11, 0, 0, F14],
            [F21, F22, 0, F24],
            [F31, F32, F33, F34],
            [0, 0, 0, 1],
        ])

    H = np.array([[0, 0, alpha, 0]])
    return F, H, A


def _run_filter_loop(F, H, Q, R, z_n, n_states):
    """Standard discrete Kalman filter loop shared by both models.

    Iteration zero predicts from x=0 with no measurement yet (matching the
    scripts' "no known input to add" comment -- u enters only as a random-walk
    state, not an exogenous term), then each step updates on z_n[i] and
    predicts forward to the next step.
    """
    n_steps = len(z_n)
    x = np.zeros((n_states, 1))
    P = np.diag([1e4] * n_states).astype(float)

    x = F @ x
    P = F @ P @ F.T + Q

    x_est_all = np.zeros((n_steps, n_states))
    for n_i in range(n_steps):
        z = np.array([[z_n[n_i]]])
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x = x + K @ (z - H @ x)
        I_mat = np.eye(n_states)
        P = (I_mat - K @ H) @ P @ (I_mat - K @ H).T + K @ R @ K.T

        x_est_all[n_i] = x.flatten()

        if n_i < n_steps - 1:
            x = F @ x
            P = F @ P @ F.T + Q

    return x_est_all


def run_kalman_1step(z_n, params):
    """Runs the 1-step Kalman filter over an externally supplied noisy trace.

    `z_n` is the noisy fluorescence measurement sequence to filter (e.g.
    generated by synthetic_expression.simulate_true_1step + measurement
    noise, in the Synthetic Gene Expression tab), sampled every `dt`.
    `params` keys: km, kb, kd, alpha, dt, qI, qM, qu, sigma_F (the filter's
    *assumed* measurement noise std, used to build R -- may differ from
    whatever noise std actually generated `z_n`).
    Raises ValueError if km+kd or kb+kd is not > 0 (division by zero in the
    closed-form F entries).
    """
    km, kb, kd, alpha = params["km"], params["kb"], params["kd"], params["alpha"]
    dt = params["dt"]

    a = km + kd
    b = kb + kd
    if a <= 0 or b <= 0:
        raise ValueError("km + kd and kb + kd must both be greater than 0.")

    F, H, A = build_1step_system(km, kb, kd, alpha, dt)
    Qc = np.diag([params["qI"], params["qM"], params["qu"]])
    Q = _van_loan_Q(A, Qc, F, dt)
    R = np.array([[params["sigma_F"] ** 2]])

    F_check = expm(A * dt)
    max_F_diff = float(np.max(np.abs(F - F_check)))

    x_est_all = _run_filter_loop(F, H, Q, R, np.asarray(z_n, dtype=float), n_states=3)
    I_est, M_est, u_est = x_est_all[:, 0], x_est_all[:, 1], x_est_all[:, 2]

    return {"I_est": I_est, "M_est": M_est, "u_est": u_est, "max_F_diff": max_F_diff}


def run_kalman_2step(z_n, params):
    """Runs the 2-step Kalman filter over an externally supplied noisy trace.

    `z_n` is the noisy fluorescence measurement sequence to filter (e.g.
    generated by synthetic_expression.simulate_true_2step + measurement
    noise, in the Synthetic Gene Expression tab), sampled every `dt`.
    `params` keys: km1, km2, kb, kd, alpha, dt, qI, qX, qM, qu, sigma_F (the
    filter's *assumed* measurement noise std, used to build R -- may differ
    from whatever noise std actually generated `z_n`).
    Raises ValueError if km1+kd, km2+kd, or kb+kd is not > 0.
    """
    km1, km2, kb, kd, alpha = (
        params["km1"], params["km2"], params["kb"], params["kd"], params["alpha"],
    )
    dt = params["dt"]

    a1 = km1 + kd
    a2 = km2 + kd
    b = kb + kd
    if a1 <= 0 or a2 <= 0 or b <= 0:
        raise ValueError("km1 + kd, km2 + kd, and kb + kd must all be greater than 0.")

    F, H, A = build_2step_system(km1, km2, kb, kd, alpha, dt)
    Qc = np.diag([params["qI"], params["qX"], params["qM"], params["qu"]])
    Q = _van_loan_Q(A, Qc, F, dt)
    R = np.array([[params["sigma_F"] ** 2]])

    F_check = expm(A * dt)
    max_F_diff = float(np.max(np.abs(F - F_check)))

    x_est_all = _run_filter_loop(F, H, Q, R, np.asarray(z_n, dtype=float), n_states=4)
    I_est, X_est, M_est, u_est = (
        x_est_all[:, 0], x_est_all[:, 1], x_est_all[:, 2], x_est_all[:, 3],
    )

    return {"I_est": I_est, "X_est": X_est, "M_est": M_est, "u_est": u_est, "max_F_diff": max_F_diff}
