"""
Kalman Filter: Fluorescent Protein Maturation Delay (1-step model)
-- F is hard-coded from the closed-form cascade solution (Laplace, solved
   equation-by-equation exploiting the triangular structure of A) --

State vector: x = [I, M, u]^T
a = k_m + k_d   (effective immature-protein decay rate)
b = k_b + k_d   (effective mature-protein decay rate)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# ---------------------------------------------------------------------
# 1. Calibrated parameters
# ---------------------------------------------------------------------
km = 0.15
kb = 0.03
kd = 0.01
alpha = 2.0
dt = 5.0

qI = 1e-4
qM = 1e-4
qu = 1.0          # key tuning parameter -- controls how fast u is allowed to drift

sigma_F = 5.0
R = np.array([[sigma_F**2]])

a = km + kd
b = kb + kd

# ---------------------------------------------------------------------
# 2. Hard-coded F, from the closed-form solution derived via Laplace
#    (solving du/dt, dI/dt, dM/dt sequentially, exploiting A's triangular
#    structure -- no coupling from M back into I or u)
# ---------------------------------------------------------------------
eps = 1e-10  # guards against division by zero if a == b (degenerate case)

if abs(a - b) < eps:
    # degenerate case a ~= b: the (b-a) terms need L'Hopital's rule.
    # limiting form: (e^{-at} - e^{-bt})/(b-a) -> t*e^{-at}  as b -> a
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
    [F11, 0,   F13],
    [F21, F22, F23],
    [0,   0,   1  ],
])

# ---------------------------------------------------------------------
# 3. H, and continuous-time Qc (still needed for Van Loan below)
# ---------------------------------------------------------------------
H = np.array([[0, alpha, 0]])

A = np.array([
    [-a, 0,  1],
    [km, -b, 0],
    [0,  0,  0],
])   # kept only to build Q via Van Loan -- F itself no longer comes from expm(A*dt)

Qc = np.diag([qI, qM, qu])

# ---------------------------------------------------------------------
# 4. Q via Van Loan (no equivalent triangular shortcut exists for Q,
#    since qI, qM, qu are three independent noise sources, unlike the
#    rocket's single shared noise channel)
# ---------------------------------------------------------------------
n = A.shape[0]
Mblock = np.block([[-A, Qc], [np.zeros((n, n)), A.T]]) * dt
expM = expm(Mblock)
Q = F @ expM[:n, n:]

# Sanity check: confirm hard-coded F matches expm(A*dt)
F_check = expm(A * dt)
print("Max difference between hard-coded F and expm(A*dt):",
      np.max(np.abs(F - F_check)))

# ---------------------------------------------------------------------
# 5. Generate synthetic data (replace this block with your real fluorescence
#    trace when available -- this is only here so the script runs end-to-end)
# ---------------------------------------------------------------------
np.random.seed(0)
n_steps = 60

t_true = np.arange(n_steps) * dt
true_u = 5 + 3 * np.sin(2 * np.pi * t_true / (n_steps * dt) * 2)   # synthetic gene expression

true_I = np.zeros(n_steps)
true_M = np.zeros(n_steps)
Ii, Mi = 0.0, 0.0
for k in range(n_steps):
    dI = true_u[k] - a * Ii
    dM = km * Ii - b * Mi
    Ii += dI * dt
    Mi += dM * dt
    true_I[k] = Ii
    true_M[k] = Mi

z_n = alpha * true_M + np.random.normal(0, sigma_F, size=n_steps)

# ---------------------------------------------------------------------
# 6. Iteration Zero: initialize, then predict (no known input to add)
# ---------------------------------------------------------------------
x = np.array([[0.0], [0.0], [0.0]])
P = np.diag([1e4, 1e4, 1e4]).astype(float)

x = F @ x
P = F @ P @ F.T + Q

# ---------------------------------------------------------------------
# 7. Iterations 1..N
# ---------------------------------------------------------------------
I_est = np.zeros(n_steps)
M_est = np.zeros(n_steps)
u_est = np.zeros(n_steps)

x_est_all = np.zeros((n_steps, 3))
P_est_all = np.zeros((n_steps, 3, 3))
x_pred_all = np.zeros((n_steps, 3))
P_pred_all = np.zeros((n_steps, 3, 3))

for n_i in range(n_steps):
    z = np.array([[z_n[n_i]]])

    # --- Update ---
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x = x + K @ (z - H @ x)
    I3 = np.eye(3)
    P = (I3 - K @ H) @ P @ (I3 - K @ H).T + K @ R @ K.T

    I_est[n_i] = x[0, 0]
    M_est[n_i] = x[1, 0]
    u_est[n_i] = x[2, 0]
    x_est_all[n_i] = x.flatten()
    P_est_all[n_i] = P

    # --- Predict --- no known input, just F and Q
    if n_i < n_steps - 1:
        x = F @ x
        P = F @ P @ F.T + Q
        x_pred_all[n_i + 1] = x.flatten()
        P_pred_all[n_i + 1] = P

print(f"{'n':>3} {'measured F':>12} {'est. I':>10} {'est. M':>10} {'est. u':>10}")
for n_i in range(n_steps):
    print(f"{n_i+1:3d} {z_n[n_i]:12.2f} {I_est[n_i]:10.3f} {M_est[n_i]:10.3f} {u_est[n_i]:10.3f}")


# ---------------------------------------------------------------------
# 8. RTS smoother
# ---------------------------------------------------------------------
def rts_smooth(x_est_all, P_est_all, x_pred_all, P_pred_all, F):
    n_steps = len(x_est_all)
    x_smooth = x_est_all.copy()
    P_smooth = P_est_all.copy()
    for k in range(n_steps - 2, -1, -1):
        C = P_est_all[k] @ F.T @ np.linalg.inv(P_pred_all[k + 1])
        x_smooth[k] = x_est_all[k] + C @ (x_smooth[k + 1] - x_pred_all[k + 1])
        P_smooth[k] = P_est_all[k] + C @ (P_smooth[k + 1] - P_pred_all[k + 1]) @ C.T
    return x_smooth, P_smooth

x_smooth, P_smooth = rts_smooth(x_est_all, P_est_all, x_pred_all, P_pred_all, F)
print("\nSmoothed u (final 5 points):", x_smooth[-5:, 2])

# ---------------------------------------------------------------------
# 9. Plot: fluorescence fit, and I / M / u trajectories
#    (filtered estimate vs. RTS-smoothed estimate vs. ground truth)
# ---------------------------------------------------------------------
t = np.arange(n_steps) * dt

u_smooth_std = np.sqrt(P_smooth[:, 2, 2])   # smoother's uncertainty band for u

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# --- Fluorescence: measured vs. reconstructed (alpha * M) ---
ax = axes[0, 0]
ax.scatter(t, z_n, c='gray', s=15, alpha=0.5, label='Noisy fluorescence measurements')
ax.plot(t, alpha * true_M, 'g--', lw=1.5, label='True fluorescence (alpha * M_true)')
ax.plot(t, alpha * M_est, 'b-', lw=2, label='Filtered (alpha * M_est)')
ax.plot(t, alpha * x_smooth[:, 1], 'r-', lw=2, label='Smoothed (alpha * M_smooth)')
ax.set_xlabel('Time')
ax.set_ylabel('Fluorescence')
ax.set_title('Fluorescence: measured vs. reconstructed')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Immature protein I(t) ---
ax = axes[0, 1]
ax.plot(t, true_I, 'g--', lw=1.5, label='True I(t)')
ax.plot(t, I_est, 'b-', lw=2, label='Filtered I(t)')
ax.plot(t, x_smooth[:, 0], 'r-', lw=2, label='Smoothed I(t)')
ax.set_xlabel('Time')
ax.set_ylabel('Immature protein I')
ax.set_title('Immature protein pool')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Mature protein M(t) ---
ax = axes[1, 0]
ax.plot(t, true_M, 'g--', lw=1.5, label='True M(t)')
ax.plot(t, M_est, 'b-', lw=2, label='Filtered M(t)')
ax.plot(t, x_smooth[:, 1], 'r-', lw=2, label='Smoothed M(t)')
ax.set_xlabel('Time')
ax.set_ylabel('Mature protein M')
ax.set_title('Mature protein pool')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# --- Reconstructed synthesis rate u(t) -- the actual scientific target ---
ax = axes[1, 1]
ax.plot(t, true_u, 'g--', lw=1.5, label='True u(t) (gene expression)')
ax.plot(t, u_est, 'b-', lw=1.5, alpha=0.6, label='Filtered u(t)')
ax.plot(t, x_smooth[:, 2], 'r-', lw=2, label='Smoothed u(t)')
ax.fill_between(t, x_smooth[:, 2] - u_smooth_std, x_smooth[:, 2] + u_smooth_std,
                 color='r', alpha=0.15, label='Smoothed +/-1 std dev')
ax.set_xlabel('Time')
ax.set_ylabel('Synthesis rate u')
ax.set_title('Reconstructed gene expression rate (the target)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()