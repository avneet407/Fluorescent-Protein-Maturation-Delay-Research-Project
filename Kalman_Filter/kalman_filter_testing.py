"""
Kalman Filter + RTS Smoother: Fluorescent Protein Maturation Delay (1-step model)

Same model as 1-step_Kalman_Filter.py, followed by a backward
Rauch-Tung-Striebel (RTS) smoothing pass. The forward filter only uses
measurements up to each time step; the smoother revisits every step using
the whole trace (including future measurements).

The synthetic "true" data are generated separately from the filter:
  - u(t) is a continuous function of time, and the ODEs are integrated
    continuously (solve_ivp), so u varies within each measurement interval.
  - The true rate constants / alpha / noise (section 1) can differ from the
    calibrated values the filter assumes (section 2), as they would for real
    data, where a least-squares fit never recovers the truth exactly.
Only the measurements are discrete: fluorescence is sampled every dt.

State vector: x = [I, M, u]^T
a = k_m + k_d   (effective immature-protein decay rate)
b = k_b + k_d   (effective mature-protein decay rate)

Run this file directly (e.g. the Run button in VS Code) to see the plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm


# 1. True parameters (used only to generate the synthetic data)
km_true = 0.15
kb_true = 0.03
kd_true = 0.01
alpha_true = 2.0
sigma_F_true = 5.0 # standard deviation of the noise actually added to the measurements


def u_true(t):
    """True gene expression signal u(t), continuous in t (min)."""
    # return 5 + 3 * np.sin(2 * np.pi * t / 150)
    return 5 + 3 * np.sin(2*np.pi*t/120) + 1.5*np.sin((2*np.pi*t/15)+5)


# Measurement sampling
dt = 1.0 # Time between successive measurements (min)
n_steps = 300 # Number of measurements (run length = n_steps * dt)


# 2. Calibrated parameters assumed by the filter (would be obtained from least squares optimisation)
km = 0.15
kb = 0.03
kd = 0.01
alpha = 2.0 # Obtained by binomial partitioning method from Jan

# Process Noise
qI = 1e-4
qM = 1e-4
qu = 1
# Large qu: the model expects u to be able to jump by a large amount between measurements, purely from this noise term.
# Small qu: the model is saying u is expected to barely move between measurements.

sigma_F = 5.0 # the filter's assumed standard deviation of measurement noise
R = np.array([[sigma_F**2]]) # 1x1 array for measurement noise since only fluorescence is measured

a = km + kd
b = kb + kd


# 3. Hard-coded F from solving ODEs via Laplace.
# u(t) is modelled as random walk (all action is from process noise qu not F)

eps = 1e-10  # guards against division by zero if a == b (degenerate case)

if abs(a - b) < eps:
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


# 4. Defining H and Qc
# F_obs = alpha*M + noise
H = np.array([[0, alpha, 0]])

# dx/dt = Ax
A = np.array([
    [-a, 0,  1],
    [km, -b, 0],
    [0,  0,  0],
])

Qc = np.diag([qI, qM, qu])


# 5. Q via Van Loan
n = A.shape[0]
Mblock = np.block([[-A, Qc], [np.zeros((n, n)), A.T]]) * dt
expM = expm(Mblock)
Q = F @ expM[:n, n:]

# Sanity check: confirm hard-coded F matches expm(A*dt)
F_check = expm(A * dt)
print("Max difference between hard-coded F and expm(A*dt):",
      np.max(np.abs(F - F_check))) # Elemental differences


# 6. Synthetic data generation (will be replaced by real data)
# Integrate the true 1-step ODEs continuously from I = M = 0, then sample
# fluorescence every dt and add measurement noise.
def true_ode(t, y):
    I, M = y
    dIdt = u_true(t) - (km_true + kd_true) * I
    dMdt = km_true * I - (kb_true + kd_true) * M
    return [dIdt, dMdt]


t = np.arange(n_steps) * dt # measurement times
# max_step stops the solver stepping over fast features of u(t)
sol = solve_ivp(true_ode, (0, t[-1]), [0.0, 0.0], method="RK45",
                dense_output=True, rtol=1e-8, atol=1e-10, max_step=dt / 4)

true_I, true_M = sol.sol(t) # true states at the measurement times
true_u = u_true(t)

# Fine time grid, only for drawing the continuous true curves
t_fine = np.linspace(0, t[-1], 20 * n_steps)
true_I_fine, true_M_fine = sol.sol(t_fine)
true_u_fine = u_true(t_fine)

np.random.seed(0)
z_n = alpha_true * true_M + np.random.normal(0, sigma_F_true, size=n_steps)


# 7. Iteration Zero: initialize, then predict (no known input to add)
x = np.array([[0.0], [0.0], [0.0]])
P = np.diag([1e4, 1e4, 1e4]).astype(float) # huge initial covariance as we have no idea

x = F @ x
P = F @ P @ F.T + Q


# 8. Forward pass: Kalman filter, iterations 1..N
x_est_all = np.zeros((n_steps, 3))      # x_{n|n}: filtered (posterior) estimates
P_est_all = np.zeros((n_steps, 3, 3))   # P_{n|n}
x_pred_all = np.zeros((n_steps, 3))     # x_{n|n-1}: predicted (prior) estimates
P_pred_all = np.zeros((n_steps, 3, 3))  # P_{n|n-1}
x_pred_all[0] = x.flatten()
P_pred_all[0] = P

I3 = np.eye(3)
for n_i in range(n_steps):
    z = np.array([[z_n[n_i]]])

    # Update
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R) # Kalman Gain
    x = x + K @ (z - H @ x) # State Update
    P = (I3 - K @ H) @ P @ (I3 - K @ H).T + K @ R @ K.T # Covariance Update (Joseph form)

    x_est_all[n_i] = x.flatten()
    P_est_all[n_i] = P

    # Predict
    if n_i < n_steps - 1:
        x = F @ x  # State Extrapolation
        P = F @ P @ F.T + Q # Covariance Extrapolation
        x_pred_all[n_i + 1] = x.flatten()
        P_pred_all[n_i + 1] = P

I_est, M_est, u_est = x_est_all[:, 0], x_est_all[:, 1], x_est_all[:, 2]


# 9. Backward pass: RTS smoother
# Starts from the last step (smoothed == filtered there, as there is no future
# data) and works backwards using the smoother gain
#   C_n = P_{n|n} F^T (P_{n+1|n})^{-1}
x_smooth_all = np.zeros((n_steps, 3))   # x_{n|N}
P_smooth_all = np.zeros((n_steps, 3, 3))  # P_{n|N}
x_smooth_all[-1] = x_est_all[-1]
P_smooth_all[-1] = P_est_all[-1]

for n_i in range(n_steps - 2, -1, -1):
    P_filt_n = P_est_all[n_i]
    P_pred_np1 = P_pred_all[n_i + 1]

    # Solve instead of explicitly inverting P_pred_np1, for numerical stability
    C_n = np.linalg.solve(P_pred_np1.T, (P_filt_n @ F.T).T).T

    x_smooth_all[n_i] = x_est_all[n_i] + C_n @ (x_smooth_all[n_i + 1] - x_pred_all[n_i + 1])
    P_smooth_all[n_i] = P_filt_n + C_n @ (P_smooth_all[n_i + 1] - P_pred_np1) @ C_n.T

I_smooth, M_smooth, u_smooth = x_smooth_all[:, 0], x_smooth_all[:, 1], x_smooth_all[:, 2]


# 10. Results table and RMSE vs ground truth (at the measurement times)
print(f"{'n':>3} {'measured F':>12} {'filt. I':>10} {'filt. M':>10} {'filt. u':>10} "
      f"{'smth. I':>10} {'smth. M':>10} {'smth. u':>10}")
for n_i in range(n_steps):
    print(f"{n_i+1:3d} {z_n[n_i]:12.2f} {I_est[n_i]:10.3f} {M_est[n_i]:10.3f} {u_est[n_i]:10.3f} "
          f"{I_smooth[n_i]:10.3f} {M_smooth[n_i]:10.3f} {u_smooth[n_i]:10.3f}")


def rmse(est, true):
    return np.sqrt(np.mean((est - true) ** 2))


print(f"\n{'RMSE':<14} {'Filtered':>10} {'Smoothed':>10}")
for name, filt, smooth, true in [
    ("Fluorescence", alpha * M_est, alpha * M_smooth, alpha_true * true_M),
    ("I", I_est, I_smooth, true_I),
    ("M", M_est, M_smooth, true_M),
    ("u", u_est, u_smooth, true_u),
]:
    print(f"{name:<14} {rmse(filt, true):10.4f} {rmse(smooth, true):10.4f}")


# 11. Plotting: filtered and smoothed vs ground truth
# True curves are drawn on the fine grid (continuous); estimates exist only at the measurement times.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Fluorescence: measured vs. reconstructed (alpha * M)
ax = axes[0, 0]
ax.scatter(t, z_n, c='gray', s=15, alpha=0.5, label='Noisy fluorescence measurements')
ax.plot(t_fine, alpha_true * true_M_fine, 'g--', lw=1.5, label='True fluorescence (alpha_true * M_true)')
ax.plot(t, alpha * M_est, 'b-', lw=1.5, alpha=0.7, label='Filtered (alpha * M_est)')
ax.plot(t, alpha * M_smooth, 'r-', lw=1.5, label='Smoothed (alpha * M_smooth)')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Fluorescence')
ax.set_title('Fluorescence: measured vs. reconstructed')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Immature protein I(t)
ax = axes[0, 1]
ax.plot(t_fine, true_I_fine, 'g--', lw=1.5, label='True I(t)')
ax.plot(t, I_est, 'b-', lw=1.5, alpha=0.7, label='Filtered I(t)')
ax.plot(t, I_smooth, 'r-', lw=1.5, label='Smoothed I(t)')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Immature protein I')
ax.set_title('Immature protein pool')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Mature protein M(t)
ax = axes[1, 0]
ax.plot(t_fine, true_M_fine, 'g--', lw=1.5, label='True M(t)')
ax.plot(t, M_est, 'b-', lw=1.5, alpha=0.7, label='Filtered M(t)')
ax.plot(t, M_smooth, 'r-', lw=1.5, label='Smoothed M(t)')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Mature protein M')
ax.set_title('Mature protein pool')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Reconstructed synthesis rate u(t) -- the actual scientific target
ax = axes[1, 1]
ax.plot(t_fine, true_u_fine, 'g--', lw=1.5, label='True u(t) (gene expression)')
ax.plot(t, u_est, 'b-', lw=1.5, alpha=0.7, label='Filtered u(t)')
ax.plot(t, u_smooth, 'r-', lw=1.5, label='Smoothed u(t)')
ax.set_xlabel('Time (min)')
ax.set_ylabel('Synthesis rate u')
ax.set_title('Reconstructed gene expression rate (the target)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
