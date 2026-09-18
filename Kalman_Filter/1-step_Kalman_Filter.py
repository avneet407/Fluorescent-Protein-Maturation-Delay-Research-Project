"""
Kalman Filter: Fluorescent Protein Maturation Delay (1-step model)

State vector: x = [I, M, u]^T
a = k_m + k_d   (effective immature-protein decay rate)
b = k_b + k_d   (effective mature-protein decay rate)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


# 1. Calibrated parameters (would be obtained from least squares optimisation)
km = 0.15
kb = 0.03
kd = 0.01
alpha = 2.0 # Obtained by binomial partitioning method from Jan
dt = 5.0 # Time between successive measurements

# Process Noise
qI = 1e-4
qM = 1e-4
qu = 1.0
# Large qu: the model expects u to be able to jump by a large amount between measurements, purely from this noise term.
# Small qu: the model is saying u is expected to barely move between measurements — any large jump in the true trajectory is treated as statistically unlikely by construction, regardless of what the data shows.

sigma_F = 5.0 # standard deviation of measurement noise
R = np.array([[sigma_F**2]]) # 1x1 array for measurement noise since only fluroescence is measured

a = km + kd
b = kb + kd


# 2. Hard-coded F from solving ODEs via laplace.
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


# 3. Defining H and Qc
# F_obs = alpha*M + noise
H = np.array([[0, alpha, 0]])

# dx/dt = Ax
A = np.array([
    [-a, 0,  1],
    [km, -b, 0],
    [0,  0,  0],
])

Qc = np.diag([qI, qM, qu])


# 4. Q via Van Loan
n = A.shape[0]
Mblock = np.block([[-A, Qc], [np.zeros((n, n)), A.T]]) * dt
expM = expm(Mblock)
Q = F @ expM[:n, n:]

# Sanity check: confirm hard-coded F matches expm(A*dt)
F_check = expm(A * dt)
print("Max difference between hard-coded F and expm(A*dt):",
      np.max(np.abs(F - F_check))) # Elemental differences

# Synthetic data generation for fluroescence tarce (will be replaced by real data)
np.random.seed(0)
n_steps = 60

t_true = np.arange(n_steps) * dt
true_u = 5 + 3 * np.sin(2 * np.pi * t_true / (n_steps * dt) * 2)   # synthetic gene expression

true_I = np.zeros(n_steps)
true_M = np.zeros(n_steps)
Ii, Mi = 0.0, 0.0
for k in range(n_steps):
    Ii_next = F11 * Ii + F13 * true_u[k]
    Mi_next = F21 * Ii + F22 * Mi + F23 * true_u[k]
    Ii, Mi = Ii_next, Mi_next
    true_I[k] = Ii
    true_M[k] = Mi

z_n = alpha * true_M + np.random.normal(0, sigma_F, size=n_steps)


# 6. Iteration Zero: initialize, then predict (no known input to add)
x = np.array([[0.0], [0.0], [0.0]])
P = np.diag([1e4, 1e4, 1e4]).astype(float) # huge initial covariance as we have no idea

x = F @ x
P = F @ P @ F.T + Q


# 7. Iterations 1..N
I_est = np.zeros(n_steps)
M_est = np.zeros(n_steps)
u_est = np.zeros(n_steps)

x_est_all = np.zeros((n_steps, 3))
P_est_all = np.zeros((n_steps, 3, 3))
x_pred_all = np.zeros((n_steps, 3))
P_pred_all = np.zeros((n_steps, 3, 3))

for n_i in range(n_steps):
    z = np.array([[z_n[n_i]]])

    # Update
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R) # Kalman Gain (how much to trust measurement vs prediction based on uncertainties)
    x = x + K @ (z - H @ x) # State Update
    I3 = np.eye(3)
    P = (I3 - K @ H) @ P @ (I3 - K @ H).T + K @ R @ K.T # Covariance Update

    I_est[n_i] = x[0, 0]
    M_est[n_i] = x[1, 0]
    u_est[n_i] = x[2, 0]
    x_est_all[n_i] = x.flatten() # Filtered (Posterior) estimate incorporating measurement at step n_i
    P_est_all[n_i] = P

    # Predict
    if n_i < n_steps - 1:
        x = F @ x  # State Extrapolation
        P = F @ P @ F.T + Q # Covariance Extrapolation
        x_pred_all[n_i + 1] = x.flatten() # Predicted (Prior) Estimate for step n_i+1 before measurement has been incorporated
        P_pred_all[n_i + 1] = P

print(f"{'n':>3} {'measured F':>12} {'est. I':>10} {'est. M':>10} {'est. u':>10}")
for n_i in range(n_steps):
    print(f"{n_i+1:3d} {z_n[n_i]:12.2f} {I_est[n_i]:10.3f} {M_est[n_i]:10.3f} {u_est[n_i]:10.3f}")


# Plotting fluroescence fit and I, M, U trajectories for filtered vs ground truth
t = np.arange(n_steps) * dt

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Fluorescence: measured vs. reconstructed (alpha * M) 
ax = axes[0, 0]
ax.scatter(t, z_n, c='gray', s=15, alpha=0.5, label='Noisy fluorescence measurements')
ax.plot(t, alpha * true_M, 'g--', lw=1.5, label='True fluorescence (alpha * M_true)')
ax.plot(t, alpha * M_est, 'b-', lw=2, label='Filtered (alpha * M_est)')
ax.set_xlabel('Time')
ax.set_ylabel('Fluorescence')
ax.set_title('Fluorescence: measured vs. reconstructed')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Immature protein I(t) 
ax = axes[0, 1]
ax.plot(t, true_I, 'g--', lw=1.5, label='True I(t)')
ax.plot(t, I_est, 'b-', lw=2, label='Filtered I(t)')
ax.set_xlabel('Time')
ax.set_ylabel('Immature protein I')
ax.set_title('Immature protein pool')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Mature protein M(t) 
ax = axes[1, 0]
ax.plot(t, true_M, 'g--', lw=1.5, label='True M(t)')
ax.plot(t, M_est, 'b-', lw=2, label='Filtered M(t)')
ax.set_xlabel('Time')
ax.set_ylabel('Mature protein M')
ax.set_title('Mature protein pool')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Reconstructed synthesis rate u(t) -- the actual scientific target
ax = axes[1, 1]
ax.plot(t, true_u, 'g--', lw=1.5, label='True u(t) (gene expression)')
ax.plot(t, u_est, 'b-', lw=1.5, alpha=0.6, label='Filtered u(t)')
ax.set_xlabel('Time')
ax.set_ylabel('Synthesis rate u')
ax.set_title('Reconstructed gene expression rate (the target)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
