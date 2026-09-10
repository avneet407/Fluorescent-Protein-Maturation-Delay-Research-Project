"""
Kalman Filter: Rocket Altitude Estimation (Example 10, kalmanfilter.net style)
-- matches the book's exact iteration structure, including gravity compensation --
"""

import numpy as np
import matplotlib.pyplot as plt

dt = 0.25
sigma_a = 0.1
sigma_z = 20.0
g = 9.8            # gravity -- subtracted from every accelerometer reading

F = np.array([[1, dt],
              [0, 1]])
G = np.array([[0.5 * dt**2],
              [dt]])
H = np.array([[1, 0]])
Q = np.array([[dt**4/4, dt**3/2],
              [dt**3/2, dt**2]]) * sigma_a**2
R = np.array([[sigma_z**2]])

h_n = np.array([
    6.43, 1.30, 39.43, 45.89, 41.44, 48.70, 78.06, 80.08,
    61.77, 75.15, 110.39, 127.83, 158.75, 156.55, 213.32, 229.82, 262.80,
    297.57, 335.69, 367.92, 377.19, 411.18, 460.70, 468.39, 553.90, 583.97,
    655.15, 723.09, 736.85, 787.22
])
a_n = np.array([
    39.81, 39.67, 39.81, 39.84, 40.05, 39.85, 39.78, 39.65,
    39.67, 39.78, 39.59, 39.87, 39.85, 39.59, 39.84, 39.90, 39.63,
    39.59, 39.76, 39.79, 39.73, 39.93, 39.83, 39.85, 39.94, 39.86,
    39.76, 39.86, 39.74, 39.94
])
n_steps = len(h_n)

# Iteration Zero: initialize, then predict with u0 = 0 (no accel reading yet)
x = np.array([[0.0], [0.0]])
P = np.array([[500.0, 0.0], [0.0, 500.0]])
x = F @ x + G * 0.0
P = F @ P @ F.T + Q

altitude_est = np.zeros(n_steps)
velocity_est = np.zeros(n_steps)

for n in range(n_steps):
    z_n = np.array([[h_n[n]]])

    # Step 2 - Update
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x = x + K @ (z_n - H @ x)
    I = np.eye(2)
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T

    altitude_est[n] = x[0, 0]
    velocity_est[n] = x[1, 0]

    # Step 3 - Predict 
    if n < n_steps - 1:
        u_n = a_n[n] - g          
        x = F @ x + G * u_n
        P = F @ P @ F.T + Q

print(f"{'n':>3} {'measured h':>12} {'est. altitude':>15} {'est. velocity':>15}")
for n in range(n_steps):
    print(f"{n+1:3d} {h_n[n]:12.2f} {altitude_est[n]:15.2f} {velocity_est[n]:15.2f}")

# Plot
t = np.arange(1, n_steps + 1) * dt
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(t, h_n, c='gray', s=30, alpha=0.6, label='Noisy altimeter measurements')
ax.plot(t, altitude_est, 'b-', lw=2, label='Kalman filter altitude estimate')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Altitude (m)')
ax.set_title('Rocket altitude: measured vs. estimated')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(t, velocity_est, 'r-', lw=2, label='Kalman filter velocity estimate')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('Rocket velocity estimate (never directly measured)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()