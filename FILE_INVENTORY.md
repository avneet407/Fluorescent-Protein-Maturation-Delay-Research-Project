# File Inventory

Overview of what each file in this project does, and the key functions in
each. The app is a Streamlit UI (`streamlit_app.py`) that orchestrates
several plain-Python modules; the modules have no Streamlit dependency and
can be imported/tested independently.

## streamlit_app.py

The Streamlit UI and orchestration layer. Defines three tabs:

- **Simulation** — runs the ODE model with sidebar parameters and plots
  I/M/B/F vs. time (`solve_ivp` called directly here).
- **Least Squares Fitting** — lets the user either upload an experimental
  CSV or generate synthetic noisy data, select a fitting region, run a
  single least-squares fit, or run a multi-start fit (many fits from random
  initial guesses) and view diagnostic plots.
- **Bode Plot** — frequency response of the currently selected model,
  overlaid with the synthetic-data ground truth and the most recent fit
  result if available.

Holds no reusable functions of its own — all model/fitting/plotting logic
is imported from the other files below. Key pieces of state kept in
`st.session_state` across reruns:

- `synthetic_data_df` — the generated synthetic Slice/Mean/Time DataFrame.
- `synthetic_params` — the ground-truth rate constants used to generate it
  (`is_two_step`, `I0`, `km`/`k1`/`k2`, `kb`, `kd`, `alpha`).
- `fit_result` — the parameters from the most recent least-squares fit
  (single-fit button), plus which model was used and whether the data was
  synthetic or experimental.

## Maturation_Models.py

The ODE definitions for the two maturation models, plus simulation and
residual helpers used by both the Simulation tab and the fitting code.

- `model_1step(t, y, params)` — right-hand side of the 1-step model
  (I → M → B), for `solve_ivp`. `params`: `u, km, kb, kd`.
- `model_2step(t, y, params)` — right-hand side of the 2-step model
  (I → X → M → B), for `solve_ivp`. `params`: `u, k1, k2, kb, kd`.
- `simulate_1step(t, params, I0, M0, B0)` — integrates `model_1step` with
  `solve_ivp` and returns `(t, I, M, B, F)`, where `F = alpha * M`.
- `simulate_2step(t, params, I0, X0, M0, B0)` — same for the 2-step model,
  returns `(t, I, X, M, B, F)`.
- `residuals_1step(x, t, F_meas, fixed)` — unpacks
  `x = [I0, km, kb, kd, alpha]`, simulates, and returns `F - F_meas` for
  `scipy.optimize.least_squares`.
- `residuals_2step(x, t, F_meas, fixed)` — same for
  `x = [I0, k1, k2, kb, kd, alpha]`.

## bode_plot.py

Frequency-response (Bode) analysis, derived from each model's linear
transfer function (built with `scipy.signal`).

- `transfer_function_1step(alpha, km, kb, kd)` / `transfer_function_2step(alpha, k1, k2, kb, kd)`
  — build a `scipy.signal.TransferFunction` for each model.
- `bode_1step(alpha, km, kb, kd, w)` / `bode_2step(alpha, k1, k2, kb, kd, w)`
  — return `(w, mag, phase)` via `scipy.signal.bode`.
- `analytical_cutoff_1step(km, kb, kd)` — exact -3 dB cutoff frequency,
  solved in closed form from the model's two real poles.
- `analytical_cutoff_2step(k1, k2, kb, kd)` — exact -3 dB cutoff, solved as
  the one positive real root of a cubic in `wc^2`.
- `numerical_cutoff(w, mag)` — estimates the -3 dB cutoff by interpolating
  directly off a computed magnitude curve (independent of the analytical
  formulas — used as a cross-check, and reflects whatever frequency range
  was actually plotted).

## gaussian_noise.py

Generates synthetic "experimental" data by perturbing rate constants with
Gaussian noise and Euler-integrating the model forward (rather than adding
noise to the trace directly).

- `_noisy_rate(nominal, std, rng)` — draws one Gaussian-perturbed rate
  constant (clipped at zero); returns `nominal` unchanged if `std <= 0`.
- `simulate_1step_noisy(t, params, I0, M0, B0, km_std, kb_std, seed)` —
  Euler-integrates the 1-step model, redrawing `km` and `kb` from
  `N(rate, std)` at every time step. Returns `(t, I, M, B, F)`.
- `simulate_2step_noisy(t, params, I0, X0, M0, B0, k1_std, k2_std, kb_std, seed)`
  — same for the 2-step model, perturbing `k1`, `k2`, `kb`.
- `add_measurement_noise(F, std, seed)` — adds independent Gaussian
  measurement/readout noise directly to an intensity trace (e.g. camera/shot
  noise), on top of whatever rate-constant noise was already applied.

## multi_start_fit.py

Runs the same least-squares fit many times from independently randomized
initial guesses, to check convergence robustness and parameter
identifiability (e.g. whether `km` and `kd` trade off against each other
while their sum stays well-constrained).

- `sample_log_uniform(center, rng, decade_span)` — draws one positive value,
  log-uniform over `decade_span` decades centered on `center`.
- `sample_initial_guess(centers, rng, decade_span)` — draws a full parameter
  vector, each entry independently log-uniform around its center.
- `run_multi_start(residual_fn, param_names, centers, bounds, args, n_runs, seed, decade_span)`
  — the main entry point. Runs `scipy.optimize.least_squares` `n_runs`
  times (same `residual_fn`/`bounds`/`args` the single-fit button uses,
  just from a fresh random `x0` each time), and returns a DataFrame with
  one row per run: each fitted parameter, `cost`, `converged`, `message`,
  `nfev`, `run`. Non-converged runs are kept and flagged, not dropped.

## multi_start_plots.py

Plotting helpers for multi-start fit results — each takes a results
DataFrame (with derived quantities already added as columns) and returns a
matplotlib `Figure`; the caller (`streamlit_app.py`) displays it via
`st.pyplot`.

- `plot_histograms(df, names, true_values, color, figsize_per_panel)` —
  one histogram subplot per name in `names`; draws a red dashed vertical
  line at `true_values[name]` when known. Used for both the raw-parameter
  and derived-quantity histogram panels.
- `plot_scatter_1step(df, true_values)` — `km` vs `kd` scatter, with the
  constraint line `km + kd = a` overlaid (using the true `a` if known, else
  the median fitted `a`).
- `plot_scatter_2step(df, true_values)` — `k1` vs `kd` and `k2` vs `kd`
  scatter plots, with their respective constraint lines
  (`k1 + kd = a`, `k2 + kd = c`) overlaid.

## requirements.txt

Python dependencies: `numpy`, `matplotlib`, `scipy`, `streamlit`, `pandas`.

