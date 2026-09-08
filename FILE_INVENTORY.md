# File Inventory

Overview of what each file in this project does, and the key functions in
each. The app is a Streamlit UI split across `streamlit_app.py` (thin entry
point) and the `app/` package (one module per tab), which orchestrate
several plain-Python model/fitting/plotting modules at the top level; those
modules have no Streamlit dependency and can be imported/tested
independently.

## streamlit_app.py

Thin Streamlit entry point. Sets page config/title, imports one `render_*`
function per tab from `app/`, creates the tab strip via `st.tabs(...)`, and
calls each tab's render function inside its `with <tab>:` block. The
Simulation tab's return value (`sim_state`) is passed into the Data tab
since it needs the Simulation tab's current settings as defaults; every
other tab communicates purely via `st.session_state`. Holds no model logic
of its own.

**Tabs, in registration order:**

1. **Simulation** (`app.tab_simulation`) — run the full maturation model,
   set live params.
2. **Data** (`app.tab_data`) — upload/generate the fluorescence trace to
   fit.
3. **Bleaching Only Simulation** (`app.tab_bleaching`) — pure
   photobleaching-decay fit, plus a known-bleaching-pole fit of the full
   model.
4. **Bleaching Fit History** (`app.tab_bleaching_history`) — saved runs
   from tab 3.
5. **Variable Bleaching** (`app.tab_variable_bleaching`) — overlay F(t)
   across several `kb` values, joint least-squares fit with shared `km`.
6. **Variable Bleaching History** (`app.tab_variable_bleaching_history`) —
   saved joint fit runs from tab 5.
7. **Least Squares Fitting** (`app.tab_fitting`) — single and multi-start
   fits of the full model.
8. **Profile Likelihood** (`app.tab_profile_likelihood`) — parameter
   identifiability sweeps (1D and 2D).
9. **Bode Plot** (`app.tab_bode`) — frequency response for user-entered
   parameter sets.
10. **Multi-Start History** (`app.tab_multistart_history`) — saved
    multi-start fit runs from tab 7.
11. **Profile Likelihood History** (`app.tab_profile_history`) — saved
    profile likelihood runs from tab 8.

## app/ (Streamlit tab modules)

### app/shared.py

Render helpers and LaTeX formula constants shared by more than one tab
module (a live-run tab and its matching history tab). No `render_*_tab` of
its own.

- `compute_dataset_key(data_label, data_df)` — stable content-hash identity
  for a loaded dataset (used to group Profile Likelihood History by
  dataset). Duplicated verbatim in `app/tab_data.py`.
- `render_bode_result(br, title)` — renders a magnitude/phase Bode plot +
  cutoff metrics for one saved Bode result; currently unused since Bode
  plots were removed from the Bleaching tab, kept for compatibility.
- `render_multi_start_results(results_df, param_names, derived_names, true_values, include_nonconverged)`
  — renders histograms, summary stats table, and raw results table for a
  multi-start fit result. Shared by the Fitting, Bleaching, and their
  history tabs.
- `render_profile_likelihood_result(profile_df, profile_target, true_value)`
  — renders the SSE-vs-value plot + raw data table for a 1D profile
  likelihood run.
- `render_profile_2d_result(profile_df)` — renders the 2D (a, b) SSE
  contour plot + raw data table.
- Constants `FORMULA_1STEP`, `FORMULA_2STEP`, `FORMULA_BLEACH` — LaTeX
  transfer-function strings shown alongside Bode plots.
- `_multi_start_part(result)` / `_save_bleach_tab_history_entry()` —
  leftover/unused duplicates of logic now defined directly in
  `app/tab_bleaching.py`.

### app/tab_simulation.py — Simulation tab

Runs the full 1-step/2-step maturation ODE model with live sidebar
parameters and plots I/M/B/F vs. time. Returns a `sim_state` dict used as
defaults by the Data and Bode Plot tabs.

- `render_simulation_tab()` — renders model choice, initial conditions,
  rate constants, and derived-parameter metrics (a, b, G, G·I0); on
  **Run Simulation** integrates the model via `solve_ivp` and plots
  I/X/M/B/F vs time; returns `sim_state`. Writes
  `st.session_state["sim_kb"/"sim_kd"/"sim_alpha"/"sim_run_version"/"sim_result"]`.

### app/tab_data.py — Data tab

Supplies the fluorescence trace to fit against: either an uploaded
experimental CSV or synthetic data generated from the Simulation tab's
current rate constants plus Gaussian noise.

- `compute_dataset_key(data_label, data_df)` — same hash-based dataset
  identity function as in `shared.py` (duplicated).
- `render_data_tab(sim_state)` — renders the data-source radio
  (upload CSV / generate synthetic), CSV upload + validation, or
  noise-parameter inputs and **Generate Synthetic Data**; plots and tables
  the resulting trace. Reads `sim_state`. Writes
  `st.session_state["synthetic_data_df"/"synthetic_params"/"noise_params"/"current_data"/"current_data_label"/"current_data_source"/"current_dataset_key"/"dataset_info"]`.

### app/tab_bleaching.py — Bleaching Only Simulation tab

Two-part tab: (1) simulate/fit the pure photobleaching-decay model to
estimate `b = kb + kd` directly, then (2) reuse that `b` as a fixed value
in a "known bleaching pole" fit of the full maturation model against the
Data tab's trace. Both fits are saved together to history when the known-b
fit's multi-start button is clicked.

- `render_bleaching_tab()` — renders initial conditions/rate constants for
  the bleach-only model, **Run Simulation** (plots M/B/F),
  **Generate Synthetic Data** (noisy pure-decay trace),
  **Run Multi-Start Fit** for b/A (bleach-only), then a "known bleaching
  pole" section fitting the full model with b fixed, whose
  **Run Multi-Start Fit** also saves history. Reads
  `st.session_state["sim_run_version"/"sim_kb"/"sim_kd"/"sim_alpha"/"current_data"/"current_data_source"/"synthetic_params"]`;
  writes
  `st.session_state["bleach_sim"/"bleach_synthetic_data"/"bleach_synthetic_params"/"bleach_noise_params"/"bleach_multi_result"/"known_b_multi_result"]`
  plus many `bleach_*`/`known_b_*` widget keys.
- `_multi_start_part(result)` — converts a stored multi-start result dict
  into the bleach-history "part" schema.
- `_save_bleach_tab_history_entry()` — bundles the current bleach-only and
  known-b fit results into one history entry and appends it via
  `history_store.append_bleach_entry`.

### app/tab_bleaching_history.py — Bleaching Fit History tab

Displays saved runs from the Bleaching Only Simulation tab, loaded from
`bleach_fit_history.json`, most recent first.

- `render_bleaching_history_tab()` — lists each saved entry with a summary
  line, **Display**/**Delete**/**Clear history** buttons; on Display,
  renders the bleach-only and known-b fit results via
  `render_multi_start_results`. Writes
  `st.session_state["bleach_history_displayed"]`.

### app/tab_variable_bleaching.py — Variable Bleaching tab

Fixes `kd` at 0 and lets the user add multiple parameter sets (each its own
`kb` and initial conditions) sharing one `km`/`k1,k2`/`u`/`alpha`, overlays
their F(t) curves, generates synthetic noisy data per trace, and jointly
fits all traces (shared `km`) via multi-start least squares, saving results
to history.

- `render_variable_bleaching_tab()` — renders shared rate constants,
  per-set `kb`/I0/X0/M0/B0 inputs and **Run (add to graph)**, plots
  overlaid F(t) curves, **Generate Synthetic Data** (per-trace noise), and
  **Run Multi-Start Fit (N runs)** jointly fitting all traces (via
  `variable_bleaching_fit.residuals_*_shared_*`), rendering
  histograms/summary tables per trace and appending to history via
  `history_store.append_variable_bleaching_entry`. Writes
  `st.session_state["vb_kb_sets"/"vb_synthetic_results"/"vb_synthetic_meta"/"vb_fit_result"]`
  plus many `vb_*` widget keys.

### app/tab_variable_bleaching_history.py — Variable Bleaching History tab

Displays saved joint-fit runs from the Variable Bleaching tab, loaded from
`variable_bleaching_history.json`, most recent first. Distinguishes newer
multi-start-schema entries from older legacy single-run entries.

- `render_variable_bleaching_history_tab()` — lists each saved entry with a
  summary line, **Display**/**Delete**/**Clear history** buttons;
  dispatches to the multi-start or legacy renderer plus a noise-parameter
  caption. Writes `st.session_state["vb_history_displayed"]`.
- `_render_multi_start_entry(entry, index)` — renders
  histograms/summary tables/raw table for a multi-start-schema entry.
- `_render_legacy_single_run_entry(entry)` — renders metrics for an older
  single-run-schema entry (pre multi-start).

### app/tab_fitting.py — Least Squares Fitting tab

Fits I0/`km` (or k1, k2)/`kb`/`kd`/alpha to whatever trace is loaded in the
Data tab over a user-selected time region, either as a single
`least_squares` call or a multi-start sweep.

- `render_fitting_tab()` — renders time-region slider, baseline input,
  model choice, initial-guess expander; on **Fit Parameters** runs a
  single least-squares fit and plots raw/corrected/fitted curves plus
  goodness-of-fit metrics and a synthetic-truth comparison table; on
  **Run Multi-Start Fit (N runs)** runs `multi_start_fit.run_multi_start`
  and renders results via `render_multi_start_results`, appending to
  history via `history_store.append_multi_start_entry`. Reads
  `st.session_state["current_data"/"current_data_label"/"current_data_source"/"synthetic_params"]`;
  writes
  `st.session_state["current_fit_data"/"fit_result"/"fit_single_result"/"fit_multi_result"]`.

### app/tab_profile_likelihood.py — Profile Likelihood tab

Sweeps one raw parameter/derived quantity (1D) or a joint (a, b) grid (2D)
across fixed values, re-optimizing everything else at each point, to
assess parameter identifiability, using the fit setup from the Least
Squares Fitting tab.

- `render_profile_likelihood_tab()` — renders the parameter/quantity
  selector and grid-range inputs; on **Run Profile Likelihood** calls
  `profile_likelihood.profile_raw_parameter`/`profile_derived_quantity`
  and renders via `render_profile_likelihood_result`; on
  **Run 2D Profile Likelihood (a, b)** calls
  `profile_likelihood_2D.profile_ab_2d` and renders via
  `render_profile_2d_result`; both save to history via
  `history_store.append_profile_entry`. Reads
  `st.session_state["current_fit_data"/"synthetic_params"/"dataset_info"]`;
  writes `st.session_state["profile_1d_result"/"profile_2d_result"]`.

### app/tab_bode.py — Bode Plot tab

Lets the user directly enter one or more 1-step/2-step parameter sets and
overlay their magnitude/phase Bode curves and -3dB cutoffs on one graph.

- `render_bode_tab()` — renders a parameter-set input form and
  **Run (add to graph)**, a manageable list of added sets, frequency-range
  inputs, then computes/plots magnitude+phase via
  `bode_plot.bode_1step`/`bode_2step` and a cutoff-frequency summary
  table. Writes `st.session_state["bode_param_sets"]`.

### app/tab_multistart_history.py — Multi-Start History tab

Displays saved multi-start fit runs from the Least Squares Fitting tab,
loaded from `multi_start_history.json`, most recent first, filterable by
1-step/2-step model.

- `render_multistart_history_tab()` — lists entries (filtered by model
  radio) with summary line, **Display**/**Delete**/**Clear history**
  buttons; on Display, renders via `render_multi_start_results`. Writes
  `st.session_state["ms_history_displayed"]`.

### app/tab_profile_history.py — Profile Likelihood History tab

Displays saved profile likelihood runs from `profile_likelihood_history.json`,
grouped by dataset then by run, filterable by 1-step/2-step model.

- `render_profile_history_tab()` — groups entries by `dataset_key` then
  `run_key`; **Display**/**Delete**/**Clear history** buttons dispatch to
  `render_profile_likelihood_result` (1D) or `render_profile_2d_result`
  (2D). Writes `st.session_state["pl_history_displayed"]`.
- `_render_pl_dataset_group(group, run_items)` — renders one dataset's
  ground-truth/noise header and its runs.

## Top-level model/fitting modules

### Maturation_Models.py

Core ODE definitions for the full 1-step (I → M → B) and 2-step
(I → X → M → B) fluorescent-protein maturation + photobleaching models,
plus their `simulate_*`/`residuals_*` helpers used by the Least Squares
Fitting tab. No Streamlit dependency.

- `model_1step(t, y, params)` — right-hand side of the 1-step model.
  `params`: `u, km, kb, kd`.
- `model_2step(t, y, params)` — right-hand side of the 2-step model.
  `params`: `u, k1, k2, kb, kd`.
- `simulate_1step(t, params, I0, M0, B0)` — integrates `model_1step` with
  `solve_ivp` and returns `(t, I, M, B, F)`, where `F = alpha * M`.
- `simulate_2step(t, params, I0, X0, M0, B0)` — same for the 2-step model,
  returns `(t, I, X, M, B, F)`.
- `residuals_1step(x, t, F_meas, fixed)` — unpacks
  `x = [I0, km, kb, kd, alpha]`, simulates, and returns `F - F_meas` for
  `scipy.optimize.least_squares`.
- `residuals_2step(x, t, F_meas, fixed)` — same for
  `x = [I0, k1, k2, kb, kd, alpha]`.

### Bleaching_Only_Model.py

Pure photobleaching-decay model (no maturation, I(t) ≈ 0): closed-form
`M(t) = M0 * exp(-b*t)`, `b = kb + kd`. Since M0 and alpha aren't
individually identifiable from F(t) alone, the fit estimates `b` and
`A = alpha * M0` directly (same convention as the Variable Bleaching tab's
`K = G * I0`). Used by the Bleaching Only Simulation tab.

- `model_bleach(t, y, params)` — right-hand side (dM/dt, dB/dt) for pure
  decay.
- `simulate_bleach(t, params, M0, B0)` — integrates the model, returns
  `(t, M, B, F)`.
- `analytical_M(t, M0, kb, kd)` — closed-form `M(t) = M0 * exp(-b*t)`.
- `analytical_F(t, A, kb, kd)` — closed-form `F(t) = A * exp(-b*t)`.
- `residuals_bleach(x, t, F_meas)` — least-squares residuals fitting
  `x = [b, A]`; `kd` fixed at 0.

### Maturation_Model_Known_Bleaching_Pole.py

Full maturation model variant where `b = kb + kd` is treated as
known/fixed (e.g. measured via `Bleaching_Only_Model.py`), removing one
free parameter. Since I0 and alpha aren't individually identifiable, fits
`km` (or `k1`/`k2`) and `K = G * I0` directly. Used by the "known
bleaching pole fit" section of the Bleaching Only Simulation tab.

- `model_1step_known_b(t, y, params)` — right-hand side (dI/dt, dM/dt)
  with `b` fixed.
- `model_2step_known_b(t, y, params)` — right-hand side (dI/dt, dX/dt,
  dM/dt) with `b` fixed.
- `simulate_1step_known_b(t, params, I0, M0)` — integrates, returns
  `(t, I, M, F)`.
- `simulate_2step_known_b(t, params, I0, X0, M0)` — integrates, returns
  `(t, I, X, M, F)`.
- `residuals_1step_known_b(x, t, F_meas, fixed)` — residuals fitting
  `x = [km, K]`; `b` and `kd` fixed via `fixed`.
- `residuals_2step_known_b(x, t, F_meas, fixed)` — residuals fitting
  `x = [k1, k2, K]`; `b` and `kd` fixed via `fixed`.

### variable_bleaching_fit.py

Joint least-squares fitting across multiple fluorescence traces that share
one maturation rate (`km`, or `k1`/`k2`) but each have their own `kb` and
amplitude (`K = G * I0`). Used by the Variable Bleaching tab's "fit across
traces" section.

- `residuals_1step_shared_km(x, t, F_meas_list, fixed)` — joint residuals
  across traces; `x = [km, kb_1, K_1, kb_2, K_2, ...]`.
- `residuals_2step_shared_k(x, t, F_meas_list, fixed)` — joint residuals
  across traces; `x = [k1, k2, kb_1, K_1, ...]`.

### bode_plot.py

Frequency-response (Bode) analysis, derived from each model's linear
transfer function (built with `scipy.signal`).

- `transfer_function_1step(alpha, km, kb, kd)` / `transfer_function_2step(alpha, k1, k2, kb, kd)`
  — build a `scipy.signal.TransferFunction` for each model.
- `transfer_function_bleach(alpha, M0, kb, kd)` — Laplace transform of the
  bleach-only model's impulse response.
- `bode_1step(alpha, km, kb, kd, w)` / `bode_2step(alpha, k1, k2, kb, kd, w)`
  — return `(w, mag, phase)` via `scipy.signal.bode`.
- `bode_bleach(alpha, M0, kb, kd, w)` — magnitude/phase for the bleach-only
  transfer function.
- `analytical_cutoff_1step(km, kb, kd)` — exact -3 dB cutoff frequency,
  solved in closed form from the model's two real poles.
- `analytical_cutoff_2step(k1, k2, kb, kd)` — exact -3 dB cutoff, solved as
  the one positive real root of a cubic in `wc^2`.
- `analytical_cutoff_bleach(kb, kd)` — exact -3 dB cutoff for the
  bleach-only model (= `kb + kd`).
- `numerical_cutoff(w, mag)` — estimates the -3 dB cutoff by interpolating
  directly off a computed magnitude curve (independent of the analytical
  formulas — used as a cross-check, and reflects whatever frequency range
  was actually plotted).

### gaussian_noise.py

Generates synthetic "experimental" data by perturbing rate constants with
Gaussian noise and Euler-integrating the model forward (rather than adding
noise to the trace directly), plus a separate helper for direct
measurement noise.

- `_noisy_rate(nominal, std, rng)` — draws one Gaussian-perturbed rate
  constant (clipped at zero); returns `nominal` unchanged if `std <= 0`.
- `add_measurement_noise(F, std, seed)` — adds independent Gaussian
  measurement/readout noise directly to an intensity trace (e.g.
  camera/shot noise), on top of whatever rate-constant noise was already
  applied.
- `simulate_1step_noisy(t, params, I0, M0, B0, km_std, kb_std, seed)` —
  Euler-integrates the 1-step model, redrawing `km` and `kb` from
  `N(rate, std)` at every time step. Returns `(t, I, M, B, F)`.
- `simulate_2step_noisy(t, params, I0, X0, M0, B0, k1_std, k2_std, kb_std, seed)`
  — same for the 2-step model, perturbing `k1`, `k2`, `kb`.
- `simulate_bleach_noisy(t, params, M0, B0, kb_std, seed)` — Euler-
  integrates the bleach-only model, perturbing `kb`.

### multi_start_fit.py

Runs the same least-squares fit many times from independently randomized
initial guesses, to check convergence robustness and parameter
identifiability (e.g. whether `km` and `kd` trade off against each other
while their sum stays well-constrained). Used throughout the app's fitting
tabs.

- `sample_log_uniform(center, rng, decade_span)` — draws one positive value,
  log-uniform over `decade_span` decades centered on `center`.
- `sample_initial_guess(centers, rng, decade_span)` — draws a full parameter
  vector, each entry independently log-uniform around its center.
- `run_multi_start(residual_fn, param_names, centers, bounds, args, n_runs, seed, decade_span, max_nfev)`
  — the main entry point. Runs `scipy.optimize.least_squares` `n_runs`
  times (same `residual_fn`/`bounds`/`args` the single-fit button uses,
  just from a fresh random `x0` each time), and returns a DataFrame with
  one row per run: each fitted parameter, `cost`, `converged`, `message`,
  `nfev`, `run`. Non-converged runs are kept and flagged, not dropped.

### multi_start_plots.py

Plotting helper for multi-start fit results — takes a results DataFrame
(with derived quantities already added as columns) and returns a
matplotlib `Figure`; the caller (`app/shared.py`) displays it via
`st.pyplot`.

- `plot_histograms(df, names, true_values, color, figsize_per_panel)` —
  one histogram subplot per name in `names`; draws a red dashed vertical
  line at `true_values[name]` when known. Used for both the raw-parameter
  and derived-quantity histogram panels.

### profile_likelihood.py

1D profile likelihood analysis: sweeps one raw parameter or derived
quantity across a grid, re-optimizing everything else at each point via
`least_squares`, recording SSE. Used by the Profile Likelihood tab.

- `compute_true_values(synthetic_params, fit_is_two_step)` — ground-truth
  raw params + derived quantities for synthetic data (returns `{}` if
  unavailable/mismatched model).
- `profile_raw_parameter(residual_fn, param_names, fix_name, grid_values, centers, bounds, args, seed, decade_span, max_nfev)`
  — profiles one raw parameter, fixing it at each grid value.
- `profile_derived_quantity(residual_fn, param_names, quantity_name, fit_is_two_step, grid_values, centers, bounds, args, seed, decade_span, max_nfev)`
  — profiles a derived quantity (e.g. `a = km + kd`) via reparametrization
  (one raw param computed from the target + others).
- Module-level dicts `DERIVED_QUANTITY_FORMULAS` and
  `DERIVED_QUANTITY_REPARAM` — the derived-quantity formulas and
  reparametrization specs, keyed by 1-step/2-step.

### profile_likelihood_2D.py

2D joint profile likelihood over the (a, b) decay-rate pair
(`a = km + kd` or `k1 + kd`; `b = kb + kd`), used to visualize the a/b
exchange degeneracy. Used by the Profile Likelihood tab's "2D Profile
Likelihood" section.

- `profile_ab_2d(residual_fn, param_names, fit_is_two_step, a_values, b_values, centers, bounds, args, seed, decade_span, max_nfev)`
  — sweeps `a` and `b` jointly over a grid, re-optimizing remaining free
  params (`kd` absorbs the constraint); returns a DataFrame of `a`, `b`,
  `sse`, `converged`, fitted params.
- `plot_profile_2d(profile_df)` — contour plot of SSE over the (a, b)
  grid.

### history_store.py

Disk persistence layer (JSON) for four kinds of run history, so past runs
survive process restarts (unlike `st.session_state`). Each kind has
`load_*`, `append_*`, `clear_*`, `delete_*` functions plus private
record ↔ entry converters (DataFrame ⇄ JSON-safe dict).

- `_load_records(path)` / `_save_records(path, records)` — internal JSON
  read/write helpers.
- **Multi-start history** (`multi_start_history.json`):
  `load_multi_start_history()`, `append_multi_start_entry(entry)`,
  `clear_multi_start_history()`, `delete_multi_start_entry(index)`.
- **Profile likelihood history** (`profile_likelihood_history.json`):
  `load_profile_history()`, `append_profile_entry(entry)`,
  `clear_profile_history()`, `delete_profile_entry(index)`.
- **Bleaching tab history** (`bleach_fit_history.json`):
  `load_bleach_history()`, `append_bleach_entry(entry)`,
  `clear_bleach_history()`, `delete_bleach_entry(index)` — bundles the
  bleach-only fit and known-b fit together per entry.
- **Variable Bleaching history** (`variable_bleaching_history.json`):
  `load_variable_bleaching_history()`,
  `append_variable_bleaching_entry(entry)`,
  `clear_variable_bleaching_history()`,
  `delete_variable_bleaching_entry(index)`.

## Data files (not code)

Written/read exclusively via `history_store.py`; each stores persisted
fit-run history as a JSON array of records (most recent last — the UI
reverses this for "most recent first" display) so runs survive process
restarts.

- **bleach_fit_history.json** — one record per save from the Bleaching
  Only Simulation tab, bundling the bleach-only fit (`bleach_only`) and
  known-bleaching-pole fit (`known_b`) results (each with
  `results_records`, a serialized multi-start results DataFrame), plus
  legacy `bleach_bode`/`known_b_bode` keys from before Bode plots were
  removed from that tab.
- **multi_start_history.json** — one record per multi-start fit run from
  the Least Squares Fitting tab: model type, data source, param/derived
  names, true values, and the results DataFrame as `results_records`.
- **profile_likelihood_history.json** — one record per profile likelihood
  run (1D or 2D) from the Profile Likelihood tab: dataset identity/label,
  synthetic ground truth, noise params, fit region, profiled target, and
  the profile DataFrame as `profile_records`.
- **variable_bleaching_history.json** — one record per joint multi-start
  fit run from the Variable Bleaching tab: shared/trace param names, true
  values, noise params, fit seed, and results as `results_records` (older
  entries use a different legacy single-run schema with
  `shared_true`/`shared_fitted`/`per_trace_rows`).

## requirements.txt

Python dependencies: `numpy`, `matplotlib`, `scipy`, `streamlit`, `pandas`.
