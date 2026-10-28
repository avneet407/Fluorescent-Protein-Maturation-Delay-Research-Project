"""Render helpers and constants shared by more than one tab module.

Each function here is used by a live-run tab (right after a fit/plot) and
its matching history tab (replaying a saved run), so they're factored out
here rather than duplicated.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import streamlit as st
from app.plot_style import show_figure

from Parameter_Identification.multi_start_plots import plot_histograms
from Parameter_Identification.profile_likelihood_2D import plot_profile_2d
from Parameter_Identification.plot_labels import math_label
from app.tab_simulation import SPECIES_COLORS

# Transfer-function formulas (production input u -> fluorescence F) shown
# alongside every Bode plot in the UI, matching bode_plot.py's
# transfer_function_1step / transfer_function_2step / transfer_function_bleach.
FORMULA_1STEP = r"H(s) = \dfrac{\alpha\, k_m}{(s + k_m + k_d)(s + k_b + k_d)}"
FORMULA_2STEP = (
    r"H(s) = \dfrac{\alpha\, k_1 k_2}{(s + k_1 + k_d)(s + k_2 + k_d)(s + k_b + k_d)}"
)
FORMULA_BLEACH = r"H(s) = \dfrac{\alpha\, M_0}{s + k_b + k_d} \quad\text{(Laplace transform of } F(t)=\alpha M_0 e^{-(k_b+k_d)t}\text{)}"


def compute_dataset_key(data_label, data_df):
    """Stable content-based identity for a loaded dataset.

    Used to group Profile Likelihood History entries by the underlying data
    they were run against: regenerating/re-uploading the same trace yields
    the same key, while any change to the actual Mean values yields a new one.
    """
    payload = data_label + "|" + ",".join(f"{v:.10g}" for v in data_df["Mean"].to_numpy(dtype=float))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _multi_start_part(result):
    """Build a bleach-history fit-result part from a stored multi-start result dict, or None.

    Shared by the bleach-only and known-b saves in the Bleaching Only tab so
    each save bundles both fits' latest results into one history entry.
    """
    if result is None:
        return None
    results_df = result["results_df"]
    return {
        "n_total": len(results_df),
        "n_converged": int(results_df["converged"].sum()),
        "param_names": result["param_names"],
        "derived_names": result["derived_names"],
        "true_values": result["true_values"],
        "extra_info": result.get("extra_info", {}),
        "results_df": results_df,
    }


def _save_bleach_tab_history_entry():
    """Save the current Bleaching Only tab state as one combined history entry.

    Called whenever either "Plot Bode Response" button in that tab (the
    bleach-only fit's or the known bleaching pole fit's) is clicked, bundling
    everything currently run in the tab this session: both fits and both
    Bode plots. Any piece not yet run is saved as None.
    """
    append_bleach_entry({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bleach_only": _multi_start_part(st.session_state.get("bleach_multi_result")),
        "bleach_bode": st.session_state.get("bleach_bode_result"),
        "known_b": _multi_start_part(st.session_state.get("known_b_multi_result")),
        "known_b_bode": st.session_state.get("known_b_bode_result"),
    })


def render_bode_result(br, title):
    """Render the magnitude/phase Bode plot + cutoff metrics for one Bleaching Only tab Bode result.

    Shared by the live view (right after clicking Plot Bode Response for
    either the bleach-only fit or the known bleaching pole fit) and the
    Bleaching tab history (replaying a saved run).
    """
    if br.get("formula_latex"):
        st.latex(br["formula_latex"])

    fig_bode, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6))

    ax_mag.semilogx(br["w"], br["mag_syn"], color="tab:green", linestyle="--", label="Synthetic input")
    ax_mag.semilogx(br["w"], br["mag_fit"], color="tab:red", linestyle=":", label=br["fit_label"])
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_xlabel("Frequency (rad/min)")
    ax_mag.set_title(title)
    ax_mag.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_mag.legend()

    ax_phase.semilogx(br["w"], br["phase_syn"], color="tab:green", linestyle="--", label="Synthetic input")
    ax_phase.semilogx(br["w"], br["phase_fit"], color="tab:red", linestyle=":", label=br["fit_label"])
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.set_xlabel("Frequency (rad/min)")
    ax_phase.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_phase.legend()

    fig_bode.tight_layout()
    show_figure(fig_bode)

    cutoff_col1, cutoff_col2 = st.columns(2)
    if br["cutoff_syn"] is not None:
        cutoff_col1.metric("Synthetic input -3 dB cutoff (rad/min)", f"{br['cutoff_syn']:.6f}")
    else:
        cutoff_col1.warning("Synthetic input cutoff not found (no positive real root).")
    if br["cutoff_fit"] is not None:
        cutoff_col2.metric("Multi-start fit -3 dB cutoff (rad/min)", f"{br['cutoff_fit']:.6f}")
    else:
        cutoff_col2.warning("Multi-start fit cutoff not found (no positive real root).")


def render_multi_start_results(results_df, param_names, derived_names, true_values, include_nonconverged):
    """Render the histograms + summary table for one multi-start fit result.

    Shared by the Least Squares Fitting tab (right after a run) and the
    History tab (replaying a stored past run). `derived_names` entries
    already present in `param_names` are skipped (some fits, e.g. the
    Bleaching Only tab's b/A fit, now fit the identifiable quantities
    directly, leaving nothing further to derive) — the "Derived quantities"
    section is omitted entirely when nothing is left after that filter.
    """
    n_converged = int(results_df["converged"].sum())
    n_total = len(results_df)
    st.markdown(f"**Multi-start fit complete: {n_converged} / {n_total} runs converged**")

    plot_df = results_df if include_nonconverged else results_df[results_df["converged"]]

    if len(plot_df) == 0:
        st.warning(
            "No runs to display (no converged runs, and non-converged "
            "runs are excluded)."
        )
        return

    st.markdown("**Raw fitted parameters across runs**")
    fig_raw = plot_histograms(plot_df, param_names, true_values, color="tab:blue")
    show_figure(fig_raw)

    extra_derived = [name for name in derived_names if name not in param_names]
    if extra_derived:
        st.markdown("**Derived quantities across runs**")
        fig_der = plot_histograms(plot_df, extra_derived, true_values, color="tab:purple")
        show_figure(fig_der)

    st.markdown("**Summary statistics across runs (mean, std, coefficient of variation)**")
    summary_rows = []
    for name in param_names + extra_derived:
        vals = plot_df[name].to_numpy(dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv = std / mean if mean != 0 else float("nan")
        summary_rows.append({
            "Parameter": name,
            "Synthetic input": true_values.get(name, float("nan")),
            "Mean": mean,
            "Std dev": std,
            "CV": cv,
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df.set_index("Parameter"))

    with st.expander("All multi-start fit results (raw table)"):
        st.markdown(
            "Each run shows two rows: its final **Fitted** values, and the "
            "**Initial guess** it started from directly below."
        )
        detail_cols = (
            ["run", "Type"] + param_names + extra_derived + ["cost", "converged", "message", "nfev"]
        )
        detail_rows = []
        for _, r in results_df.iterrows():
            detail_rows.append({
                "run": int(r["run"]),
                "Type": "Fitted",
                **{name: r[name] for name in param_names},
                **{name: f"{r[name]:.6g}" for name in extra_derived},
                "cost": f"{r['cost']:.6g}",
                "converged": str(bool(r["converged"])),
                "message": str(r["message"]),
                "nfev": str(int(r["nfev"])),
            })
            detail_rows.append({
                "run": int(r["run"]),
                "Type": "Initial guess",
                **{name: r[f"{name}_init"] for name in param_names},
                **{name: "n/a" for name in extra_derived},
                "cost": "n/a",
                "converged": "n/a",
                "message": "n/a",
                "nfev": "n/a",
            })
        # Derived columns (and cost/converged/message/nfev) are formatted as
        # strings above (rather than left as their native numeric/bool dtype)
        # so that mixing them with the "n/a" placeholder on Initial guess rows
        # doesn't create a column with inconsistent types, which
        # Streamlit/PyArrow cannot serialize.
        detail_df = pd.DataFrame(detail_rows, columns=detail_cols)
        st.dataframe(detail_df, hide_index=True)


def render_profile_likelihood_result(profile_df, profile_target, true_value=None):
    """Render the SSE-vs-value plot + raw data table for one profile likelihood run.

    Shared by the Profile Likelihood tab (right after a run) and the Profile
    Likelihood History tab (replaying a stored past run).
    """
    best_row = profile_df.loc[profile_df["sse"].idxmin()]
    st.success(
        f"Lowest SSE {best_row['sse']:.6g} at {profile_target} = {best_row['value']:.6g}"
    )

    fig_p, ax_p = plt.subplots(figsize=(8, 5))
    ax_p.plot(profile_df["value"], profile_df["sse"], "o-", color="tab:blue")
    ax_p.axvline(
        best_row["value"], color="tab:green", linestyle="--",
        label="Best-fit value (this profile)",
    )
    if true_value is not None:
        ax_p.axvline(
            true_value, color="tab:red", linestyle=":",
            label="Synthetic input (true value)",
        )
    ax_p.set_xlabel(math_label(profile_target))
    ax_p.set_ylabel("SSE (sum of squared residuals)")
    ax_p.set_title(f"Profile Likelihood: {math_label(profile_target)}")
    ax_p.legend()
    fig_p.tight_layout()
    show_figure(fig_p)

    with st.expander("Profile likelihood raw data"):
        st.dataframe(profile_df, hide_index=True)


def render_kalman_result(result):
    """Render the fluorescence/state-trajectory plots for one Kalman filter run.

    Shared by the Kalman Filter tab (right after a run) and the Kalman Filter
    History tab (replaying a saved run). Layout mirrors
    Kalman_Filter/1-step_Kalman_Filter.py and 2-step_Kalman_Filter.py: a 2x2
    grid (F, I, M, u) for the 1-step model, 2x3 (F, I, X, M, u, blank) for
    the 2-step model. "True" curves come from the Synthetic Gene Expression
    tab's ground-truth rate constants; "Filtered"/"Smoothed" curves come from
    the Kalman Filter tab's own (e.g. least-squares-calibrated) rate constants
    -- true vs. filtered/smoothed are expected to differ, that's the point of
    the comparison. Also shows a comparative RMSE table (filtered vs. smoothed
    against ground truth) below the plots. `alpha_true`/`alpha_est` fall back
    to a single legacy `alpha` key, and the RTS-smoothed curves/RMSE rows are
    skipped entirely, for history entries saved before those existed.
    """
    t = result["t"]
    alpha_true = result.get("alpha_true", result.get("alpha"))
    alpha_est = result.get("alpha_est", result.get("alpha"))
    has_smooth = "u_smooth" in result

    st.caption(
        "Sanity check -- max element-wise difference between the closed-form "
        f"F and `expm(A*dt)`: {result['max_F_diff']:.3g} (should be ~0)."
    )

    if result["is_two_step"]:
        fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    filt_kwargs = dict(color="b", lw=1.5, alpha=0.7) if has_smooth else dict(color="b", lw=2, alpha=1.0)

    ax = axes[0, 0]
    ax.scatter(t, result["z_n"], c="gray", s=15, alpha=0.5, label="Noisy fluorescence measurements")
    ax.plot(t, alpha_true * result["true_M"], "g--", lw=1.5, label=r"True fluorescence ($\alpha_{\mathrm{true}}\,M_{\mathrm{true}}$)")
    ax.plot(t, alpha_est * result["M_est"], linestyle="-", label=r"Filtered ($\alpha_{\mathrm{est}}\,M_{\mathrm{est}}$)", **filt_kwargs)
    if has_smooth:
        ax.plot(t, alpha_est * result["M_smooth"], "r-", lw=1.5, label=r"Smoothed ($\alpha_{\mathrm{est}}\,M_{\mathrm{smooth}}$)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Fluorescence")
    ax.set_title("Fluorescence: Measured Vs. Reconstructed")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, result["true_I"], "g--", lw=1.5, label="True $I(t)$")
    ax.plot(t, result["I_est"], linestyle="-", label="Filtered $I(t)$", **filt_kwargs)
    if has_smooth:
        ax.plot(t, result["I_smooth"], "r-", lw=1.5, label="Smoothed $I(t)$")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Immature protein concentration $I$")
    ax.set_title("Immature Protein Pool")
    ax.legend()
    ax.grid(alpha=0.3)

    if result["is_two_step"]:
        ax = axes[0, 2]
        ax.plot(t, result["true_X"], "g--", lw=1.5, label="True $X(t)$")
        ax.plot(t, result["X_est"], linestyle="-", label="Filtered $X(t)$", **filt_kwargs)
        if has_smooth:
            ax.plot(t, result["X_smooth"], "r-", lw=1.5, label="Smoothed $X(t)$")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Intermediate protein concentration $X$")
        ax.set_title("Intermediate Protein Pool")
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, result["true_M"], "g--", lw=1.5, label="True $M(t)$")
    ax.plot(t, result["M_est"], linestyle="-", label="Filtered $M(t)$", **filt_kwargs)
    if has_smooth:
        ax.plot(t, result["M_smooth"], "r-", lw=1.5, label="Smoothed $M(t)$")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mature protein concentration $M$")
    ax.set_title("Mature Protein Pool")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, result["true_u"], "g--", lw=1.5, label="True $u(t)$ (gene expression)")
    ax.plot(t, result["u_est"], linestyle="-", label="Filtered $u(t)$", **filt_kwargs)
    if has_smooth:
        ax.plot(t, result["u_smooth"], "r-", lw=1.5, label="Smoothed $u(t)$")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Synthesis rate $u$")
    ax.set_title("Reconstructed Gene Expression Rate (The Target)")
    ax.legend()
    ax.grid(alpha=0.3)

    if result["is_two_step"]:
        axes[1, 2].axis("off")

    fig.tight_layout()
    show_figure(fig)

    def _rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    st.markdown("**Comparative RMSE (filtered vs. smoothed, against ground truth)**")
    rows = [
        {
            "Quantity": "Fluorescence",
            "Filtered RMSE": _rmse(alpha_est * result["M_est"], alpha_true * result["true_M"]),
            "Smoothed RMSE": (
                _rmse(alpha_est * result["M_smooth"], alpha_true * result["true_M"]) if has_smooth else None
            ),
        },
        {
            "Quantity": "I(t)",
            "Filtered RMSE": _rmse(result["I_est"], result["true_I"]),
            "Smoothed RMSE": _rmse(result["I_smooth"], result["true_I"]) if has_smooth else None,
        },
    ]
    if result["is_two_step"]:
        rows.append({
            "Quantity": "X(t)",
            "Filtered RMSE": _rmse(result["X_est"], result["true_X"]),
            "Smoothed RMSE": _rmse(result["X_smooth"], result["true_X"]) if has_smooth else None,
        })
    rows.append({
        "Quantity": "M(t)",
        "Filtered RMSE": _rmse(result["M_est"], result["true_M"]),
        "Smoothed RMSE": _rmse(result["M_smooth"], result["true_M"]) if has_smooth else None,
    })
    rows.append({
        "Quantity": "u(t) (gene expression)",
        "Filtered RMSE": _rmse(result["u_est"], result["true_u"]),
        "Smoothed RMSE": _rmse(result["u_smooth"], result["true_u"]) if has_smooth else None,
    })

    rmse_df = pd.DataFrame(rows).set_index("Quantity")
    if not has_smooth:
        st.caption("This run predates the RTS smoother -- only the filtered RMSE is available.")
        rmse_df = rmse_df.drop(columns=["Smoothed RMSE"])
    st.dataframe(rmse_df.style.format("{:.4f}"))


def render_synthetic_expression_result(result):
    """Render the ground-truth u(t)/I/(X)/M/fluorescence plots for one
    Synthetic Gene Expression run.

    Same layout convention as `render_kalman_result` (2x2 for 1-step, 2x3
    for 2-step), but with only "true"/noisy-measurement curves -- there is
    no filter estimate yet at this stage.
    """
    t = result["t"]
    alpha = result["alpha"]

    if result["is_two_step"]:
        fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    ax.scatter(t, result["z_n"], c="gray", s=15, alpha=0.5, label="Noisy fluorescence measurements")
    ax.plot(t, alpha * result["true_M"], "g--", lw=1.5, label=r"True fluorescence ($\alpha\,M_{\mathrm{true}}$)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Fluorescence")
    ax.set_title("Fluorescence: True Vs. Noisy Measurement")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, result["true_I"], "g--", lw=1.5, label="True $I(t)$")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Immature protein concentration $I$")
    ax.set_title("Immature Protein Pool")
    ax.legend()
    ax.grid(alpha=0.3)

    if result["is_two_step"]:
        ax = axes[0, 2]
        ax.plot(t, result["true_X"], "g--", lw=1.5, label="True $X(t)$")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Intermediate protein concentration $X$")
        ax.set_title("Intermediate Protein Pool")
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, result["true_M"], "g--", lw=1.5, label="True $M(t)$")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mature protein concentration $M$")
    ax.set_title("Mature Protein Pool")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, result["true_u"], "g--", lw=1.5, label="True $u(t)$ (gene expression)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Synthesis rate $u$")
    ax.set_title("Gene Expression Signal (User-Defined $u(t)$)")
    ax.legend()
    ax.grid(alpha=0.3)

    if result["is_two_step"]:
        axes[1, 2].axis("off")

    fig.tight_layout()
    show_figure(fig)

    # Gene expression and protein pools on one plot. u(t) is a rate (/min),
    # typically far smaller than the protein amounts, so it gets its own
    # right axis.
    t = np.asarray(t, dtype=float)
    true_u = np.asarray(result["true_u"], dtype=float)
    true_M = np.asarray(result["true_M"], dtype=float)

    fig, ax = plt.subplots(figsize=(15, 7))
    observed = [ax.scatter(
        t, result["z_n"], color=SPECIES_COLORS["F"], s=15, alpha=0.5,
        label="Noisy fluorescence measurements",
    )]
    hidden = ax.plot(t, result["true_I"], lw=1.5, color=SPECIES_COLORS["I"], label="$I(t)$ (immature)")
    if result["is_two_step"]:
        hidden += ax.plot(t, result["true_X"], lw=1.5, color=SPECIES_COLORS["X"], label="$X(t)$ (intermediate)")
    hidden += ax.plot(t, true_M, lw=1.5, color=SPECIES_COLORS["M"], label="$M(t)$ (mature)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Protein Concentration / Fluorescence")

    ax_u = ax.twinx()
    hidden += ax_u.plot(t, true_u, "--", color=SPECIES_COLORS["u"], lw=1.5, label="$u(t)$ (gene expression)")
    ax_u.set_ylabel("Synthesis rate $u$ (/min)")
    # Both axes start at 0, and the u axis runs to twice u's peak so u(t)
    # stays in the lower half instead of stretching over the plot.
    left_max = max(
        [float(np.max(result["z_n"]))] + [np.max(line.get_ydata()) for line in ax.get_lines()]
    )
    ax.set_ylim(0, 1.05 * left_max if left_max > 0 else 1)
    u_max = float(np.max(true_u))
    if u_max > 0:
        ax_u.set_ylim(0, 2 * u_max)

    # Legend grouped under "Observed" (what the experiment measures) and
    # "Hidden" (the states the Kalman filter has to reconstruct), placed
    # outside the plot so it never covers the data.
    _grouped_legend(ax, {"Observed": observed, "Hidden": hidden},
                    loc="upper left", bbox_to_anchor=(1.12, 1.0))
    ax.set_title("Gene Expression Vs. Protein Pools")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    show_figure(fig)


def _grouped_legend(ax, groups, **legend_kwargs):
    """Legend on `ax` with a bold heading row above each group of artists.

    `groups` maps heading text to a list of plotted artists. Heading rows
    have no marker, and their empty marker slot is collapsed so the heading
    lines up flush left with the legend box.
    """
    handles, labels, heading_rows = [], [], set()
    for heading, artists in groups.items():
        heading_rows.add(len(handles))
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(rf"$\mathbf{{{heading}}}$")
        handles += artists
        labels += [a.get_label() for a in artists]
    legend = ax.legend(handles, labels, **legend_kwargs)
    # Collapse the marker slot of heading rows. Uses matplotlib's legend
    # layout internals, so skip quietly (heading just stays indented) if
    # those ever change.
    try:
        rows = [row for column in legend._legend_handle_box.get_children()
                for row in column.get_children()]
        for i in heading_rows:
            rows[i].get_children()[0].width = 0
            rows[i].sep = 0
    except (AttributeError, IndexError):
        pass
    return legend


def render_profile_2d_result(profile_df):
    """Render the 2D (a, b) SSE contour plot + raw data table for one 2D profile likelihood run.

    Shared by the Profile Likelihood tab (right after a run) and the Profile
    Likelihood History tab (replaying a stored past run).
    """
    best_row = profile_df.loc[profile_df["sse"].idxmin()]
    st.success(
        f"Lowest SSE {best_row['sse']:.6g} at a = {best_row['a']:.6g}, b = {best_row['b']:.6g}"
    )

    fig = plot_profile_2d(profile_df)
    show_figure(fig)

    with st.expander("2D profile likelihood raw data"):
        st.dataframe(profile_df, hide_index=True)
