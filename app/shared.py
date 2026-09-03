"""Render helpers and constants shared by more than one tab module.

Each function here is used by a live-run tab (right after a fit/plot) and
its matching history tab (replaying a saved run), so they're factored out
here rather than duplicated.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from multi_start_plots import plot_histograms
from profile_likelihood_2D import plot_profile_2d

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
    ax_mag.set_xlabel("Frequency (rad/sec)")
    ax_mag.set_title(title)
    ax_mag.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_mag.legend(fontsize=8)

    ax_phase.semilogx(br["w"], br["phase_syn"], color="tab:green", linestyle="--", label="Synthetic input")
    ax_phase.semilogx(br["w"], br["phase_fit"], color="tab:red", linestyle=":", label=br["fit_label"])
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.set_xlabel("Frequency (rad/sec)")
    ax_phase.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_phase.legend(fontsize=8)

    fig_bode.tight_layout()
    st.pyplot(fig_bode)

    cutoff_col1, cutoff_col2 = st.columns(2)
    if br["cutoff_syn"] is not None:
        cutoff_col1.metric("Synthetic input -3 dB cutoff (rad/sec)", f"{br['cutoff_syn']:.6f}")
    else:
        cutoff_col1.warning("Synthetic input cutoff not found (no positive real root).")
    if br["cutoff_fit"] is not None:
        cutoff_col2.metric("Multi-start fit -3 dB cutoff (rad/sec)", f"{br['cutoff_fit']:.6f}")
    else:
        cutoff_col2.warning("Multi-start fit cutoff not found (no positive real root).")


def render_multi_start_results(results_df, param_names, derived_names, true_values, include_nonconverged):
    """Render the histograms + summary table for one multi-start fit result.

    Shared by the Least Squares Fitting tab (right after a run) and the
    History tab (replaying a stored past run).
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
    st.pyplot(fig_raw)

    st.markdown("**Derived quantities across runs**")
    fig_der = plot_histograms(plot_df, derived_names, true_values, color="tab:purple")
    st.pyplot(fig_der)

    st.markdown("**Summary statistics across runs (mean, std, coefficient of variation)**")
    summary_rows = []
    for name in param_names + derived_names:
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
            ["run", "Type"] + param_names + derived_names + ["cost", "converged", "message", "nfev"]
        )
        detail_rows = []
        for _, r in results_df.iterrows():
            detail_rows.append({
                "run": int(r["run"]),
                "Type": "Fitted",
                **{name: r[name] for name in param_names},
                **{name: f"{r[name]:.6g}" for name in derived_names},
                "cost": f"{r['cost']:.6g}",
                "converged": str(bool(r["converged"])),
                "message": str(r["message"]),
                "nfev": str(int(r["nfev"])),
            })
            detail_rows.append({
                "run": int(r["run"]),
                "Type": "Initial guess",
                **{name: r[f"{name}_init"] for name in param_names},
                **{name: "n/a" for name in derived_names},
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
    ax_p.set_xlabel(profile_target)
    ax_p.set_ylabel("SSE (sum of squared residuals)")
    ax_p.set_title(f"Profile likelihood: {profile_target}")
    ax_p.legend(fontsize=8)
    fig_p.tight_layout()
    st.pyplot(fig_p)

    with st.expander("Profile likelihood raw data"):
        st.dataframe(profile_df, hide_index=True)


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
    st.pyplot(fig)

    with st.expander("2D profile likelihood raw data"):
        st.dataframe(profile_df, hide_index=True)
