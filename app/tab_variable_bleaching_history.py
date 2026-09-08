# --- Variable Bleaching History tab: saved joint least-squares fit runs --
# Loaded from disk (variable_bleaching_history.json), most recent first.
# Each entry is one "Run Multi-Start Fit (N runs)" click from the Variable
# Bleaching tab: the shared km (or k1/k2) and every trace's kb/K, re-fit
# from many randomized initial guesses to check convergence robustness, the
# noise std devs used to generate the fitted synthetic data, and fit
# diagnostics. Older entries (saved before this tab used multi-start
# fitting) carry a single-run result instead and are displayed separately.

import numpy as np
import pandas as pd
import streamlit as st

from history_store import (
    load_variable_bleaching_history,
    clear_variable_bleaching_history,
    delete_variable_bleaching_entry,
)

from multi_start_plots import plot_histograms


def _render_multi_start_entry(entry, index):
    results_df = entry["results_df"]
    n_converged = int(results_df["converged"].sum())
    st.markdown(f"**Multi-start fit: {n_converged} / {len(results_df)} runs converged**")

    include_nonconverged = st.checkbox(
        "Include non-converged runs", value=False, key=f"vb_history_nonconv_{index}",
    )
    plot_df = results_df if include_nonconverged else results_df[results_df["converged"]]

    if len(plot_df) == 0:
        st.warning("No runs to display (no converged runs, and non-converged runs are excluded).")
        return

    shared_names = entry["shared_names"]
    shared_label = "k1, k2" if entry["is_two_step"] else "km"
    true_values = entry["true_values"]

    st.markdown(f"**{shared_label} across runs** (shared across all traces)")
    fig_shared = plot_histograms(plot_df, shared_names, true_values, color="tab:blue")
    st.pyplot(fig_shared)

    summary_rows = []
    for name in shared_names:
        vals = plot_df[name].to_numpy(dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append({
            "Quantity": name,
            "Synthetic input": true_values[name],
            "Mean (across runs)": mean,
            "Std dev": std,
            "CV": std / mean if mean != 0 else float("nan"),
        })
    st.markdown("**Synthetic input vs. jointly fitted shared parameters**")
    st.dataframe(pd.DataFrame(summary_rows).set_index("Quantity"))

    trace_labels = entry["trace_labels"]
    kb_display_names = [f"kb ({label})" for label in trace_labels]
    K_display_names = [f"K ({label})" for label in trace_labels]
    display_df = plot_df.copy()
    display_true_values = dict(true_values)
    for i, label in enumerate(trace_labels):
        kb_name, K_name = f"kb_{i + 1}", f"K_{i + 1}"
        display_df[kb_display_names[i]] = display_df[kb_name]
        display_df[K_display_names[i]] = display_df[K_name]
        display_true_values[kb_display_names[i]] = true_values[kb_name]
        display_true_values[K_display_names[i]] = true_values[K_name]

    st.markdown("**kb across runs, per trace**")
    fig_kb = plot_histograms(display_df, kb_display_names, display_true_values, color="tab:purple")
    st.pyplot(fig_kb)

    st.markdown("**K across runs, per trace** (K = G\\*I0 / G3\\*I0)")
    fig_K = plot_histograms(display_df, K_display_names, display_true_values, color="tab:green")
    st.pyplot(fig_K)

    st.markdown(
        "**Per-trace fitted parameters** (kb, and K = G\\*I0 / G3\\*I0 — I0 and "
        "alpha aren't fit individually), mean ± std across runs"
    )
    trace_rows = []
    for i, label in enumerate(entry["trace_labels"]):
        kb_name, K_name = f"kb_{i + 1}", f"K_{i + 1}"
        kb_vals = plot_df[kb_name].to_numpy(dtype=float)
        K_vals = plot_df[K_name].to_numpy(dtype=float)
        trace_rows.append({
            "Label": label,
            "kb (true)": true_values[kb_name],
            "kb (mean fitted)": float(np.mean(kb_vals)),
            "kb (std)": float(np.std(kb_vals, ddof=1)) if len(kb_vals) > 1 else 0.0,
            "K (true)": true_values[K_name],
            "K (mean fitted)": float(np.mean(K_vals)),
            "K (std)": float(np.std(K_vals, ddof=1)) if len(K_vals) > 1 else 0.0,
        })
    st.dataframe(pd.DataFrame(trace_rows).set_index("Label"))

    with st.expander(f"All {len(results_df)} runs (raw table)"):
        st.dataframe(results_df, hide_index=True)


def _render_legacy_single_run_entry(entry):
    st.caption(
        "Saved before this tab used multi-start fitting — shows the single "
        "fit run recorded at the time."
    )
    shared_init = entry.get("shared_init")

    if entry["is_two_step"]:
        shared_cols = st.columns(2)
        for col, idx, name in ((shared_cols[0], 0, "k1"), (shared_cols[1], 1, "k2")):
            if shared_init is not None:
                col.metric(
                    f"{name} — true / init guess / jointly fitted",
                    f"{entry['shared_true'][idx]:.4f} / {shared_init[idx]:.4f} / "
                    f"{entry['shared_fitted'][idx]:.4f}",
                )
            else:
                col.metric(
                    f"{name} — true / jointly fitted",
                    f"{entry['shared_true'][idx]:.4f} / {entry['shared_fitted'][idx]:.4f}",
                )
    else:
        if shared_init is not None:
            st.metric(
                "km — true / init guess / jointly fitted (shared across all traces)",
                f"{entry['shared_true'][0]:.4f} / {shared_init[0]:.4f} / "
                f"{entry['shared_fitted'][0]:.4f}",
            )
        else:
            st.metric(
                "km — true / jointly fitted (shared across all traces)",
                f"{entry['shared_true'][0]:.4f} / {entry['shared_fitted'][0]:.4f}",
            )

    converged_label = "converged" if entry["converged"] else "did NOT converge"
    st.markdown(
        f"Fit {converged_label} ({entry['message']}), "
        f"cost={entry['cost']:.6g}, nfev={entry['nfev']}."
    )

    st.markdown(
        "**Per-trace fitted parameters** (kb, and K = G\\*I0 / G3\\*I0), "
        "with true/init-guess/fitted values"
    )
    st.dataframe(pd.DataFrame(entry["per_trace_rows"]).set_index("Label"))


def render_variable_bleaching_history_tab():
    st.markdown(
        "Every multi-start joint least-squares fit run from the **Variable "
        "Bleaching** tab, saved to disk (`variable_bleaching_history.json`) "
        "so it survives app restarts — most recent first. Each entry keeps "
        "the shared km (or k1/k2) and per-trace kb/K results across all "
        "runs, the noise std devs used to generate the data it was fit "
        "against, and fit diagnostics exactly as they were at the time it "
        "ran — click **Display** to view it again."
    )

    history = load_variable_bleaching_history()

    if not history:
        st.info(
            "No runs saved yet. In the **Variable Bleaching** tab, generate "
            "synthetic data and click **Run Multi-Start Fit (N runs)** to "
            "save a run here."
        )
        return

    displayed_vb = st.session_state.setdefault("vb_history_displayed", set())

    clear_col1, clear_col2 = st.columns([3, 1])
    with clear_col2:
        if st.button("Clear history", key="clear_vb_history"):
            clear_variable_bleaching_history()
            st.session_state["vb_history_displayed"] = set()
            st.rerun()

    for i in range(len(history) - 1, -1, -1):
        entry = history[i]
        is_multi_start = "results_df" in entry
        model_label = "2-step" if entry["is_two_step"] else "1-step"
        shared_label = "k1, k2" if entry["is_two_step"] else "km"

        if is_multi_start:
            results_df = entry["results_df"]
            n_converged = int(results_df["converged"].sum())
            shared_true = [entry["true_values"][n] for n in entry["shared_names"]]
            shared_true_str = ", ".join(f"{v:.4f}" for v in shared_true)
            summary_line = (
                f"{model_label} model, {entry['n_traces']} traces, {entry['n_runs']} runs "
                f"({n_converged}/{len(results_df)} converged), {shared_label} true "
                f"[{shared_true_str}]"
            )
        else:
            converged_label = "converged" if entry["converged"] else "did NOT converge"
            shared_true_str = ", ".join(f"{v:.4f}" for v in entry["shared_true"])
            shared_fitted_str = ", ".join(f"{v:.4f}" for v in entry["shared_fitted"])
            summary_line = (
                f"{model_label} model, {entry['n_traces']} traces (single run, legacy), "
                f"{shared_label} true [{shared_true_str}] / fitted [{shared_fitted_str}], "
                f"{converged_label}"
            )

        with st.container(border=True):
            st.markdown(f"**Run** — {entry['timestamp']} — {summary_line}")
            row_cols = st.columns([2, 1])
            with row_cols[0]:
                display_clicked = st.button("Display", key=f"vb_history_display_{i}")
            with row_cols[1]:
                delete_clicked = st.button("Delete", key=f"vb_history_delete_{i}")

            if delete_clicked:
                delete_variable_bleaching_entry(i)
                st.session_state["vb_history_displayed"] = set()
                st.rerun()

            if display_clicked:
                displayed_vb.add(i)

            if i in displayed_vb:
                if is_multi_start:
                    _render_multi_start_entry(entry, i)
                else:
                    _render_legacy_single_run_entry(entry)

                np_ = entry["noise_params"]
                noise_bits = []
                if np_.get("km_noise_std") is not None:
                    noise_bits.append(f"km noise std={np_['km_noise_std']:.4f}")
                if np_.get("k1_noise_std") is not None:
                    noise_bits.append(f"k1 noise std={np_['k1_noise_std']:.4f}")
                if np_.get("k2_noise_std") is not None:
                    noise_bits.append(f"k2 noise std={np_['k2_noise_std']:.4f}")
                noise_bits.append(f"kb noise std={np_['kb_noise_std']:.4f}")
                noise_bits.append(f"measurement noise std={np_['measurement_noise_std']:.4f}")
                noise_bits.append(
                    f"data seed={np_['seed']}" if np_.get("seed") is not None
                    else "data seed=random (not fixed)"
                )
                fit_seed = entry.get("fit_seed")
                noise_bits.append(
                    f"init-guess seed={fit_seed}" if fit_seed is not None
                    else "init-guess seed=random (not fixed)"
                )
                st.caption(
                    f"u={entry['u']:.4f}, alpha={entry['alpha']:.4f} (used to generate the data); "
                    + ", ".join(noise_bits)
                )
