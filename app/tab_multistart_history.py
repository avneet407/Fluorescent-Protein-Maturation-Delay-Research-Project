# --- Multi-Start History tab: saved multi-start fit runs -----------------
# Loaded from disk (multi_start_history.json), most recent first, with a
# 1-step/2-step filter to pick which runs are listed.

import streamlit as st

from history_store import (
    load_multi_start_history,
    clear_multi_start_history,
    delete_multi_start_entry,
)

from app.shared import render_multi_start_results


def render_multistart_history_tab():
    st.markdown(
        "Every multi-start fit run, saved to disk (`multi_start_history.json`) "
        "so it survives app restarts — most recent first. Each entry keeps its "
        "fitted results and the synthetic-data ground truth (if any) exactly "
        "as they were at the time it ran — click **Display** to view its "
        "histograms and summary statistics again, even after changing "
        "Simulation tab parameters or running other fits since."
    )

    history = load_multi_start_history()

    if not history:
        st.info(
            "No multi-start fit runs saved yet. Run one from the "
            "**Run Multi-Start Fit (N runs)** button in the Least Squares "
            "Fitting tab."
        )
    else:
        displayed_ms = st.session_state.setdefault("ms_history_displayed", set())

        clear_col1, clear_col2 = st.columns([3, 1])
        with clear_col2:
            if st.button("Clear history", key="clear_ms_history"):
                clear_multi_start_history()
                st.session_state["ms_history_displayed"] = set()
                st.rerun()

        ms_model_choice = st.radio(
            "Show runs for model", ["1-step", "2-step"], horizontal=True, key="ms_history_model_filter",
        )
        ms_show_two_step = ms_model_choice == "2-step"

        for i in range(len(history) - 1, -1, -1):
            entry = history[i]
            if entry["fit_is_two_step"] != ms_show_two_step:
                continue

            model_label = ms_model_choice
            source_label = (
                "synthetic data"
                if entry["data_source"] == "Generate synthetic data from simulation"
                else "experimental data"
            )
            truth_label = "available" if entry["true_values"] else "not available"

            with st.container(border=True):
                st.markdown(
                    f"**Run {i + 1}** — {entry['timestamp']} — {model_label} model, "
                    f"fit to {source_label}, {entry['n_converged']}/{entry['n_total']} "
                    f"converged, ground truth {truth_label}"
                )
                hist_col1, hist_col2, hist_col3 = st.columns([2, 1, 1])
                with hist_col1:
                    hist_include_nonconverged = st.checkbox(
                        "Include non-converged runs", value=False, key=f"history_nonconv_{i}",
                    )
                with hist_col2:
                    display_clicked = st.button("Display", key=f"history_display_{i}")
                with hist_col3:
                    delete_clicked = st.button("Delete", key=f"history_delete_{i}")

                if delete_clicked:
                    delete_multi_start_entry(i)
                    st.session_state["ms_history_displayed"] = set()
                    st.rerun()

                if display_clicked:
                    displayed_ms.add(i)

                if i in displayed_ms:
                    render_multi_start_results(
                        entry["results_df"], entry["param_names"], entry["derived_names"],
                        entry["true_values"], hist_include_nonconverged,
                    )
