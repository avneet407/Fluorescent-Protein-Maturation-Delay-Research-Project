# --- Kalman Filter History tab: saved Kalman filter runs -----------------
# Loaded from disk (kalman_history.json), most recent first, with a
# 1-step/2-step filter to pick which runs are listed. Each entry stores the
# full input params and result trajectories, so a past run's plots can be
# redrawn exactly without re-running the filter.

import streamlit as st

from history_store import (
    load_kalman_history,
    clear_kalman_history,
    delete_kalman_entry,
)

from app.shared import render_kalman_result


def render_kalman_history_tab():
    st.markdown(
        "Every Kalman filter run, saved to disk (`kalman_history.json`) so "
        "it survives app restarts -- most recent first. Each entry keeps the "
        "parameters it was run with and its full result trajectories -- "
        "click **Display** to redraw its plots again."
    )

    history = load_kalman_history()

    if not history:
        st.info(
            "No Kalman filter runs saved yet. Run one from the "
            "**Run Kalman Filter** button in the Kalman Filter tab."
        )
        return

    displayed = st.session_state.setdefault("kalman_history_displayed", set())

    clear_col1, clear_col2 = st.columns([3, 1])
    with clear_col2:
        if st.button("Clear history", key="clear_kalman_history"):
            clear_kalman_history()
            st.session_state["kalman_history_displayed"] = set()
            st.rerun()

    model_choice = st.radio(
        "Show runs for model", ["1-step", "2-step"], horizontal=True, key="kalman_history_model_filter",
    )
    show_two_step = model_choice == "2-step"

    for i in range(len(history) - 1, -1, -1):
        entry = history[i]
        result = entry["result"]
        if bool(result["is_two_step"]) != show_two_step:
            continue

        params = entry["params"]
        n_steps = int(params.get("n_steps", len(result["t"])))

        with st.container(border=True):
            st.markdown(
                f"**Run {i + 1}** -- {entry['timestamp']} -- {model_choice} model, "
                f"dt={params.get('dt'):.4g}, n_steps={n_steps}, "
                f"alpha={params.get('alpha'):.4g}, sigma_F={params.get('sigma_F'):.4g}"
            )
            hist_col1, hist_col2 = st.columns([1, 1])
            with hist_col1:
                display_clicked = st.button("Display", key=f"kalman_history_display_{i}")
            with hist_col2:
                delete_clicked = st.button("Delete", key=f"kalman_history_delete_{i}")

            if delete_clicked:
                delete_kalman_entry(i)
                st.session_state["kalman_history_displayed"] = set()
                st.rerun()

            if display_clicked:
                displayed.add(i)

            if i in displayed:
                with st.expander("Parameters used for this run"):
                    st.json(params)
                render_kalman_result(result)
