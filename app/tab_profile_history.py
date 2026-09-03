# --- Profile Likelihood History tab: saved profile-likelihood sweeps -----
# Loaded from disk (profile_likelihood_history.json), grouped by the dataset
# they were run against and then by run, with a 1-step/2-step filter since a
# dataset can carry runs of both model types.

import streamlit as st

from history_store import (
    load_profile_history,
    clear_profile_history,
    delete_profile_entry,
)

from app.shared import render_profile_likelihood_result, render_profile_2d_result


def render_profile_history_tab():
    st.markdown(
        "Every profile likelihood run, saved to disk "
        "(`profile_likelihood_history.json`) so it survives app restarts. "
        "Runs are grouped by the **dataset** they were run against "
        "(regenerating/re-uploading the same trace groups new runs alongside "
        "past ones), then by **run** — one specific fitting region + model "
        "choice, since multiple parameters/quantities can each be profiled "
        "against the same run (e.g. Run 1: km, then Run 1: kb). Each dataset "
        "group records the synthetic ground truth and noise settings used to "
        "generate it, if applicable."
    )

    pl_history = load_profile_history()

    if not pl_history:
        st.info(
            "No profile likelihood runs saved yet. Run one from the "
            "**Profile Likelihood** tab."
        )
    else:
        displayed_pl = st.session_state.setdefault("pl_history_displayed", set())

        clear_col1, clear_col2 = st.columns([3, 1])
        with clear_col2:
            if st.button("Clear history", key="clear_pl_history"):
                clear_profile_history()
                st.session_state["pl_history_displayed"] = set()
                st.rerun()

        # Group entries by dataset, preserving first-seen (oldest-first) order.
        datasets = {}
        dataset_order = []
        for idx, entry in enumerate(pl_history):
            dk = entry["dataset_key"]
            if dk not in datasets:
                datasets[dk] = {
                    "label": entry["dataset_label"],
                    "synthetic_params": entry["synthetic_params"],
                    "noise_params": entry["noise_params"],
                    "entries": [],
                }
                dataset_order.append(dk)
            datasets[dk]["entries"].append((idx, entry))

        def _render_pl_dataset_group(group, run_items):
            """Render one dataset's ground-truth/noise header plus the given (n, rk, run_entries) runs.

            `run_items` is pre-filtered to the currently selected model type.
            """
            st.markdown(f"**Dataset: {group['label']}**")

            synthetic_params = group["synthetic_params"]
            noise_params = group["noise_params"]
            if synthetic_params is not None:
                truth_bits = []
                if synthetic_params.get("is_two_step"):
                    truth_bits.append(f"k1={synthetic_params['k1']:.4g}")
                    truth_bits.append(f"k2={synthetic_params['k2']:.4g}")
                else:
                    truth_bits.append(f"km={synthetic_params['km']:.4g}")
                truth_bits.append(f"I0={synthetic_params['I0']:.4g}")
                truth_bits.append(f"kb={synthetic_params['kb']:.4g}")
                truth_bits.append(f"kd={synthetic_params['kd']:.4g}")
                truth_bits.append(f"alpha={synthetic_params['alpha']:.4g}")
                st.caption("Synthetic input: " + ", ".join(truth_bits))

                if noise_params is not None:
                    noise_bits = []
                    for key in ("km_noise_std", "k1_noise_std", "k2_noise_std", "kb_noise_std"):
                        val = noise_params.get(key)
                        if val is not None:
                            noise_bits.append(f"{key}={val:.4g}")
                    if noise_params.get("measurement_noise_std") is not None:
                        noise_bits.append(
                            f"measurement_noise_std={noise_params['measurement_noise_std']:.4g}"
                        )
                    seed_val = noise_params.get("seed")
                    noise_bits.append(f"seed={seed_val if seed_val is not None else 'random'}")
                    st.caption("Noise added: " + ", ".join(noise_bits))
            else:
                st.caption("Experimental data (no synthetic ground truth).")

            for n, rk, run_entries in run_items:
                first_entry = run_entries[0][1]
                model_label = "2-step" if first_entry["fit_is_two_step"] else "1-step"
                st.markdown(
                    f"Run {n} — {model_label} model, "
                    f"t=[{first_entry['time_start']:.4g}, {first_entry['time_end']:.4g}] sec, "
                    f"baseline={first_entry['baseline']:.4g}, u={first_entry['u_step']:.4g}"
                )

                for idx, entry in run_entries:
                    profile_type = entry.get("profile_type", "1d")
                    if profile_type == "2d":
                        label_line = (
                            f"&nbsp;&nbsp;**{entry['profile_target']}** "
                            f"— {entry['timestamp']}"
                        )
                    else:
                        kind_label = "derived quantity" if entry["is_derived"] else "raw parameter"
                        label_line = (
                            f"&nbsp;&nbsp;**{entry['profile_target']}** ({kind_label}) "
                            f"— {entry['timestamp']}"
                        )

                    row_col1, row_col2, row_col3 = st.columns([3, 1, 1])
                    with row_col1:
                        st.markdown(label_line)
                    with row_col2:
                        display_clicked_p = st.button("Display", key=f"pl_history_display_{idx}")
                    with row_col3:
                        delete_clicked_p = st.button("Delete", key=f"pl_history_delete_{idx}")

                    if delete_clicked_p:
                        delete_profile_entry(idx)
                        st.session_state["pl_history_displayed"] = set()
                        st.rerun()

                    if display_clicked_p:
                        displayed_pl.add(idx)

                    if idx in displayed_pl:
                        if profile_type == "2d":
                            render_profile_2d_result(entry["profile_df"])
                        else:
                            render_profile_likelihood_result(
                                entry["profile_df"], entry["profile_target"], entry["true_value"],
                            )

        pl_model_choice = st.radio(
            "Show runs for model", ["1-step", "2-step"], horizontal=True, key="pl_history_model_filter",
        )
        pl_show_two_step = pl_model_choice == "2-step"

        for dk in reversed(dataset_order):  # most recently seen dataset first
            group = datasets[dk]

            # Group this dataset's entries by run, numbering runs in first-seen order.
            runs = {}
            run_order = []
            for idx, entry in group["entries"]:
                rk = entry["run_key"]
                if rk not in runs:
                    runs[rk] = []
                    run_order.append(rk)
                runs[rk].append((idx, entry))

            # Keep each run's original position (n) as its stable label
            # within the dataset, filtering to the selected model type.
            matching_runs = [
                (n, rk, runs[rk]) for n, rk in enumerate(run_order, start=1)
                if runs[rk][0][1]["fit_is_two_step"] == pl_show_two_step
            ]

            if matching_runs:
                with st.container(border=True):
                    _render_pl_dataset_group(group, matching_runs)
