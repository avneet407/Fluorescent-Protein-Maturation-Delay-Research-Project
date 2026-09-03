# --- Bleaching Fit History tab: saved runs from Bleaching Only Sim. ------
# Loaded from disk (bleach_fit_history.json). Each entry bundles up to two
# pieces saved together by one click of the known-b fit's "Run Multi-Start
# Fit" button: the bleach-only fit and the known bleaching pole fit.
# Older entries (saved before Bode plots were removed from the tab) may
# still carry "bleach_bode"/"known_b_bode" data; it's simply not displayed.

import streamlit as st

from history_store import load_bleach_history, clear_bleach_history, delete_bleach_entry

from app.shared import render_multi_start_results


def render_bleaching_history_tab():
    st.markdown(
        "Every time the known bleaching pole fit's **Run Multi-Start Fit** "
        "button (in the **Bleaching Only Simulation** tab) is clicked, "
        "everything currently run in that tab — the bleach-only fit and the "
        "known bleaching pole fit — is saved together as one entry, to disk "
        "(`bleach_fit_history.json`) so it survives app restarts — most "
        "recent first. Click **Display** to view both again, exactly as "
        "they were when saved, even after changing tab parameters or "
        "running other fits since."
    )

    bleach_history = load_bleach_history()

    if not bleach_history:
        st.info(
            "No runs saved yet. In the **Bleaching Only Simulation** tab, "
            "run the bleach-only fit, then click **Run Multi-Start Fit** "
            "under the known bleaching pole fit to save everything "
            "currently run in that tab to history."
        )
    else:
        displayed_bleach = st.session_state.setdefault("bleach_history_displayed", set())

        bh_clear_col1, bh_clear_col2 = st.columns([3, 1])
        with bh_clear_col2:
            if st.button("Clear history", key="clear_bleach_history"):
                clear_bleach_history()
                st.session_state["bleach_history_displayed"] = set()
                st.rerun()

        for i in range(len(bleach_history) - 1, -1, -1):
            entry = bleach_history[i]
            bleach_part = entry.get("bleach_only")
            kb_part = entry.get("known_b")

            summary_bits = []
            if bleach_part is not None:
                summary_bits.append(
                    f"bleaching-only fit ({bleach_part['n_converged']}/{bleach_part['n_total']} converged)"
                )
            if kb_part is not None:
                kb_extra = kb_part.get("extra_info", {})
                kb_model_label = "2-step" if kb_extra.get("fit_is_two_step") else "1-step"
                summary_bits.append(
                    f"known bleaching pole fit, {kb_model_label} model, "
                    f"b fixed at {kb_extra.get('b_known', float('nan')):.4f} "
                    f"({kb_part['n_converged']}/{kb_part['n_total']} converged)"
                )
            summary_line = "; ".join(summary_bits) if summary_bits else "no results saved"

            with st.container(border=True):
                st.markdown(f"**Run** — {entry['timestamp']} — {summary_line}")
                bh_col1, bh_col2, bh_col3 = st.columns([2, 1, 1])
                with bh_col1:
                    bh_include_nonconverged = st.checkbox(
                        "Include non-converged runs", value=False, key=f"bleach_history_nonconv_{i}",
                    )
                with bh_col2:
                    bh_display_clicked = st.button("Display", key=f"bleach_history_display_{i}")
                with bh_col3:
                    bh_delete_clicked = st.button("Delete", key=f"bleach_history_delete_{i}")

                if bh_delete_clicked:
                    delete_bleach_entry(i)
                    st.session_state["bleach_history_displayed"] = set()
                    st.rerun()

                if bh_display_clicked:
                    displayed_bleach.add(i)

                if i in displayed_bleach:
                    if bleach_part is not None:
                        st.markdown("#### Bleaching-only fit")
                        render_multi_start_results(
                            bleach_part["results_df"], bleach_part["param_names"],
                            bleach_part["derived_names"], bleach_part["true_values"],
                            bh_include_nonconverged,
                        )
                    else:
                        st.info("No bleaching-only fit result was saved with this entry.")

                    if kb_part is not None:
                        st.markdown("#### Known bleaching pole fit (full model)")
                        render_multi_start_results(
                            kb_part["results_df"], kb_part["param_names"],
                            kb_part["derived_names"], kb_part["true_values"],
                            bh_include_nonconverged,
                        )
                    else:
                        st.info("No known bleaching pole fit result was saved with this entry.")
