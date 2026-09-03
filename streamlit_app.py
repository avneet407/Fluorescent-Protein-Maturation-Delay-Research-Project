"""Streamlit UI for exploring fluorescent protein maturation/bleaching models.

Lets a user simulate 1-step and 2-step maturation ODE models, generate or
upload fluorescence decay data, fit model parameters to that data (single
fit or many randomized multi-start fits), profile individual parameters'
identifiability, inspect frequency-domain (Bode) behavior, and separately
fit a pure photobleaching decay model plus a "known bleaching pole" variant
of the full model.

This file is the thin entry point: it sets up the page and the tab strip,
then delegates each tab's content to a `render_*` function in `app/`. The
Simulation tab's render function returns a `sim_state` dict of its current
widget values, passed into the Data and Bode Plot tabs (the only two that
need "what's currently set in the Simulation tab" as defaults) -- the same
data flow the original single-file version had via plain script-level
variables, made explicit now that each tab lives in its own module. Every
other tab communicates via `st.session_state`, same as before.

The plain-Python model/fitting/plotting logic lives in the sibling modules
imported by the `app/` tab modules (Maturation_Models.py,
Bleaching_Only_Model.py, etc.) and has no Streamlit dependency.
"""

import streamlit as st

from app.tab_simulation import render_simulation_tab
from app.tab_data import render_data_tab
from app.tab_bleaching import render_bleaching_tab
from app.tab_bleaching_history import render_bleaching_history_tab
from app.tab_fitting import render_fitting_tab
from app.tab_profile_likelihood import render_profile_likelihood_tab
from app.tab_bode import render_bode_tab
from app.tab_multistart_history import render_multistart_history_tab
from app.tab_profile_history import render_profile_history_tab

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
# Tabs, in display order:
#   Simulation              - run the full maturation model, set live params
#   Data                    - upload/generate the fluorescence trace to fit
#   Bleaching Only Simulation - pure photobleaching-decay model: fit it, fit
#                              the full model with its decay pole fixed, and
#                              Bode-plot both against the synthetic input
#   Bleaching Fit History    - saved runs from the tab above
#   Least Squares Fitting    - single and multi-start fits of the full model
#   Profile Likelihood       - parameter identifiability sweeps (1D and 2D)
#   Bode Plot                - frequency response of the full model
#   Multi-Start History      - saved multi-start fit runs
#   Profile Likelihood History - saved profile likelihood runs

st.set_page_config(page_title="Fluorescent Protein Maturation", layout="wide")
st.title("Fluorescent Protein Maturation Delay Model")

(
    sim_tab, upload_tab, bleach_tab, bleach_history_tab, fit_tab, profile_tab, bode_tab,
    ms_history_tab, pl_history_tab,
) = st.tabs(
    [
        "Simulation", "Data", "Bleaching Only Simulation", "Bleaching Fit History",
        "Least Squares Fitting", "Profile Likelihood", "Bode Plot", "Multi-Start History",
        "Profile Likelihood History",
    ]
)

with sim_tab:
    sim_state = render_simulation_tab()

with upload_tab:
    render_data_tab(sim_state)

with bleach_tab:
    render_bleaching_tab()

with bleach_history_tab:
    render_bleaching_history_tab()

with fit_tab:
    render_fitting_tab()

with profile_tab:
    render_profile_likelihood_tab()

with bode_tab:
    render_bode_tab(sim_state)

with ms_history_tab:
    render_multistart_history_tab()

with pl_history_tab:
    render_profile_history_tab()
