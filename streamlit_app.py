"""Streamlit UI for exploring fluorescent protein maturation/bleaching models.

Lets a user simulate 1-step and 2-step maturation ODE models, generate or
upload fluorescence decay data, fit model parameters to that data (single
fit or many randomized multi-start fits), profile individual parameters'
identifiability, inspect frequency-domain (Bode) behavior, separately fit a
pure photobleaching decay model plus a "known bleaching pole" variant of the
full model, generate a ground-truth noisy trace from a user-typed gene
expression signal u(t), and reconstruct hidden states from that trace with
a Kalman filter using separately calibrated rate constants.

This file is the thin entry point: it sets up the page, builds one
`st.Page` per tab wrapping a `render_*` function from `app/`, and groups
them into a sidebar `st.navigation` with two sections ("Parameter
Identification", "Kalman Filter") -- the sidebar's own section headers
already separate the two areas, so there's no separate Home page.
`st.navigation` runs only the current page's function each rerun (unlike
the old single-page tab strip, where every tab's code ran every rerun), so
the Simulation page's
`sim_state` -- needed as defaults by the Data page -- is stashed in
`st.session_state["sim_state"]` so the Data page can read it back on a later,
separate rerun. Every other page communicates purely via `st.session_state`,
same as before.

The plain-Python model/fitting/plotting logic lives in the sibling modules
imported by the `app/` tab modules (Maturation_Models.py,
Bleaching_Only_Model.py, Kalman_Filter/Kalman_Filter_Model.py, etc.) and has
no Streamlit dependency.
"""

import streamlit as st

from app.tab_simulation import render_simulation_tab
from app.tab_data import render_data_tab
from app.tab_bleaching import render_bleaching_tab
from app.tab_bleaching_history import render_bleaching_history_tab
from app.tab_variable_bleaching import render_variable_bleaching_tab
from app.tab_variable_bleaching_history import render_variable_bleaching_history_tab
from app.tab_fitting import render_fitting_tab
from app.tab_profile_likelihood import render_profile_likelihood_tab
from app.tab_bode import render_bode_tab
from app.tab_synthetic_expression import render_synthetic_expression_tab
from app.tab_kalman import render_kalman_tab
from app.tab_kalman_history import render_kalman_history_tab
from app.tab_multistart_history import render_multistart_history_tab
from app.tab_profile_history import render_profile_history_tab

APP_TITLE = "Fluorescent Protein Maturation Delay Model"

st.set_page_config(page_title=APP_TITLE, layout="wide")


def _page(render_fn):
    """Wrap a no-arg tab render function into an `st.navigation` page callable.

    Each page is its own script rerun under `st.navigation`, so the page
    title (shown once above every tab under the old single-page layout) has
    to be set again on every page. Streamlit infers a page's URL pathname
    from the wrapper function's `__name__` when `url_path` isn't given
    explicitly to `st.Page` -- since every page here is wrapped by this same
    function, they'd otherwise all collide on the pathname `_run`, so every
    `st.Page(...)` call below passes its own `url_path` explicitly.
    """
    def _run():
        st.title(APP_TITLE)
        render_fn()
    return _run


def _simulation_page():
    st.title(APP_TITLE)
    st.session_state["sim_state"] = render_simulation_tab()


def _data_page():
    st.title(APP_TITLE)
    sim_state = st.session_state.get("sim_state")
    if sim_state is None:
        st.info(
            "Visit the **Simulation** page first (in **Parameter "
            "Identification**) to set the parameters used as this page's "
            "defaults."
        )
    else:
        render_data_tab(sim_state)


# ---------------------------------------------------------
# Pages
# ---------------------------------------------------------

sim_page = st.Page(_simulation_page, title="Simulation", url_path="simulation", default=True)
data_page = st.Page(_data_page, title="Data", url_path="data")
bleach_page = st.Page(
    _page(render_bleaching_tab), title="Bleaching Only Simulation", url_path="bleaching",
)
bleach_history_page = st.Page(
    _page(render_bleaching_history_tab), title="Bleaching Fit History", url_path="bleaching-history",
)
variable_bleach_page = st.Page(
    _page(render_variable_bleaching_tab), title="Variable Bleaching", url_path="variable-bleaching",
)
variable_bleach_history_page = st.Page(
    _page(render_variable_bleaching_history_tab), title="Variable Bleaching History",
    url_path="variable-bleaching-history",
)
fit_page = st.Page(_page(render_fitting_tab), title="Least Squares Fitting", url_path="fitting")
profile_page = st.Page(
    _page(render_profile_likelihood_tab), title="Profile Likelihood", url_path="profile-likelihood",
)
bode_page = st.Page(_page(render_bode_tab), title="Bode Plot", url_path="bode")
ms_history_page = st.Page(
    _page(render_multistart_history_tab), title="Multi-Start History", url_path="multi-start-history",
)
pl_history_page = st.Page(
    _page(render_profile_history_tab), title="Profile Likelihood History",
    url_path="profile-likelihood-history",
)

synthetic_expression_page = st.Page(
    _page(render_synthetic_expression_tab), title="Synthetic Gene Expression", url_path="synthetic-expression",
)
kalman_page = st.Page(_page(render_kalman_tab), title="Kalman Filter", url_path="kalman-filter")
kalman_history_page = st.Page(
    _page(render_kalman_history_tab), title="Kalman Filter History", url_path="kalman-filter-history",
)


nav = st.navigation({
    "Parameter Identification": [
        sim_page, data_page, bleach_page, bleach_history_page,
        variable_bleach_page, variable_bleach_history_page,
        fit_page, profile_page, bode_page,
        ms_history_page, pl_history_page,
    ],
    "Kalman Filter": [synthetic_expression_page, kalman_page, kalman_history_page],
})
nav.run()
