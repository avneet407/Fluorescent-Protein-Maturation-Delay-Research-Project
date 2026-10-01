"""Global matplotlib styling for every figure the Streamlit UI draws.

Figures use Lato for all text, including mathtext (u(t), I(t), k_m, ...),
from the font files bundled in `static/fonts/`. rcParams are
process-global, so importing this module once from `streamlit_app.py`
(before any figure is built) also covers figures built in the
non-Streamlit helper modules, e.g. `Parameter_Identification/multi_start_plots.py`.
(The page text outside the figures uses Latin Modern, set in
`.streamlit/config.toml`.)
"""

import hashlib
import io
import re
from pathlib import Path

import matplotlib
import streamlit as st
from matplotlib import font_manager

# Resolution every figure is rendered at in the UI (and saved at by default).
FIGURE_DPI = 400

# Lato isn't a system font on most machines, so register the bundled copy
# (without this, matplotlib silently falls back to its default font).
FONT_DIR = Path(__file__).resolve().parent.parent / "static" / "fonts"
for _font_path in FONT_DIR.glob("Lato-*.ttf"):
    font_manager.fontManager.addfont(str(_font_path))

matplotlib.rcParams.update({
    "font.family": "Lato",
    "mathtext.fontset": "custom",       # makes u(t), I(t), k_m etc. use Lato too
    "mathtext.rm": "Lato",
    "mathtext.it": "Lato:italic",
    "mathtext.bf": "Lato:bold",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "figure.titlesize": 20,
    "savefig.dpi": FIGURE_DPI,
})


def _figure_file_name(fig):
    """File name for a figure's download, taken from its first title."""
    title = fig._suptitle.get_text() if fig._suptitle else ""
    for ax in fig.axes:
        if title:
            break
        title = ax.get_title()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return f"{slug or 'figure'}_{FIGURE_DPI}dpi.png"


def show_figure(fig):
    """Display `fig` in the page, with a button to download it at FIGURE_DPI.

    Used instead of `st.pyplot`, which always renders at a fixed 200 dpi.
    `st.image` downsizes anything wider than ~1460 px before sending it to
    the browser, so right-click -> Save image gives the shrunk copy; the
    download button serves the full-resolution PNG instead.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIGURE_DPI, bbox_inches="tight")
    png = buf.getvalue()
    st.image(png, width="stretch", output_format="PNG")
    # download_button's widget ID ignores `data`, so key on the image bytes
    # to keep several figures on one page from colliding.
    st.download_button(
        f"Download PNG ({FIGURE_DPI} dpi)",
        data=png,
        file_name=_figure_file_name(fig),
        mime="image/png",
        key=f"download_{hashlib.md5(png).hexdigest()}",
        on_click="ignore",
    )
