"""Plotting helpers for multi-start least-squares fit results.

Each function builds and returns a matplotlib Figure from a multi-start
results DataFrame (as produced by `multi_start_fit.run_multi_start`, with
derived quantities already added as extra columns); the caller is
responsible for displaying it (e.g. via `st.pyplot`).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_histograms(df, names, true_values, color="tab:blue", figsize_per_panel=3.6):
    """Marginal histograms of `names` (one subplot per name), across runs in `df`.

    `true_values` is a dict of {name: value}; when a name is present, a red
    dashed vertical line marks that true/ground-truth value on its subplot.
    """
    fig, axes = plt.subplots(1, len(names), figsize=(figsize_per_panel * len(names), 3.5))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        values = df[name].to_numpy(dtype=float)
        data_min, data_max = float(np.min(values)), float(np.max(values))
        if name in true_values:
            data_min = min(data_min, true_values[name])
            data_max = max(data_max, true_values[name])
        span = data_max - data_min
        pad = span * 0.08 if span > 0 else max(abs(data_max), 1.0) * 0.05
        hist_range = (data_min - pad, data_max + pad)

        ax.hist(values, bins=15, range=hist_range, color=color, alpha=0.75)
        if name in true_values:
            ax.axvline(
                true_values[name], color="tab:red", linestyle="--",
                linewidth=2, label="True value",
            )
            ax.legend(fontsize=7)
        ax.set_xlim(hist_range)
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
    fig.tight_layout()
    return fig
