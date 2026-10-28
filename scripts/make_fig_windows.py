#!/usr/bin/env python3
"""Draw the crossfade's total covariance error against window length for every sensitivity curve.

Reads ``<data>/results/exact/<arm>__specified__4096.json`` of the arms in the window table and plots
the error against the window, one curve per arm, with the threshold lines the recommendation
uses.

Usage::

    uv run python scripts/make_fig_windows.py [--data DIR] [--output DIR]
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from noise_stitching_data.dependence import crossfade_curve
from noise_stitching_data.manuscript_inputs import load
from noise_stitching_data.paths import STYLE, parse

FIGURE_WIDTH = 15.0 / 2.54
FIGURE_HEIGHT = 9.0 / 2.54

#: Arm, label, colour and line style: LIGO and LIGO-India in blue, Virgo in green, KAGRA in orange,
#: Cosmic Explorer in red, Einstein Telescope in purple. Marker shapes distinguish
#: curves within a detector family even when short legend lines hide the dash pattern.
ARMS = (
    ("aligo_o4high_smooth", "aLIGO O4", "tab:blue", (0, (3, 2)), "o"),
    ("aligo_o3h1_lines", "aLIGO O3 H1 (lines)", "tab:blue", (0, (1, 1.5)), "s"),
    ("aligo_aplus", "A+ / LIGO-India", "tab:blue", "-", "^"),
    ("virgo_o4", "Virgo O4", "tab:green", (0, (3, 2)), "o"),
    ("virgo_design", "Virgo design", "tab:green", "-", "^"),
    ("kagra_o4", "KAGRA observing (25 Mpc)", "tab:orange", (0, (3, 2)), "o"),
    ("kagra_design", "KAGRA design", "tab:orange", "-", "^"),
    ("ce_40", "CE", "tab:red", "-", "o"),
    ("ce_wideband", "CE wideband", "tab:red", (0, (3, 2)), "s"),
    ("et_d_smooth", "ET-D", "tab:purple", "-", "o"),
    ("et10_full_cryo_lines", "ET-10 (lines)", "tab:purple", (0, (1, 1.5)), "s"),
)
THRESHOLDS = (1e-2, 1e-3, 1e-4)
PDF_METADATA = {"Creator": "scripts/make_fig_windows.py", "Producer": "matplotlib", "CreationDate": None}


def main() -> None:
    """Draw the figure."""
    data, output_dir = parse(__doc__.splitlines()[0])
    output = output_dir / "figures" / "fig_windows.pdf"
    plt.style.use(STYLE)
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    for arm, label, colour, style, marker in ARMS:
        snapshot = load(data / "results" / "exact" / f"{arm}__specified__4096.json")
        curve = crossfade_curve(snapshot["results"]["exact_errors"])
        axis.plot(
            list(curve),
            list(curve.values()),
            linestyle=style,
            color=colour,
            marker=marker,
            markersize=3.0,
            label=label,
        )
    for threshold in THRESHOLDS:
        axis.axhline(threshold, color="0.6", lw=0.5, zorder=0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Window length (s)")
    axis.set_ylabel(r"Total covariance error $\epsilon_{\rm tot}$")
    axis.legend(ncol=2, frameon=False, loc="lower left", handlelength=4.0, numpoints=2)
    axis.grid(True, which="major", lw=0.3, alpha=0.5)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata=PDF_METADATA)


if __name__ == "__main__":
    main()
