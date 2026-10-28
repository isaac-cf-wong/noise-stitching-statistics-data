#!/usr/bin/env python3
"""Draw the paper's closed-form verification figure from the archived residuals.

Reads ``<data>/results/verification.json``: the standardized residuals of the measured
crossfade covariance against its closed form, over lag and block phase, for the
generator's blend phases and for an alternative weight pair. Writes
``<output>/figures/fig_c1_residuals.pdf`` and prints every number the text quotes from the
verification: each arm's summary, including the quarter-statistics arm the figure does not draw.

Usage::

    uv run python scripts/make_fig_c1_residuals.py [--data DIR] [--output DIR]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from noise_stitching_data.paths import STYLE, parse


@dataclass(frozen=True)
class Comparison:
    """One arm of the verification: its residuals on the phase-by-lag grid.

    Attributes:
        phases: The block phases compared.
        lags: The lags compared.
        residuals: The residuals, one per estimable cell.
        residual_grid: The same residuals on the grid, ``nan`` where a cell was not estimable.
        critical_value: The critical value the largest residual is compared with.
    """

    phases: np.ndarray
    lags: np.ndarray
    residuals: np.ndarray
    residual_grid: np.ndarray
    critical_value: float

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Comparison:
        """Return the comparison stored in one arm of the snapshot."""
        grid = [[math.nan if z is None else z for z in row] for row in record["residual_grid"]]
        return cls(
            phases=np.asarray(record["phases"]),
            lags=np.asarray(record["lags"]),
            residuals=np.asarray(record["residuals"], dtype=float),
            residual_grid=np.asarray(grid, dtype=float),
            critical_value=float(record["critical_value"]),
        )


@dataclass(frozen=True)
class Campaign:
    """The two arms the figure draws, and the lag beyond which the closed form is zero."""

    headline: Comparison
    alternative: Comparison
    support: int


def load_campaign(data: Path) -> Campaign:
    """Return the archived verification residuals."""
    results = json.loads((data / "results" / "verification.json").read_text(encoding="utf-8"))["results"]
    return Campaign(
        headline=Comparison.from_record(results["headline"]),
        alternative=Comparison.from_record(results["alternative"]),
        support=int(results["support_half_width"]),
    )


#: Figure width in inches: 15 cm, the journal's full-width figure.
FIGURE_WIDTH = 15.0 / 2.54

#: Figure height in inches, enough for two residual maps and a histogram at a
#: font size that stays legible at the final size.
FIGURE_HEIGHT = 13.0 / 2.54

#: Residual size at which the measured tail fraction is compared with the
#: standard normal's, as a cheap check against a distribution this is not.
TAIL_THRESHOLD = 2.0

#: PDF metadata. `CreationDate` is dropped rather than written, because a
#: timestamp in the file would make an identical figure a different file and the
#: determinism check below meaningless.
PDF_METADATA = {
    "Title": "Closed-form verification: standardized residuals over lag and window phase",
    "Creator": "scripts/make_fig_c1_residuals.py",
    "Producer": "matplotlib",
    "CreationDate": None,
}


def _draw_residual_map(axes: plt.Axes, result: Comparison, support: int, title: str) -> object:
    """Draw one arm's residuals over the lag-by-phase grid.

    Args:
        axes: The axes to draw on.
        result: The arm's comparison.
        support: The lag at and beyond which the closed form is exactly zero.
        title: The panel's tag.

    Returns:
        The image, so that a colour bar can be attached to it.
    """
    lags = result.lags
    # The colour scale ends exactly at the critical value, so a saturated cell
    # is a residual beyond it. None is, and that is readable from the figure
    # without reading a number off it.
    limit = result.critical_value
    image = axes.imshow(
        result.residual_grid,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        extent=(float(lags[0]) - 0.5, float(lags[-1]) + 0.5, -0.5, float(result.phases.size) - 0.5),
    )
    for edge in (-support, support):
        axes.axvline(edge, color="black", linestyle=(0, (4, 2)), linewidth=0.7)
    axes.set_yticks(np.arange(result.phases.size))
    axes.set_yticklabels([str(int(phase)) for phase in result.phases])
    axes.set_ylabel("Block phase $p$")
    axes.set_title(title, loc="left")

    # The extreme cell is marked as well as quoted: the largest residual is the
    # quantity compared with the critical value, and a reader should be able to see
    # where on the grid it sits rather than take its size on trust.
    flat = int(np.nanargmax(np.abs(result.residual_grid)))
    phase_index, lag_index = np.unravel_index(flat, result.residual_grid.shape)
    axes.plot(
        lags[lag_index],
        phase_index,
        marker="o",
        markersize=4.0,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=0.8,
    )
    return image


def _draw_histogram(axes: plt.Axes, campaign: Campaign) -> None:
    """Draw the residual distributions against a standard normal.

    Args:
        axes: The axes to draw on.
        campaign: The campaign whose arms are drawn.
    """
    edges = np.linspace(-5.0, 5.0, 61)
    styles = (("solid", "Generator blend phases"), ((0, (4, 2)), r"\texttt{LALSimulation} blend phases"))
    for (linestyle, label), result in zip(styles, (campaign.headline, campaign.alternative), strict=True):
        axes.hist(
            result.residuals,
            bins=edges,
            density=True,
            histtype="step",
            linestyle=linestyle,
            color="black",
            label=label,
        )
    centres = 0.5 * (edges[:-1] + edges[1:])
    axes.plot(
        centres,
        np.exp(-0.5 * centres**2) / np.sqrt(2.0 * np.pi),
        color="0.45",
        linestyle=(0, (1, 1.5)),
        label="Standard normal",
    )
    for sign in (-1.0, 1.0):
        axes.axvline(sign * campaign.headline.critical_value, color="0.45", linestyle=(0, (5, 2, 1, 2)), linewidth=0.7)
    axes.set_xlim(edges[0], edges[-1])
    axes.set_xlabel("Standardized residual $z$")
    axes.set_ylabel("Density")
    axes.set_title("(c)", loc="left")
    axes.legend(frameon=False, loc="upper left")


def draw(campaign: Campaign) -> plt.Figure:
    """Return the figure for a campaign.

    Args:
        campaign: The campaign to draw.

    Returns:
        The figure.
    """
    support = campaign.support
    with plt.style.context(STYLE):
        figure, (top, middle, bottom) = plt.subplots(
            3, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), height_ratios=(1.0, 1.0, 1.35), layout="constrained"
        )
        image = _draw_residual_map(top, campaign.headline, support, "(a)")
        _draw_residual_map(middle, campaign.alternative, support, "(b)")
        middle.set_xlabel(r"Lag $\tau$ (samples)")
        top.set_xticklabels([])
        bar = figure.colorbar(image, ax=(top, middle), pad=0.015, aspect=30)
        bar.set_label("Standardized residual $z$")
        limit = campaign.headline.critical_value
        bar.set_ticks([-limit, -2.0, 0.0, 2.0, limit])
        bar.set_ticklabels([f"$-{limit:.2f}$", "$-2$", "$0$", "$2$", f"${limit:.2f}$"])
        _draw_histogram(bottom, campaign)
    return figure


#: The summary fields of each arm that the text quotes.
SUMMARY_FIELDS = (
    "n_realizations",
    "n_blocks",
    "n_cells",
    "max_abs_residual",
    "critical_value",
    "mean_residual",
    "std_residual",
    "beyond_support_max_abs_residual",
    "max_deviation",
    "median_standard_error",
    "admissible_scale",
    "ks_distance",
)


def report(data: Path, campaign: Campaign) -> None:
    """Print each arm's summary and the share of residuals beyond the tail threshold.

    Args:
        data: The unpacked data release.
        campaign: The campaign that was drawn.
    """
    results = json.loads((data / "results" / "verification.json").read_text(encoding="utf-8"))["results"]
    for arm in ("headline", "quarter", "alternative"):
        print(f"{arm}: {results[arm]['label']}")
        for name in SUMMARY_FIELDS:
            print(f"  {name}: {results[arm][name]}")
    fraction_outside = float(np.mean(np.abs(campaign.headline.residuals) > TAIL_THRESHOLD))
    expected = math.erfc(TAIL_THRESHOLD / math.sqrt(2.0))
    print(f"cells beyond |z| = {TAIL_THRESHOLD:g}: measured {fraction_outside:.4f}, standard normal {expected:.4f}")


def main() -> None:
    """Draw the figure and print its numbers."""
    data, output_dir = parse(__doc__.splitlines()[0])
    output = output_dir / "figures" / "fig_c1_residuals.pdf"
    campaign = load_campaign(data)
    with plt.style.context(STYLE):
        figure = draw(campaign)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="pdf", metadata=PDF_METADATA)
    plt.close(figure)
    report(data, campaign)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
