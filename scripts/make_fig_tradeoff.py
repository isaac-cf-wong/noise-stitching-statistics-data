#!/usr/bin/env python3
"""Draw the accuracy-throughput trade-off and write the table behind it.

Reads the exact errors under ``<data>/results/exact/``, the cost tables under
``<data>/results/costs/`` and the stored-state measurement in
``<data>/results/reproducibility.json``, joins them per spectrum
arm and edge convention at 4096 Hz, and plots the total covariance error against
single-thread throughput. Marker area follows the stored state. The methods no other method
beats on both axes are joined by a line.

Usage::

    uv run python scripts/make_fig_tradeoff.py [--data DIR] [--output DIR]

The figures go to ``<output>/figures/`` and the table to ``<output>/generated/tradeoff-points.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from noise_stitching_data.paths import STYLE, parse
from noise_stitching_data.tradeoff import (
    TradeoffPoint,
    join,
    label_of,
    load_snapshot,
    pareto_front,
    unmatched,
)

ARMS = {
    "aligo_o4high_smooth": "aLIGO smooth",
    "aligo_o3h1_lines": "aLIGO with lines",
    "et_d_smooth": "ET smooth",
    "et10_full_cryo_lines": "ET with lines",
}
EDGES = ("specified", "hard")
SAMPLING_FREQUENCY = 4096

#: The segment index the stored state is quoted at.
STATE_SEGMENT = 100
FAMILY_STYLE = {
    "crossfade_gwmock": ("Crossfade", "tab:blue", "o"),
    "overlap_save_fir": ("Overlap-save FIR", "tab:orange", "s"),
    "ar_levinson": ("AR (Levinson)", "tab:green", "^"),
    "arma_state_space": ("ARMA", "tab:red", "v"),
    "pycbc_reproducible": (r"\texttt{PyCBC} filter", "tab:purple", "D"),
}
PDF_METADATA = {"Creator": "scripts/make_fig_tradeoff.py", "Producer": "matplotlib", "CreationDate": None}


def stored_state_bytes(results: Path) -> dict[tuple[str, str], int]:
    """Return the stored-state bytes of each method configuration, from the n = 100 segment.

    The key is ``(method, label)``; the crossfade, whose state does not depend on the window,
    is stored under every window label.
    """
    state: dict[tuple[str, str], int] = {}
    records = load_snapshot(results / "reproducibility.json")["results"]["reproducibility"]
    for record in records:
        if record["arm"] != "aligo_o4high_smooth" or record["segment_index"] != STATE_SEGMENT:
            continue
        name = record["method"]
        if "stored_state_bytes" not in record:
            continue
        state[(name, label_of(name, record["parameters"]))] = int(record["stored_state_bytes"])
    return state


def points_for(
    results: Path, arm: str, edge: str, state: dict[tuple[str, str], int]
) -> tuple[list[TradeoffPoint], list[str]]:
    """Return the joined points of one arm and edge, with the stored state filled in, and the unmatched rows."""
    exact = load_snapshot(results / "exact" / f"{arm}__{edge}__{SAMPLING_FREQUENCY}.json")["results"]["exact_errors"]
    costs = load_snapshot(results / "costs" / f"{arm}__64s_50.json")["results"]["costs"]
    window_costs = {}
    for path in sorted((results / "costs").glob(f"{arm}__*s_*__crossfade_gwmock.json")):
        grid = path.name.split("__")[1]
        record = next(r for r in load_snapshot(path)["results"]["costs"] if r["name"] == "crossfade_gwmock")
        window_costs[grid] = float(record["samples_per_second"])
    points = join(exact, costs, crossfade_throughput=window_costs)
    crossfade_state = state.get(("crossfade_gwmock", ""))
    filled = []
    for point in points:
        stored = state.get((point.name, point.label))
        if point.name == "crossfade_gwmock":
            stored = crossfade_state
        filled.append(TradeoffPoint(point.name, point.label, point.error, point.samples_per_second, stored))
    return filled, unmatched(exact, points)


def draw(results: Path, edge: str, state: dict[tuple[str, str], int], output: Path) -> list[dict[str, object]]:
    """Draw one edge convention's four panels and return its table rows."""
    plt.style.use(STYLE)
    figure, axes = plt.subplots(2, 2, figsize=(15.0 / 2.54, 12.0 / 2.54), sharex=True)
    rows: list[dict[str, object]] = []
    for axis, (arm, title) in zip(axes.flat, ARMS.items(), strict=True):
        points, missing = points_for(results, arm, edge, state)
        front = {(p.name, p.label) for p in pareto_front(points)}
        for name, (label, colour, marker) in FAMILY_STYLE.items():
            family = [p for p in points if p.name == name]
            if not family:
                continue
            sizes = [12.0 + 6.0 * math.log10(max(p.state_bytes or 1, 1)) for p in family]
            axis.scatter(
                [p.samples_per_second / 1e6 for p in family],
                [p.error for p in family],
                s=sizes,
                c=colour,
                marker=marker,
                label=label,
                linewidths=0.4,
                edgecolors="black",
                zorder=3,
            )
        on_front = sorted((p for p in points if (p.name, p.label) in front), key=lambda p: p.samples_per_second)
        axis.plot(
            [p.samples_per_second / 1e6 for p in on_front], [p.error for p in on_front], color="0.35", lw=0.7, zorder=2
        )
        for point in points:
            rows.append(
                {
                    "arm": arm,
                    "edge": edge,
                    "method": point.name,
                    "configuration": point.label,
                    "error": point.error,
                    "samples_per_second": point.samples_per_second,
                    "stored_state_bytes": point.state_bytes,
                    "on_front": (point.name, point.label) in front,
                }
            )
        rows.extend({"arm": arm, "edge": edge, "unmatched": text} for text in missing)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(True, which="major", lw=0.3, alpha=0.5)
    for axis in axes[1]:
        axis.set_xlabel("Throughput (M samples/s, one thread)")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Total covariance error $\epsilon_{\rm tot}$")
    axes[0, 0].legend(loc="upper right", frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, metadata=PDF_METADATA)
    plt.close(figure)
    return rows


def main() -> None:
    """Draw both edge conventions and write the table."""
    data, output = parse(__doc__.splitlines()[0])
    results = data / "results"
    figure_dir = output / "figures"
    table = output / "generated" / "tradeoff-points.json"
    state = stored_state_bytes(results)
    rows: list[dict[str, object]] = []
    for edge in EDGES:
        rows += draw(results, edge, state, figure_dir / f"fig_tradeoff_{edge}.pdf")
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
