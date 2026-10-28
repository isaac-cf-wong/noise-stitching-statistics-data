"""Accuracy against throughput, with the stored state alongside.

Joins the exact total covariance errors of :mod:`noise_stitching_data.comparison`
with the throughput and state bytes of the cost table, method by method, and finds the
methods no other method beats on both accuracy and throughput. Nothing here measures:
it reads frozen result snapshots and joins them on the method's name and configuration.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TradeoffPoint:
    """One method configuration with its accuracy, speed and state.

    Attributes:
        name: The method.
        label: The configuration, e.g. ``L=4096`` or ``window 64 s``.
        error: The total covariance error against the target.
        samples_per_second: Single-thread throughput of the method.
        state_bytes: Serialized continuation state, or ``None`` when the method keeps none.
    """

    name: str
    label: str
    error: float
    samples_per_second: float
    state_bytes: int | None


#: The parameter that identifies a configuration of each method, in the cost table and
#: in the exact errors, and the label it is shown under.
CONFIGURATION_KEYS: dict[str, tuple[tuple[str, ...], str]] = {
    "overlap_save_fir": (("filter_length",), "L={}"),
    "ar_levinson": (("order",), "p={}"),
    "arma_state_space": (("ar_order", "ma_order"), "{}/{}"),
    "pycbc_reproducible": (("filter_duration",), "{:g} s"),
}


def configuration(row_name: str, parameters: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Return the identifying configuration of a row, or ``None`` for a crossfade row."""
    entry = CONFIGURATION_KEYS.get(row_name)
    if entry is None:
        return None
    return tuple(parameters[key] for key in entry[0])


def label_of(row_name: str, parameters: Mapping[str, Any]) -> str:
    """Return the short label of a row's configuration."""
    entry = CONFIGURATION_KEYS.get(row_name)
    if entry is None:
        grid = str(parameters.get("grid", ""))
        if not grid:
            return ""
        window, overlap = grid.split("s_", maxsplit=1)[0], grid.split("_")[1]
        return f"{window} s" if overlap == "50" else f"{window} s, {overlap}%"
    return entry[1].format(*configuration(row_name, parameters))


def window_seconds(grid_id: str) -> float:
    """Return the window length in seconds of a grid id such as ``64s_50``."""
    return float(grid_id.split("s_", maxsplit=1)[0])


def join(
    exact_rows: Iterable[Mapping[str, Any]],
    cost_rows: Iterable[Mapping[str, Any]],
    *,
    crossfade_grids: Iterable[str] | None = None,
    crossfade_throughput: Mapping[str, float] | None = None,
) -> list[TradeoffPoint]:
    """Join exact-error rows with cost-table rows.

    A row of the exact errors that has no cost row, or whose cost row did not run, is left
    out and reported by :func:`unmatched`. The crossfade's throughput depends on the window,
    so it is taken per window where ``crossfade_throughput`` has it.

    Args:
        exact_rows: Rows of ``results.exact_errors`` of one arm, edge and sampling rate.
        cost_rows: Records of ``results.costs`` of the same arm.
        crossfade_grids: The grid ids to keep for the crossfade, or all when ``None``.
        crossfade_throughput: Samples per second of the crossfade at each grid id it was
            measured at. A window with no entry takes the throughput of the cost table's
            crossfade row.

    Returns:
        One point per matched row.
    """
    costs: dict[tuple[str, tuple[Any, ...] | None], Mapping[str, Any]] = {}
    for record in cost_rows:
        if record.get("status") != "ran":
            continue
        name = record["name"]
        costs[(name, configuration(name, record["parameters"]))] = record
    keep = None if crossfade_grids is None else set(crossfade_grids)
    points = []
    for row in exact_rows:
        if "failure" in row:
            continue
        name = row["name"]
        if name == "crossfade_lal":
            continue
        if name == "crossfade_gwmock" and keep is not None and row["parameters"]["grid"] not in keep:
            continue
        cost = costs.get((name, configuration(name, row["parameters"])))
        if cost is None:
            continue
        speed = float(cost["samples_per_second"])
        if name == "crossfade_gwmock" and crossfade_throughput:
            speed = float(crossfade_throughput.get(row["parameters"]["grid"], speed))
        points.append(
            TradeoffPoint(
                name=name,
                label=label_of(name, row["parameters"]),
                error=float(row["eps_total"]),
                samples_per_second=speed,
                state_bytes=cost.get("state_bytes"),
            )
        )
    return points


def unmatched(exact_rows: Iterable[Mapping[str, Any]], points: Iterable[TradeoffPoint]) -> list[str]:
    """Return the names and labels of fitted exact rows that did not reach the join."""
    joined = {(point.name, point.label) for point in points}
    missing = []
    for row in exact_rows:
        if "failure" in row or row["name"] == "crossfade_lal":
            continue
        if (row["name"], label_of(row["name"], row["parameters"])) not in joined:
            missing.append(f"{row['name']} {label_of(row['name'], row['parameters'])}")
    return missing


def pareto_front(points: Iterable[TradeoffPoint]) -> list[TradeoffPoint]:
    """Return the points no other point beats on both error (lower) and throughput (higher).

    A point is dropped when another is at least as accurate and at least as fast and
    strictly better in one of the two. The result is ordered from the most accurate.

    Args:
        points: The points to compare.

    Returns:
        The non-dominated points.
    """
    candidates = list(points)
    front = [
        point
        for point in candidates
        if not any(
            other.error <= point.error
            and other.samples_per_second >= point.samples_per_second
            and (other.error < point.error or other.samples_per_second > point.samples_per_second)
            for other in candidates
        )
    ]
    return sorted(front, key=lambda point: point.error)


def load_snapshot(path: Path) -> dict[str, Any]:
    """Read one frozen snapshot."""
    return json.loads(path.read_text(encoding="utf-8"))
