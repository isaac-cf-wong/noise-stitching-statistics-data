"""How the crossfade's accuracy depends on the window and on the sensitivity curve.

The crossfade's total covariance error falls with the window until it meets a floor. The
window a mock data challenge needs therefore depends on the spectrum: on how steeply it
falls from the low-frequency wall into the band, and on where the band starts. This module
reads the exact errors of one arm over the window sweep and answers two questions: the
shortest window that brings the error below a threshold, and the floor the error reaches.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np


def crossfade_curve(rows: Sequence[Mapping[str, Any]], *, overlap: str = "50") -> dict[float, float]:
    """Return the crossfade's total error against window length, in seconds.

    Args:
        rows: The ``exact_errors`` rows of one arm, edge and sampling rate.
        overlap: The overlap percentage of the grid points to keep.

    Returns:
        A mapping from window length to ``eps_tot`` for the gwmock phase sequence, ascending.
    """
    curve = {}
    for row in rows:
        if row["name"] != "crossfade_gwmock" or "failure" in row:
            continue
        grid = row["parameters"]["grid"]
        if grid.split("_")[1] != overlap:
            continue
        curve[float(grid.split("s_")[0])] = float(row["eps_total"])
    return dict(sorted(curve.items()))


def window_for_error(curve: Mapping[float, float], threshold: float) -> float | None:
    """Return the shortest window at which the error falls to ``threshold``.

    The error is interpolated linearly in the logarithms of window and error between the two
    window lengths that bracket the threshold.

    Args:
        curve: Window length in seconds to error, as from :func:`crossfade_curve`.
        threshold: The error to reach.

    Returns:
        The window length in seconds; the shortest window of the sweep if the error is already
        below the threshold there; ``None`` if the error stays above it over the whole sweep.
    """
    windows = sorted(curve)
    if curve[windows[0]] <= threshold:
        return windows[0]
    for short, long in pairwise(windows):
        high, low = curve[short], curve[long]
        if high > threshold >= low:
            fraction = (math.log(threshold) - math.log(high)) / (math.log(low) - math.log(high))
            return math.exp(math.log(short) + fraction * (math.log(long) - math.log(short)))
    return None


def floor_of(curve: Mapping[float, float]) -> float:
    """Return the error at the longest window of the sweep, the floor the curve has reached."""
    return curve[max(curve)]


def edge_slope(frequencies: np.ndarray, spectrum: np.ndarray, low_frequency: float) -> float:
    """Return how steeply a spectrum falls above the band edge, as a power-law index.

    The index is ``-d ln S / d ln f`` between ``low_frequency`` and ``1.5 * low_frequency``.

    Args:
        frequencies: The tabulated frequencies, ascending.
        spectrum: The tabulated spectrum.
        low_frequency: The lower edge of the analysis band.

    Returns:
        The positive index of a falling spectrum.
    """
    upper = 1.5 * low_frequency
    ratio = np.interp(upper, frequencies, spectrum) / np.interp(low_frequency, frequencies, spectrum)
    return float(-math.log(ratio) / math.log(1.5))


def wall_contrast(frequencies: np.ndarray, spectrum: np.ndarray, low_frequency: float, floor_frequency: float) -> float:
    """Return the spectrum at the target's floor frequency over its value at the band edge.

    Below ``floor_frequency`` the specified-edge target is flat, so this is the height of the
    power the target keeps just under the band relative to the band's first bin; it is one
    when the band edge lies at or below the floor frequency.

    Args:
        frequencies: The tabulated frequencies, ascending.
        spectrum: The tabulated spectrum.
        low_frequency: The lower edge of the analysis band.
        floor_frequency: Where the specified target stops rising and is held flat.

    Returns:
        The ratio ``S(floor) / S(low)``, at least one.
    """
    if low_frequency <= floor_frequency:
        return 1.0
    return float(np.interp(floor_frequency, frequencies, spectrum) / np.interp(low_frequency, frequencies, spectrum))
