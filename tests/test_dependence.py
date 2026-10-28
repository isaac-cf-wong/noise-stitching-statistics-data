"""The window-requirement helpers."""

from __future__ import annotations

import numpy as np
import pytest

from noise_stitching_data.dependence import (
    crossfade_curve,
    edge_slope,
    floor_of,
    wall_contrast,
    window_for_error,
)


def _row(name: str, grid: str, error: float) -> dict:
    return {"name": name, "parameters": {"grid": grid}, "eps_total": error}


def test_the_curve_keeps_the_gwmock_rows_at_the_requested_overlap_in_window_order() -> None:
    """Other methods, the other phase sequence, other overlaps and failed fits are left out."""
    rows = [
        _row("crossfade_gwmock", "64s_50", 1e-3),
        _row("crossfade_gwmock", "8s_50", 1e-1),
        _row("crossfade_gwmock", "64s_25", 5e-3),
        _row("crossfade_lal", "8s_50", 3e-1),
        _row("ar_levinson", "x_50", 9.0),
        {"name": "crossfade_gwmock", "parameters": {"grid": "256s_50"}, "failure": "x"},
    ]
    assert crossfade_curve(rows) == {8.0: 1e-1, 64.0: 1e-3}
    assert crossfade_curve(rows, overlap="25") == {64.0: 5e-3}


def test_the_window_for_a_threshold_interpolates_in_logarithms() -> None:
    """On a power law the interpolation is exact; outside the sweep it reports the edge or None."""
    curve = {8.0: 1e-1, 64.0: 1e-3, 512.0: 1e-5}
    assert window_for_error(curve, 1e-2) == pytest.approx(8.0 * 8.0**0.5)
    assert window_for_error(curve, 1.0) == 8.0
    assert window_for_error(curve, 1e-7) is None
    assert floor_of(curve) == 1e-5


def test_a_floor_that_never_reaches_the_threshold_is_reported_as_none() -> None:
    """A curve that stalls above the threshold has no window that reaches it."""
    assert window_for_error({8.0: 2e-2, 64.0: 1.8e-2, 512.0: 1.8e-2}, 1e-3) is None


def test_edge_slope_and_contrast_of_a_power_law() -> None:
    """A spectrum falling as f^-6 has index 6; contrast is the ratio of the floor to the band edge."""
    frequencies = np.linspace(1.0, 100.0, 4000)
    spectrum = frequencies**-6.0
    assert edge_slope(frequencies, spectrum, 10.0) == pytest.approx(6.0, rel=1e-3)
    assert wall_contrast(frequencies, spectrum, 10.0, 5.0) == pytest.approx(2.0**6, rel=1e-3)
    assert wall_contrast(frequencies, spectrum, 5.0, 5.0) == 1.0
