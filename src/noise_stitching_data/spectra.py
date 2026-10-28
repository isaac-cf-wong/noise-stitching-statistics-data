"""The analysis band of each spectrum and its tabulated power spectral density."""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: The upper edge of every analysis band, in hertz.
HIGH_FREQUENCY_HZ = 2048.0

#: The lower edge of the analysis band of each spectrum, in hertz.
LOW_FREQUENCY_HZ: dict[str, float] = {
    "aligo_o4high_smooth": 10.0,
    "aligo_o3h1_lines": 10.0,
    "aligo_aplus": 10.0,
    "aligo_aplus_fine": 10.0,
    "virgo_o4": 10.0,
    "virgo_design": 10.0,
    "virgo_design_fine": 10.0,
    "kagra_o4": 10.0,
    "kagra_design": 10.0,
    "kagra_design_fine": 10.0,
    "ce_40": 5.0,
    "ce_wideband": 5.0,
    "et_d_smooth": 5.0,
    "et10_full_cryo_lines": 5.0,
    "et15_full_cryo_lines": 5.0,
}


def analysis_band(arm: str) -> tuple[float, float]:
    """Return the analysis band of one spectrum, in hertz.

    Args:
        arm: The spectrum.

    Returns:
        The low and high band edges.

    Raises:
        ValueError: If the spectrum is not one of the release.
    """
    if arm not in LOW_FREQUENCY_HZ:
        raise ValueError(f"unknown spectrum {arm!r}; expected one of {sorted(LOW_FREQUENCY_HZ)}.")
    return (LOW_FREQUENCY_HZ[arm], HIGH_FREQUENCY_HZ)


def load_psd(data: Path, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the tabulated power spectral density of one spectrum, in increasing frequency.

    Args:
        data: The unpacked data release.
        arm: The spectrum.

    Returns:
        The frequencies in hertz and the one-sided PSD in 1/Hz.
    """
    table = np.loadtxt(data / "psd" / f"{arm}.txt", dtype=float)
    order = np.argsort(table[:, 0])
    return table[order, 0], table[order, 1]
