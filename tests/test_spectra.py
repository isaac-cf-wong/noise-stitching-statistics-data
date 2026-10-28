"""The analysis bands and the spectrum tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from noise_stitching_data.spectra import analysis_band, load_psd


def test_bands_start_at_ten_hertz_except_for_the_third_generation_curves() -> None:
    """The LIGO, Virgo and KAGRA curves start at 10 Hz, Cosmic Explorer and ET at 5 Hz."""
    assert analysis_band("aligo_o4high_smooth") == (10.0, 2048.0)
    assert analysis_band("kagra_design_fine") == (10.0, 2048.0)
    assert analysis_band("ce_wideband") == (5.0, 2048.0)
    assert analysis_band("et10_full_cryo_lines") == (5.0, 2048.0)


def test_an_unknown_spectrum_is_refused() -> None:
    """A spectrum outside the release has no band rather than a default one."""
    with pytest.raises(ValueError, match="unknown spectrum"):
        analysis_band("aligo_o4low")


def test_a_table_is_read_in_increasing_frequency(tmp_path: Path) -> None:
    """The table's header is skipped and its rows are sorted by frequency."""
    (tmp_path / "psd").mkdir()
    (tmp_path / "psd" / "x.txt").write_text("# header\n20 2e-46\n10 1e-46\n")
    frequencies, values = load_psd(tmp_path, "x")
    assert frequencies.tolist() == [10.0, 20.0]
    np.testing.assert_array_equal(values, [1e-46, 2e-46])
