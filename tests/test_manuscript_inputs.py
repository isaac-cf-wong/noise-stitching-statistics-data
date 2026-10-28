"""The manuscript's number formatting and table builders."""

from __future__ import annotations

import pytest

from noise_stitching_data.manuscript_inputs import (
    key_of,
    math_number,
    number_definitions,
    row_key,
    significant,
    tabular,
)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.0, "0"),
        (0.0486, "0.049"),
        (0.54, "0.54"),
        (345.0, r"3.5\times10^{2}"),
        (9.5e-4, r"9.5\times10^{-4}"),
        (1.2e-6, r"1.2\times10^{-6}"),
        (2.3e4, r"2.3\times10^{4}"),
        (9.96e-4, r"1.0\times10^{-3}"),
    ],
)
def test_significant_figures(value: float, text: str) -> None:
    """Two significant figures, scientific outside the range where a decimal reads well."""
    assert significant(value) == text


def test_keys_and_math_numbers() -> None:
    """Underscores become hyphens so a key is a safe control-sequence name; numbers are inline math."""
    assert key_of("eps", "aligo_o4high_smooth", "specified") == "eps:aligo-o4high-smooth:specified"
    assert math_number(0.054) == r"\ensuremath{0.054}"


def test_row_keys_name_the_method_and_its_configuration() -> None:
    """Each method's rows are told apart by the parameter that distinguishes them."""
    assert row_key({"name": "overlap_save_fir", "parameters": {"filter_length": 4096}}) == ("overlap_save_fir", "4096")
    assert row_key({"name": "arma_state_space", "parameters": {"ar_order": 256, "ma_order": 64}}) == (
        "arma_state_space",
        "256/64",
    )
    assert row_key({"name": "pycbc_reproducible", "parameters": {"filter_duration": 128.0}}) == (
        "pycbc_reproducible",
        "128",
    )
    assert row_key({"name": "crossfade_gwmock", "parameters": {"grid": "64s_50"}}) == ("crossfade_gwmock", "64")
    assert row_key({"name": "crossfade_gwmock", "parameters": {"grid": "64s_25"}}) == ("crossfade_gwmock", "64-25")


def test_definitions_and_tabular_are_latex() -> None:
    """The definitions use one macro, and the table has a rule under the header and around the body."""
    text = number_definitions([("a:b", "$1$")])
    assert r"\expandafter\gdef\csname rv@a:b\endcsname{$1$}" in text
    table = tabular(["l", "r"], ["x", "y"], [["1", "2"], ["3", "4"]], rules_after=[0])
    assert table.count(r"\hline") == 4
    assert table.startswith(r"\begin{tabular}{lr}")
