r"""The numbers and tables of the manuscript, generated from the frozen results.

Every measured number in the paper is read from a result snapshot and printed here, so it
carries the run and the commit that produced it and cannot drift from them. The generated
file defines ``\\rv{key}`` for each number and one tabular per table; the prose quotes
``\\rv{...}`` and ``\\input``s the tables.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: Arms of the comparison's main text, in column order, and their short names.
MAIN_ARMS: tuple[tuple[str, str], ...] = (
    ("aligo_o4high_smooth", "aLIGO smooth"),
    ("aligo_o3h1_lines", "aLIGO lines"),
    ("et_d_smooth", "ET smooth"),
    ("et10_full_cryo_lines", "ET lines"),
)

#: Decimal exponents of the values printed as plain decimals: from ``[0.01, 100)``.
DECIMAL_EXPONENTS = (-2, 2)

#: Window lengths shown in the tables, in seconds.
TABLE_WINDOWS: tuple[float, ...] = (8.0, 16.0, 64.0, 256.0, 1024.0, 4096.0)


def key_of(*parts: object) -> str:
    """Return a macro key from its parts: lower case, underscores replaced by hyphens."""
    return ":".join(str(part).replace("_", "-") for part in parts)


def significant(value: float, digits: int = 2) -> str:
    """Return ``value`` to ``digits`` significant figures, in scientific notation outside ``[0.01, 1000)``."""
    if value == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if DECIMAL_EXPONENTS[0] <= exponent < DECIMAL_EXPONENTS[1]:
        decimals = max(digits - 1 - exponent, 0)
        return f"{value:.{decimals}f}"
    mantissa = value / 10.0**exponent
    text = f"{mantissa:.{digits - 1}f}"
    if text.startswith("10"):
        text, exponent = f"{1.0:.{digits - 1}f}", exponent + 1
    return rf"{text}\times10^{{{exponent}}}"


def math_number(value: float, digits: int = 2) -> str:
    r"""Return ``value`` as math that prints in text and in math alike, through ``\ensuremath``."""
    return rf"\ensuremath{{{significant(value, digits)}}}"


def load(path: Path) -> dict[str, Any]:
    """Read one snapshot."""
    return json.loads(path.read_text(encoding="utf-8"))


def row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return a stable ``(method, configuration)`` key of an exact-error, cost or reproducibility record.

    The crossfade's configuration is its window in seconds, followed by the overlap when that
    is not fifty per cent; a record with no grid, such as a cost row, has an empty configuration.
    """
    parameters = row["parameters"]
    name = row.get("name", row.get("method"))
    if name == "overlap_save_fir":
        return name, str(parameters["filter_length"])
    if name == "ar_levinson":
        return name, str(parameters["order"])
    if name == "arma_state_space":
        return name, f"{parameters['ar_order']}/{parameters['ma_order']}"
    if name == "pycbc_reproducible":
        return name, f"{parameters['filter_duration']:g}"
    grid = parameters.get("grid")
    if grid is None:
        return name, ""
    window, overlap = grid.split("s_")[0], grid.split("_")[1]
    return name, window if overlap == "50" else f"{window}-{overlap}"


def errors_by_row(snapshot: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Return the exact-error rows of a snapshot by :func:`row_key`; the gwmock crossfade only."""
    rows = {}
    for row in snapshot["results"]["exact_errors"]:
        if row["name"] == "crossfade_lal":
            continue
        rows[row_key(row)] = row
    return rows


def number_definitions(entries: Iterable[tuple[str, str]]) -> str:
    r"""Return the LaTeX that defines ``\\rv`` for every ``(key, text)`` entry."""
    lines = [r"\providecommand{\rv}[1]{\csname rv@#1\endcsname}"]
    lines += [rf"\expandafter\gdef\csname rv@{key}\endcsname{{{text}}}" for key, text in entries]
    return "\n".join(lines) + "\n"


def tabular(
    columns: Sequence[str], header: Sequence[str], rows: Sequence[Sequence[str]], *, rules_after: Sequence[int] = ()
) -> str:
    """Return a booktabs-free tabular as LaTeX text.

    Args:
        columns: One column specifier per column, e.g. ``"l"`` or ``"r"``.
        header: The header cells.
        rows: The body rows, each a sequence of cells.
        rules_after: Indices of body rows after which a horizontal rule is drawn.

    Returns:
        The tabular environment.
    """
    lines = [r"\begin{tabular}{" + "".join(columns) + "}", r"\hline", " & ".join(header) + r" \\", r"\hline"]
    for index, row in enumerate(rows):
        lines.append(" & ".join(row) + r" \\")
        if index in rules_after:
            lines.append(r"\hline")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"
