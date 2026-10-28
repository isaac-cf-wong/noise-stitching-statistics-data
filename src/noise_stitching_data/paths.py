"""Where the scripts read the data release from and write their outputs to."""

from __future__ import annotations

import argparse
from pathlib import Path

#: The repository root, where the default data and output directories live.
ROOT = Path(__file__).resolve().parents[2]

#: The data directory ``scripts/fetch_data.py`` unpacks the archive into.
DEFAULT_DATA = ROOT / "data"

#: The directory the scripts write into: ``figures/`` and ``generated/`` under it.
DEFAULT_OUTPUT = ROOT / "output"

#: The figure style shared by every figure.
STYLE = ROOT / "matplotlib.style"


def parse(description: str) -> tuple[Path, Path]:
    """Return the data and output directories named on the command line.

    Args:
        description: What the script does, for ``--help``.

    Returns:
        The data directory, holding ``results/`` and ``psd/``, and the output directory.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="the unpacked data release (default: data/)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="where to write (default: output/)")
    arguments = parser.parse_args()
    return arguments.data, arguments.output
