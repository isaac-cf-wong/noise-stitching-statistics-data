"""The tables written from the fetched data are the paper's, byte for byte.

Skipped until the data has been fetched with ``scripts/fetch_data.py``.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED = ROOT / "tests" / "expected" / "generated.sha256"


@pytest.mark.skipif(not (DATA / "results").is_dir(), reason="the data has not been fetched")
def test_the_tables_are_the_published_ones(tmp_path: Path) -> None:
    """Every generated table and the numbers file match the digests of the published ones."""
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_tables.py"), "--data", str(DATA), "--output", str(tmp_path)],
        check=True,
    )
    expected = dict(reversed(line.split()) for line in EXPECTED.read_text().splitlines())
    written = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (tmp_path / "generated").iterdir()}
    assert written == expected
