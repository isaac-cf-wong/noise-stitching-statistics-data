#!/usr/bin/env python3
"""Download the data release from Zenodo, check it, and unpack it.

The record's files are downloaded into ``<data>/download/``, each checked against the
MD5 checksum Zenodo publishes for it, and ``results.tar.gz`` is unpacked into ``<data>``,
which then holds ``results/`` and ``psd/``. A file already downloaded with the right
checksum is not downloaded again.

Usage::

    uv run python scripts/fetch_data.py [--data DIR] [--record ID]
    uv run python scripts/fetch_data.py --archive results.tar.gz   # unpack a local copy

Only the standard library is used, so the script runs before the environment is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

#: The Zenodo record of the data release.
RECORD_ID = "23243286"

#: The archive inside the record that holds the results and the spectra.
ARCHIVE = "results.tar.gz"

#: The directories the archive must unpack into.
UNPACKED = ("results", "psd")

#: Where the data goes when the caller names no directory: ``data/`` at the repository root.
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"

CHUNK_BYTES = 1 << 20


def record_files(record_id: str) -> dict[str, tuple[str, str]]:
    """Return the files of a published record, as ``name -> (download URL, md5)``.

    Args:
        record_id: The Zenodo record.

    Returns:
        The files.

    Raises:
        ValueError: If a file's checksum is not an MD5 checksum.
    """
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record_id}", timeout=60) as response:
        record = json.load(response)
    files = {}
    for entry in record["files"]:
        algorithm, _, digest = entry["checksum"].partition(":")
        if algorithm != "md5":
            raise ValueError(f"{entry['key']}: unexpected checksum {entry['checksum']!r}.")
        files[entry["key"]] = (entry["links"]["self"], digest)
    return files


def md5_of(path: Path) -> str:
    """Return the MD5 checksum of a file, as hexadecimal."""
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, expected_md5: str, target: Path) -> None:
    """Download one file and check it, keeping a copy that already checks.

    Args:
        url: Where to download it from.
        expected_md5: Its published checksum.
        target: Where to write it.

    Raises:
        ValueError: If the downloaded file does not match its checksum.
    """
    if target.exists() and md5_of(target) == expected_md5:
        print(f"{target.name}: already present and checked")
        return
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream, CHUNK_BYTES)
    actual = md5_of(partial)
    if actual != expected_md5:
        partial.unlink()
        raise ValueError(f"{target.name}: checksum {actual} does not match the published {expected_md5}.")
    partial.replace(target)
    print(f"{target.name}: downloaded and checked")


def unpack(archive: Path, data: Path) -> None:
    """Unpack the archive into the data directory, replacing what an earlier unpack left.

    Args:
        archive: The ``results.tar.gz`` of the release.
        data: The data directory.

    Raises:
        ValueError: If the archive holds anything outside ``results/`` and ``psd/``.
    """
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        outside = [m.name for m in members if m.name.split("/", 1)[0] not in UNPACKED]
        if outside:
            raise ValueError(f"{archive}: unexpected entries {outside[:5]}.")
        for name in UNPACKED:
            shutil.rmtree(data / name, ignore_errors=True)
        tar.extractall(data, members=members, filter="data")
    counts = {name: sum(1 for p in (data / name).rglob("*") if p.is_file()) for name in UNPACKED}
    print(f"unpacked into {data}: " + ", ".join(f"{count} files in {name}/" for name, count in counts.items()))


def main() -> int:
    """Fetch, check and unpack the data release."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="where to unpack (default: data/)")
    parser.add_argument("--record", default=RECORD_ID, help=f"the Zenodo record (default: {RECORD_ID})")
    parser.add_argument("--archive", type=Path, help="unpack this local results.tar.gz instead of downloading")
    arguments = parser.parse_args()
    data = arguments.data
    data.mkdir(parents=True, exist_ok=True)
    archive = arguments.archive
    if archive is None:
        downloads = data / "download"
        downloads.mkdir(exist_ok=True)
        for name, (url, md5) in record_files(arguments.record).items():
            download(url, md5, downloads / name)
        archive = downloads / ARCHIVE
    unpack(archive, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
