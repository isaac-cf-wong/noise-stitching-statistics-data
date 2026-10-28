"""The fetch script: it checks what it downloads and unpacks only the release's directories."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_data.py"


@pytest.fixture(scope="module")
def fetch() -> ModuleType:
    """Return the fetch script as an imported module."""
    specification = importlib.util.spec_from_file_location("fetch_data", SCRIPT)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _archive(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_unpacks_the_results_and_the_spectra(fetch: ModuleType, tmp_path: Path) -> None:
    """The archive's two directories land under the data directory, replacing an earlier unpack."""
    archive = _archive(tmp_path / "a.tar.gz", {"results/x.json": b"{}", "psd/a.txt": b"1 2\n"})
    data = tmp_path / "data"
    (data / "results").mkdir(parents=True)
    (data / "results" / "stale.json").write_text("{}")
    fetch.unpack(archive, data)
    assert sorted(p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()) == [
        "psd/a.txt",
        "results/x.json",
    ]


@pytest.mark.parametrize("name", ["other/x.json", "../escape.json", "/abs/x.json"])
def test_refuses_anything_outside_the_release(fetch: ModuleType, tmp_path: Path, name: str) -> None:
    """An entry outside ``results/`` and ``psd/`` stops the unpack before anything is written."""
    archive = _archive(tmp_path / "a.tar.gz", {"results/x.json": b"{}", name: b"x"})
    with pytest.raises(ValueError, match="unexpected entries"):
        fetch.unpack(archive, tmp_path / "data")
    assert not (tmp_path / "data" / "results").exists()


def test_a_corrupt_download_is_refused(fetch: ModuleType, tmp_path: Path) -> None:
    """A file whose checksum does not match the published one is deleted, not kept."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"what was published")
    target = tmp_path / "out.bin"
    with pytest.raises(ValueError, match="does not match"):
        fetch.download(source.as_uri(), hashlib.md5(b"something else").hexdigest(), target)
    assert not target.exists()
    assert not target.with_name("out.bin.part").exists()


def test_a_good_download_is_kept(fetch: ModuleType, tmp_path: Path) -> None:
    """A file that matches its checksum is written under its name."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"what was published")
    target = tmp_path / "out.bin"
    fetch.download(source.as_uri(), hashlib.md5(b"what was published").hexdigest(), target)
    assert target.read_bytes() == b"what was published"


def test_reads_the_files_and_checksums_of_a_record(fetch: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each file of the record maps to its download link and its MD5 checksum."""
    record = {"files": [{"key": "results.tar.gz", "checksum": "md5:abc", "links": {"self": "https://x/content"}}]}
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *_, **__: io.BytesIO(json.dumps(record).encode()))
    assert fetch.record_files("1") == {"results.tar.gz": ("https://x/content", "abc")}


def test_refuses_a_checksum_it_cannot_check(fetch: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """A record that publishes another kind of checksum is refused rather than trusted."""
    record = {"files": [{"key": "f", "checksum": "sha1:abc", "links": {"self": "https://x"}}]}
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *_, **__: io.BytesIO(json.dumps(record).encode()))
    with pytest.raises(ValueError, match="unexpected checksum"):
        fetch.record_files("1")


def test_a_record_that_is_not_public_is_named_as_such(fetch: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing record reads as one, with the way round it, not as an HTTP traceback."""

    def missing(*_: object, **__: object) -> None:
        raise urllib.error.HTTPError("https://x", 404, "NOT FOUND", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(fetch.urllib.request, "urlopen", missing)
    with pytest.raises(LookupError, match="not public"):
        fetch.record_files("1")


def test_an_interrupted_download_leaves_no_partial_file(
    fetch: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transfer that breaks after writing part of the file removes what it wrote."""

    class Broken(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            if self.tell():
                raise ConnectionResetError("connection reset")
            return super().read(4)

    partial = tmp_path / "out.bin.part"
    seen = []
    original = Path.unlink

    def record(path: Path, *arguments: object, **options: object) -> None:
        seen.append(path.exists())
        original(path, *arguments, **options)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *_, **__: Broken(b"partial content"))
    monkeypatch.setattr(Path, "unlink", record)
    with pytest.raises(ConnectionResetError, match="connection reset"):
        fetch.download("https://x", "0" * 32, tmp_path / "out.bin")
    assert seen == [True]
    assert not partial.exists()
