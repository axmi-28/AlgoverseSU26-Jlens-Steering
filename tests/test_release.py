"""Release boundaries: integrity, path safety and non-destructive restoration."""

import gzip
import json

import pytest

from scripts import release


@pytest.mark.parametrize(
    "path", ["../private", "/tmp/private", "x/../../private", "x\\private"]
)
def test_restore_rejects_escaping_paths(tmp_path, path):
    with pytest.raises(ValueError):
        release.safe_path(tmp_path, path)


def test_restore_checks_every_conflict_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        release,
        "reference_items",
        lambda: iter(
            [
                ({"path": "results/new.json"}, b"new"),
                ({"path": "results/existing.json"}, b"reference"),
            ]
        ),
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "existing.json").write_bytes(b"my result")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        release.restore(tmp_path)
    assert not (results / "new.json").exists()
    assert (results / "existing.json").read_bytes() == b"my result"


def test_reference_checksum_rejects_corruption(tmp_path, monkeypatch):
    raw = b'{"answer": 1}'
    packed = gzip.compress(raw, mtime=0)
    (tmp_path / "result.gz").write_bytes(packed)
    manifest = tmp_path / "manifest.json"
    item = dict(
        path="results/example.json",
        packed="result.gz",
        bytes=len(raw),
        sha256=release.sha(raw),
        packed_sha256=release.sha(packed),
    )
    manifest.write_text(json.dumps(dict(schema=1, files=[item])))
    monkeypatch.setattr(release, "MANIFEST", manifest)
    assert list(release.reference_items()) == [(item, raw)]
    (tmp_path / "result.gz").write_bytes(packed + b"corruption")
    with pytest.raises(ValueError, match="Compressed checksum mismatch"):
        list(release.reference_items())


def test_archive_allowlist_omits_private_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "ROOT", tmp_path)
    for rel in [
        "README.md",
        "src/jsteer/example.py",
        "paper/main.tex",
        ".git/config",
        "notes/private.md",
        "src/jsteer/__pycache__/a.pyc",
    ]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("example")
    assert {p.relative_to(tmp_path).as_posix() for p in release.release_files()} == {
        "README.md",
        "src/jsteer/example.py",
    }


def test_symlink_cannot_escape_restore_root(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        release.safe_path(tmp_path, "link/private")
