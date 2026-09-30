"""Verify/restore reference data and create a history-free source archive.

Uses only the Python standard library. Never extracts arbitrary archive paths,
overwrites differing results, includes Git metadata, or contacts a service.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/reference/manifest.json"
TOP_FILES = {
    ".gitignore",
    ".gitattributes",
    "README.md",
    "MODEL_CARD.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "modal_app.py",
}
PREFIXES = (
    "src/jsteer/",
    "scripts/",
    "tests/",
    "configs/",
    "docs/",
    "requirements/",
    "data/heldout/",
    "data/protocols/",
    "data/reference/",
    ".github/workflows/",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or "\\" in relative:
        raise ValueError(f"Unsafe relative path: {relative}")
    target = root / p
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes destination: {relative}")
    return target


def reference_items():
    manifest = json.loads(MANIFEST.read_text())
    if manifest["schema"] != 1:
        raise ValueError("Unsupported reference manifest")
    seen = set()
    for item in manifest["files"]:
        if item["path"] in seen:
            raise ValueError(f"Duplicate destination: {item['path']}")
        seen.add(item["path"])
        if not item["path"].startswith(("results/", "data/jlens-upstream/")):
            raise ValueError(f"Unexpected reference destination: {item['path']}")
        safe_path(ROOT, item["path"])
        packed = safe_path(MANIFEST.parent, item["packed"])
        raw_gz = packed.read_bytes()
        if sha(raw_gz) != item["packed_sha256"]:
            raise ValueError(f"Compressed checksum mismatch: {item['packed']}")
        raw = gzip.decompress(raw_gz)
        if len(raw) != item["bytes"] or sha(raw) != item["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {item['path']}")
        yield item, raw


def restore(destination: Path):
    # Check ALL files and existing destinations before writing anything.
    for item, raw in reference_items():
        target = safe_path(destination, item["path"])
        if target.exists() and sha(target.read_bytes()) != sha(raw):
            raise ValueError(f"Refusing to overwrite changed result: {target}")
    count = 0
    for item, raw in reference_items():
        target = safe_path(destination, item["path"])
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".restore-tmp")
            temporary.write_bytes(raw)
            temporary.replace(target)
        count += 1
    print(f"Restored/verified {count} reference files in {destination}")


def release_files():
    """Explicit inclusion policy, independent of the user's working files."""
    files = []
    candidates = [ROOT / p for p in TOP_FILES | {"data/README.md"}]
    for prefix in PREFIXES:
        candidates.extend((ROOT / prefix).rglob("*"))
    for p in candidates:
        rel = p.relative_to(ROOT).as_posix()
        # The archive takes only curated sources/data, never caches or symlinks.
        if not (
            rel in TOP_FILES or rel == "data/README.md" or rel.startswith(PREFIXES)
        ):
            continue
        if any(
            x.startswith(".") and x != ".github" for x in p.relative_to(ROOT).parts[:-1]
        ):
            continue
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"}:
            continue
        if p.is_symlink():
            raise ValueError(f"Release contains symlink: {rel}")
        if p.is_file():
            files.append(p)
    return sorted(files)


def audit(files, deny=()):
    patterns = [
        b"/" + rb"Users/[^/\s]+/",
        b"C:" + rb"\\Users\\[^\\\s]+\\",
        rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}",
        rb"hf_[A-Za-z0-9]{30,}",
    ]
    patterns += [re.escape(word.encode()) for word in deny]
    for p in files:
        if p.stat().st_size >= 50 * 1024**2:
            raise ValueError(f"Release file exceeds 50 MiB: {p.relative_to(ROOT)}")
        if p.suffix == ".gz":
            raw = gzip.decompress(p.read_bytes())
        else:
            raw = p.read_bytes()
        # Scan Torch's ZIP metadata as well as text and compressed JSON.
        if raw.startswith(b"PK\x03\x04"):
            import io

            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                raw = b"\n".join(z.read(n) for n in z.namelist() if n.endswith(".pkl"))
        for pattern in patterns:
            if re.search(pattern, raw, re.IGNORECASE):
                raise ValueError(
                    f"Anonymity/secret check failed: {p.relative_to(ROOT)}"
                )


def archive(output: Path, deny=()):
    files = release_files()
    audit(files, deny)
    list_count = sum(1 for _ in reference_items())
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            raw = p.read_bytes()
            info = zipfile.ZipInfo("jlens-steering/" + rel, (2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = (
                zipfile.ZIP_STORED if p.suffix == ".gz" else zipfile.ZIP_DEFLATED
            )
            z.writestr(info, raw)
            checksums.append(f"{sha(raw)}  {rel}")
        info = zipfile.ZipInfo("jlens-steering/SHA256SUMS", (2026, 1, 1, 0, 0, 0))
        z.writestr(info, "\n".join(checksums) + "\n")
    print(f"Archived {len(files)} files; {list_count} reference artifacts; {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "restore", "archive", "audit"))
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "dist/jlens-steering-anonymous.zip"
    )
    parser.add_argument(
        "--deny",
        action="append",
        default=[],
        help="Additional private identifier to reject",
    )
    args = parser.parse_args()
    if args.command == "restore":
        restore(args.destination)
    elif args.command == "archive":
        archive(args.output, args.deny)
    elif args.command == "audit":
        files = release_files()
        audit(files, args.deny)
        # Tracked private files must not survive simply because archive excludes them.
        if (ROOT / ".git").exists():
            tracked = (
                subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
                .decode()
                .split("\0")
            )
            allowed = {p.relative_to(ROOT).as_posix() for p in files} | {
                "results/.gitkeep"
            }
            unexpected = [p for p in tracked if p and p not in allowed]
            if unexpected:
                raise ValueError(f"Unexpected tracked paths: {unexpected}")
        print(f"Release audit passed: {len(files)} files")
    else:
        n = sum(1 for _ in reference_items())
        print(f"Verified {n} checksummed reference artifacts")


if __name__ == "__main__":
    main()
