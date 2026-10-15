#!/usr/bin/env python3
"""Check that a proposed code release contains only intended files."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {
    "data", "datasets", "models", "checkpoints", "cache", "results", "logs",
    "artifacts", "manuscript", "paper", "submissions",
}
FORBIDDEN_SUFFIXES = {
    ".doc", ".docx", ".safetensors", ".bin", ".pt", ".pth", ".ckpt",
    ".npy", ".npz", ".parquet", ".arrow",
}
PRIVATE_PATH_MARKERS = (b"/private_nas/", b"/data1/", b"C:\\Users\\", b"D:\\Software\\", b"X:\\Work\\")
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".json", ".csv", ".toml", ".txt", ".lock"}


def tracked_files() -> list[Path]:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    return [ROOT / name.decode("utf-8") for name in names if name]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    files = tracked_files()
    source_manifest = json.loads((ROOT / "source_data/manifest.json").read_text(encoding="utf-8"))
    source_hashes = source_manifest["files"]
    for name in source_hashes:
        relative_data = Path(name)
        if relative_data.is_absolute() or relative_data.parts[:1] != ("analysis_tables",) or ".." in relative_data.parts:
            failures.append(f"Invalid source-data manifest path: {name}")
    allowed_source_files = {"source_data/README.md", "source_data/manifest.json"}
    allowed_source_files.update(f"source_data/{name}" for name in source_hashes)
    if not files:
        failures.append("No tracked files; stage the release first")
    for path in files:
        relative = path.relative_to(ROOT)
        relative_posix = relative.as_posix()
        if relative.parts[0].lower() in FORBIDDEN_DIRS:
            failures.append(f"Excluded directory is tracked: {relative}")
        if relative.parts[0] == "source_data" and relative_posix not in allowed_source_files:
            failures.append(f"Unlisted source-data file is tracked: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and relative_posix not in allowed_source_files:
            failures.append(f"Excluded file type is tracked: {relative}")
        if not path.is_file():
            failures.append(f"Tracked file is missing: {relative}")
            continue
        if path.suffix.lower() in TEXT_SUFFIXES and path.resolve() != Path(__file__).resolve():
            content = path.read_bytes()
            if any(marker in content for marker in PRIVATE_PATH_MARKERS):
                failures.append(f"Machine-specific path appears in: {relative}")

    for name, expected in source_hashes.items():
        path = ROOT / "source_data" / name
        if not path.is_file() or path not in files or sha256(path) != expected:
            failures.append(f"Source-data hash mismatch: {name}")

    inventory = json.loads((ROOT / "source_inventory.json").read_text(encoding="utf-8"))
    for relative, expected in inventory.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"Scientific source hash mismatch: {relative}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Release boundary OK: {len(files)} tracked files; {len(inventory)} source hashes and {len(source_hashes)} data hashes match")


if __name__ == "__main__":
    main()
