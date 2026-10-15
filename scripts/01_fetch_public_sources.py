#!/usr/bin/env python3
"""Download or verify only the public Stage 0 sources frozen in the M0 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage0.yaml"
SOURCE_MANIFEST_PATH = ROOT / "manifests" / "public" / "source_revisions.json"
PRIVATE_MANIFEST_PATH = ROOT / "manifests" / "private" / "downloads.json"
TARGETS = ("sro", "llama_base_8b", "minitaur_8b")


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if config["downloads"]["public_data_only"] is not True:
        raise RuntimeError("Stage 0 download contract must remain public-data-only")
    return config, source_manifest


def run(command: list[str], cwd: Path = ROOT) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_git_checkout(destination: Path, expected_revision: str, expected_url: str) -> None:
    if not (destination / ".git").is_dir():
        raise RuntimeError(f"Missing Git checkout: {destination}")
    actual_revision = run(["git", "rev-parse", "HEAD"], cwd=destination)
    actual_url = run(["git", "remote", "get-url", "origin"], cwd=destination)
    if actual_revision != expected_revision:
        raise RuntimeError(f"Revision mismatch for {destination.name}: {actual_revision}")
    if actual_url.rstrip("/") != expected_url.rstrip("/"):
        raise RuntimeError(f"Origin mismatch for {destination.name}: {actual_url}")


def download_sro(spec: dict[str, Any]) -> Path:
    destination = ROOT / spec["local_dir"]
    if destination.exists():
        verify_git_checkout(destination, spec["revision"], spec["repository"])
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--no-tags", spec["repository"], str(destination)])
    run(["git", "checkout", "--detach", spec["revision"]], cwd=destination)
    verify_git_checkout(destination, spec["revision"], spec["repository"])
    return destination


def download_modelscope(spec: dict[str, Any], cache_dir: Path, max_workers: int) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    destination = ROOT / spec["local_dir"]
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(cache_dir)
    snapshot_download(
        model_id=spec["mirror_id"],
        revision=spec["mirror_revision"],
        local_dir=str(destination),
        allow_patterns=spec["allow_patterns"],
        max_workers=max_workers,
    )
    return destination


def download_huggingface_mirror(spec: dict[str, Any], cache_dir: Path, max_workers: int) -> Path:
    from huggingface_hub import snapshot_download

    destination = ROOT / spec["local_dir"]
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_ENDPOINT"] = spec["endpoint"]
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    snapshot_download(
        repo_id=spec["official_id"],
        revision=spec["official_revision"],
        local_dir=str(destination),
        allow_patterns=spec["allow_patterns"],
        max_workers=max_workers,
        endpoint=spec["endpoint"],
    )
    return destination


def expected_files(source_manifest: dict[str, Any], target: str) -> dict[str, Any]:
    return source_manifest.get("expected_model_files", {}).get(target, {})


def verify_expected_files(destination: Path, expected: dict[str, Any]) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    for relative_path, contract in sorted(expected.items()):
        path = destination / relative_path
        if not path.is_file():
            raise RuntimeError(f"Missing expected file: {path}")
        size = path.stat().st_size
        if size != contract["bytes"]:
            raise RuntimeError(f"Size mismatch for {path}: {size}")
        digest = sha256_file(path)
        if digest != contract["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {digest}")
        verified.append({"path": relative_path, "bytes": size, "sha256": digest})
    return verified


def promote_staged_model(
    source: Path,
    destination: Path,
    spec: dict[str, Any],
    expected: dict[str, Any],
) -> Path:
    """Verify an NVMe-staged checkpoint, then resumably copy allowed files."""
    if not source.is_dir():
        raise RuntimeError(f"Missing staging directory: {source}")
    verify_expected_files(source, expected)
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "--archive", "--partial", "--protect-args"]
    for pattern in spec["allow_patterns"]:
        command.append(f"--include={pattern}")
    command.extend(["--exclude=*", f"{source}/", f"{destination}/"])
    run(command)
    return destination


def inventory(destination: Path) -> dict[str, int]:
    files = [path for path in destination.rglob("*") if path.is_file() and ".git" not in path.parts]
    return {"file_count": len(files), "total_bytes": sum(path.stat().st_size for path in files)}


def write_private_manifest(target: str, destination: Path, verified: list[dict[str, Any]]) -> None:
    PRIVATE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PRIVATE_MANIFEST_PATH.exists():
        payload = json.loads(PRIVATE_MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        payload = {"schema_version": 1, "downloads": {}}
    payload["downloads"][target] = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_path_relative_to_project": str(destination.relative_to(ROOT)),
        **inventory(destination),
        "verified_expected_files": verified,
    }
    temporary = PRIVATE_MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(PRIVATE_MANIFEST_PATH)


def process_target(
    action: str,
    target: str,
    config: dict[str, Any],
    source_manifest: dict[str, Any],
    source_dir: Path | None = None,
) -> None:
    downloads = config["downloads"]
    spec = downloads[target]
    destination = ROOT / spec["local_dir"]
    if action == "download":
        if target == "sro":
            destination = download_sro(spec)
        elif target == "llama_base_8b":
            destination = download_modelscope(
                spec,
                ROOT / config["cache"]["modelscope"],
                downloads["max_workers"],
            )
        elif target == "minitaur_8b":
            destination = download_huggingface_mirror(
                spec,
                ROOT / config["cache"]["huggingface"],
                downloads["max_workers"],
            )
    elif action == "promote":
        if target == "sro":
            raise RuntimeError("SRO is a Git checkout and cannot use model promotion")
        if source_dir is None:
            raise RuntimeError("--source-dir is required for model promotion")
        destination = promote_staged_model(
            source_dir.resolve(),
            destination,
            spec,
            expected_files(source_manifest, target),
        )
    if target == "sro":
        verify_git_checkout(destination, spec["revision"], spec["repository"])
        verified: list[dict[str, Any]] = []
    else:
        if not destination.is_dir():
            raise RuntimeError(f"Model directory does not exist: {destination}")
        verified = verify_expected_files(destination, expected_files(source_manifest, target))
    write_private_manifest(target, destination, verified)
    print(json.dumps({"target": target, "status": "verified", **inventory(destination)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("download", "promote", "verify"))
    parser.add_argument("--target", choices=(*TARGETS, "all"), default="all")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="NVMe staging directory used only with promote and one model target",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, source_manifest = load_contract()
    if args.action == "promote" and args.target in ("all", "sro"):
        raise RuntimeError("promote requires one model target")
    targets = TARGETS if args.target == "all" else (args.target,)
    for target in targets:
        process_target(args.action, target, config, source_manifest, args.source_dir)


if __name__ == "__main__":
    main()
