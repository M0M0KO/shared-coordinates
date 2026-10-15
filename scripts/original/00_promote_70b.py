#!/usr/bin/env python3
"""Verify the ModelScope-staged Llama-3.1-70B snapshot and promote it atomically."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "models/Llama-3.1-70B"
INCOMING = ROOT / "models/Llama-3.1-70B.incoming"
MANIFEST = ROOT / "manifests/private/original/llama70b_download.json"
SOURCE_INVENTORY = ROOT / "manifests/public/llama70b_modelscope_source.json"
ESSENTIAL = {
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_against_frozen_source(source: Path, payload: dict[str, Any]) -> None:
    if payload.get("model_id") != "LLM-Research/Meta-Llama-3.1-70B":
        raise RuntimeError("Frozen 70B source inventory has the wrong ModelScope model ID")
    expected = {str(item["path"]): int(item["bytes"]) for item in payload["files"]}
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("70B staging misses model.safetensors.index.json")
    index = json.loads(index_path.read_text())
    required = set(ESSENTIAL) | set(index["weight_map"].values())
    missing_inventory = sorted(name for name in required if name not in expected)
    if missing_inventory:
        raise RuntimeError(
            f"Frozen ModelScope inventory misses required files: {missing_inventory[:8]}"
        )
    mismatches = sorted(
        name
        for name in required
        if not (source / name).is_file() or (source / name).stat().st_size != expected[name]
    )
    if mismatches:
        raise RuntimeError(
            f"70B staging differs from the frozen ModelScope file sizes: {mismatches[:8]}"
        )


def _inventory(source: Path) -> dict[str, Any]:
    if not source.is_dir() or source.resolve() in {Path("/"), Path("/tmp")}:
        raise RuntimeError(f"Refusing unsafe or missing 70B source: {source}")
    incomplete = sorted(
        path.name
        for path in source.iterdir()
        if path.is_file() and (".incomplete" in path.name or ".part" in path.name)
    )
    if incomplete:
        raise RuntimeError(f"70B staging still has partial files: {incomplete[:8]}")
    missing_essential = sorted(name for name in ESSENTIAL if not (source / name).is_file())
    if missing_essential:
        raise RuntimeError(f"70B staging misses essential files: {missing_essential}")
    if not SOURCE_INVENTORY.is_file():
        raise RuntimeError("Frozen ModelScope 70B source inventory is missing")
    source_inventory = json.loads(SOURCE_INVENTORY.read_text())
    _verify_against_frozen_source(source, source_inventory)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 30:
        raise RuntimeError(f"Expected 30 indexed 70B shards, found {len(shards)}")
    missing_shards = [name for name in shards if not (source / name).is_file()]
    if missing_shards:
        raise RuntimeError(f"70B staging misses indexed shards: {missing_shards[:8]}")
    config = json.loads((source / "config.json").read_text())
    if int(config.get("num_hidden_layers", -1)) != 80:
        raise RuntimeError("Staged model is not the frozen 80-layer Llama-3.1-70B checkpoint")
    allowed_names = (
        set(shards)
        | ESSENTIAL
        | {
            "LICENSE",
            "README.md",
            "USE_POLICY.md",
            "configuration.json",
            "generation_config.json",
            "special_tokens_map.json",
        }
    )
    files = sorted(
        path for path in source.iterdir() if path.is_file() and path.name in allowed_names
    )
    entries = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files
    ]
    return {
        "source_provider": source_inventory["provider"],
        "source_model_id": source_inventory["model_id"],
        "source_requested_revision": source_inventory["requested_revision"],
        "source_inventory_sha256": _sha256(SOURCE_INVENTORY),
        "weight_shards": len(shards),
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
    }


def _verify_against_inventory(root: Path, inventory: dict[str, Any]) -> None:
    for item in inventory["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Promoted 70B file is missing or has the wrong size: {path}")
        if _sha256(path) != item["sha256"]:
            raise RuntimeError(f"Promoted 70B SHA-256 mismatch: {path}")


def promote(source: Path) -> dict[str, Any]:
    source = source.resolve()
    inventory = _inventory(source)
    if DESTINATION.exists():
        _verify_against_inventory(DESTINATION, inventory)
        status = "already_promoted_and_verified"
    else:
        INCOMING.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "--archive",
                "--partial",
                "--exclude=*.incomplete",
                "--exclude=*.part*",
                "--exclude=.cache/",
                f"{source}/",
                f"{INCOMING}/",
            ],
            check=True,
        )
        _verify_against_inventory(INCOMING, inventory)
        INCOMING.replace(DESTINATION)
        status = "promoted_and_verified"
    payload = {
        "schema_version": 1,
        "profile": "original",
        "status": status,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(DESTINATION.relative_to(ROOT)),
        **inventory,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(MANIFEST)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify", "promote"))
    parser.add_argument("--source-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "promote":
        payload = promote(args.source_dir)
    else:
        inventory = _inventory(args.source_dir.resolve())
        payload = {"status": "verified_staging", **inventory}
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
