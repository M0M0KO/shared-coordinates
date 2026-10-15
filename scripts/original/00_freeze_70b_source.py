#!/usr/bin/env python3
"""Freeze the public ModelScope inventory used for the original 70B download."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modelscope.hub.api import HubApi

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "manifests/public/llama70b_modelscope_source.json"
MODEL_ID = "LLM-Research/Meta-Llama-3.1-70B"
REVISION = "master"


def _timestamp_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def freeze_source() -> dict[str, Any]:
    api = HubApi()
    metadata = api.get_model(MODEL_ID, revision=REVISION)
    files = api.get_model_files(MODEL_ID, revision=REVISION)
    entries = sorted(
        (
            {"path": str(item["Path"]), "bytes": int(item["Size"])}
            for item in files
            if item.get("Path") and item.get("Size") is not None
        ),
        key=lambda item: item["path"],
    )
    payload = {
        "schema_version": 1,
        "profile": "original",
        "provider": "ModelScope",
        "model_id": MODEL_ID,
        "requested_revision": REVISION,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_created_at": _timestamp_value(metadata.get("CreatedAt")),
        "provider_updated_at": _timestamp_value(metadata.get("UpdatedAt")),
        "provider_file_count": len(entries),
        "provider_total_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
        "integrity_boundary": (
            "The public API exposes file sizes but not immutable per-file digests. "
            "Promotion therefore checks these frozen sizes, local SHA-256 values, the "
            "30-shard index and successful exact-BF16 model loading."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    return payload


def main() -> None:
    payload = freeze_source()
    print(
        json.dumps(
            {
                "model_id": payload["model_id"],
                "provider_file_count": payload["provider_file_count"],
                "provider_total_bytes": payload["provider_total_bytes"],
                "output": str(OUTPUT.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
