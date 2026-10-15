#!/usr/bin/env python3
"""Reconcile concurrent O3 workers into one verified model-task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/original/main.yaml"
SESSION_ROOT = ROOT / "data/derived/original/sro_sessions/development"
STATE_ROOT = ROOT / "cache/original/8b_states"
MANIFEST = ROOT / "manifests/private/original/8b_state_extraction.json"
SPLITS = ROOT / "configs/original/splits.json"
MODEL_PATHS = {
    "llama_base_8b": ROOT / "models/Llama-3.1-8B",
    "minitaur_8b": ROOT / "models/Llama-3.1-Minitaur-8B",
    "reproduced_intact_psych101_8b": ROOT / "models/Psych101-Reproduced-8B",
    "response_shuffled_psych101_8b": ROOT / "models/Psych101-Response-Shuffled-8B",
}
MODEL_REVISIONS = {
    "llama_base_8b": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "minitaur_8b": "7e784a8933931f4e8a6954aa0d8dae5f5b0b764d",
    "reproduced_intact_psych101_8b": "local_original_O8_intact_final_epoch_5",
    "response_shuffled_psych101_8b": "local_original_O8_shuffled_final_epoch_5",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_rows(task: str) -> int:
    validity = pd.read_parquet(SESSION_ROOT / f"{task}.parquet", columns=["validity"])
    return int((validity["validity"] == "primary").sum())


def _verify_task(model: str, task: str, layers: list[int]) -> dict[str, Any]:
    root = STATE_ROOT / model / task
    partials = sorted(root.glob("*.partial.npy"))
    if partials:
        raise RuntimeError(f"O3 still has partial arrays for {model}/{task}: {partials}")
    metadata_path = root / "metadata.parquet"
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing O3 metadata for {model}/{task}")
    metadata = pd.read_parquet(metadata_path)
    confirmation = set(json.loads(SPLITS.read_text())["confirmation"])
    overlap = set(metadata["participant"]) & confirmation
    if overlap:
        raise RuntimeError(f"Confirmation participants leaked into {model}/{task}: {len(overlap)}")
    expected = _expected_rows(task)
    if len(metadata) != expected:
        raise RuntimeError(f"O3 row mismatch for {model}/{task}: {len(metadata)}/{expected}")
    if metadata["state_row"].tolist() != list(range(expected)):
        raise RuntimeError(f"O3 state-row ordering is not contiguous for {model}/{task}")
    revisions = set(metadata["model_revision"].astype(str))
    if revisions != {MODEL_REVISIONS[model]}:
        raise RuntimeError(f"O3 model revision mismatch for {model}/{task}: {revisions}")
    layer_files: dict[str, str] = {}
    hidden_size = int(json.loads((MODEL_PATHS[model] / "config.json").read_text())["hidden_size"])
    for layer in layers:
        path = root / f"layer_{layer:02d}.npy"
        if not path.is_file():
            raise RuntimeError(f"Missing O3 layer array: {path}")
        array = np.load(path, mmap_mode="r")
        if array.shape != (expected, hidden_size) or array.dtype != np.float16:
            raise RuntimeError(
                f"O3 array contract mismatch for {model}/{task}/layer{layer}: "
                f"shape={array.shape}, dtype={array.dtype}"
            )
        layer_files[str(layer)] = _sha256(path)
    return {
        "status": "completed_and_reconciled",
        "rows": expected,
        "metadata_sha256": _sha256(metadata_path),
        "layer_files": layer_files,
        "model_revision": MODEL_REVISIONS[model],
    }


def reconcile(models: list[str]) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text())
    layers = [int(value) for value in config["likelihood_and_representation"]["layers_8b"]]
    payload = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"completed": {}}
    payload.update(
        {
            "schema_version": 1,
            "profile": "original",
            "config_sha256": _sha256(CONFIG),
            "reconciled_models": sorted(set(payload.get("reconciled_models", [])) | set(models)),
        }
    )
    payload.setdefault("completed", {})
    for model in models:
        if not MODEL_PATHS[model].is_dir():
            raise RuntimeError(f"Missing model before O3 reconciliation: {MODEL_PATHS[model]}")
        for task in SPECS:
            payload["completed"][f"{model}/{task}"] = _verify_task(model, task, layers)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(MANIFEST)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_PATHS), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = reconcile(args.models)
    print(
        json.dumps(
            {
                "status": "completed",
                "models": args.models,
                "model_task_entries": len(payload["completed"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
