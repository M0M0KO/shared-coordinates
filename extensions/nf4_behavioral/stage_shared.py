#!/usr/bin/env python3
"""Run frozen shared-manifold algorithms through an isolated NF4 I/O view.

The source modules are imported unchanged. Only paths, command-line model
identity and metadata attribution are adapted. The 70B NF4 replication cannot
satisfy the original exact-BF16 or 8B fine-tuning-control requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

MODELS = ("centaur_70b_nf4", "llama_base_70b_nf4")
PROFILE = "exploratory_nf4_behavioral_replication"
ACTIONS = ("rank-curve", "full", "task-specific", "haar", "held-out", "finalize")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checked_path(view: Path, relative: str) -> Path:
    path = view / relative
    if not path.resolve().is_relative_to(view.resolve()):
        raise RuntimeError(f"Output escapes the isolated project view: {path}")
    return path


def output_paths(view: Path, model: str, layer: int) -> dict[str, Path]:
    if model not in MODELS or layer != 60:
        raise RuntimeError("This adapter supports only the frozen 70B NF4 primary layer")
    suffix = f"{model}_layer_{layer:02d}"
    relative = {
        "config": "configs/original/main.yaml",
        "selection": f"results/original/selected_shared_{suffix}.json",
        "shared": f"results/original/shared_manifold_{suffix}.parquet",
        "control_cache": f"cache/original/shared_controls/{model}/layer_{layer:02d}",
        "control_result": f"results/original/shared_controls_{suffix}.parquet",
        "control_manifest": f"manifests/private/original/shared_controls_{suffix}.json",
        "preprocessing_manifest": f"manifests/private/original/preprocessing_{suffix}.json",
        "unavailable": f"manifests/private/original/shared_controls_unavailable_{suffix}.json",
        "projected": f"cache/original/preprocessing/{model}/layer_{layer:02d}/projected",
        "states": f"cache/original/8b_states/{model}",
        "final": f"cache/original/shared_final/{model}/layer_{layer:02d}",
        "cv": f"cache/original/shared_cv/{model}/layer_{layer:02d}/q512_rm0_unit0",
    }
    # State inputs may be symlinks to the immutable extraction. All outputs must
    # resolve inside the isolated view, including parent-directory symlinks.
    return {
        key: view / value if key == "states" else checked_path(view, value)
        for key, value in relative.items()
    }


def validate_roots(snapshot: Path, view: Path) -> None:
    snapshot, view = snapshot.resolve(), view.resolve()
    if snapshot == view or snapshot.is_relative_to(view) or view.is_relative_to(snapshot):
        raise RuntimeError("The NF4 project view and frozen source snapshot must be disjoint")
    if not snapshot.is_dir() or not view.is_dir():
        raise RuntimeError("Both the frozen snapshot and isolated project view must exist")


def verify_sources(snapshot: Path, view: Path) -> dict[str, Any]:
    inventory_path = snapshot / "source_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    for relative, expected in inventory.items():
        path = snapshot / relative
        if not path.resolve().is_relative_to(snapshot.resolve()):
            raise RuntimeError("Frozen source inventory escapes its snapshot")
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    config = view / "configs/original/main.yaml"
    split_path = view / "configs/original/splits.json"
    if sha256(config) != inventory["configs/original/main.yaml"]:
        raise RuntimeError("The isolated view must retain the unchanged frozen main configuration")
    if sha256(split_path) != inventory["configs/original/splits.json"]:
        raise RuntimeError("The isolated view must retain the unchanged frozen participant split")
    return {
        "source_inventory_sha256": sha256(inventory_path),
        "frozen_config_sha256": sha256(config),
        "frozen_splits_sha256": sha256(split_path),
        "verified_source_files": len(inventory),
    }


def validate_metadata_records(
    records: Iterable[dict[str, Any]],
    development: dict[str, int],
    confirmation: set[str],
    folds: int = 5,
) -> dict[str, Any]:
    """Reject participant leakage and fold reassignment before any PCA fitting."""
    if set(development) & confirmation:
        raise RuntimeError("The frozen development and confirmation cohorts overlap")
    participants: set[str] = set()
    count = 0
    for row in records:
        participant = str(row["participant"])
        fold = row["fold"]
        if participant in confirmation or participant not in development:
            raise RuntimeError(f"Non-development participant in fitting data: {participant}")
        if isinstance(fold, bool) or fold is None:
            raise RuntimeError(f"Invalid development fold for {participant}")
        try:
            integer_fold = int(fold)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeError(f"Invalid development fold for {participant}") from error
        if integer_fold != fold or integer_fold != development[participant]:
            raise RuntimeError(f"Development fold reassignment for {participant}")
        if not 0 <= integer_fold < folds:
            raise RuntimeError(f"Development fold out of range for {participant}")
        participants.add(participant)
        count += 1
    if not count:
        raise RuntimeError("Development task metadata is empty")
    return {"rows": count, "participants": len(participants), "confirmation_rows": 0}


def validate_development_inputs(view: Path, paths: dict[str, Path], tasks: Iterable[str]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    split = json.loads((view / "configs/original/splits.json").read_text())
    development = {str(item["participant"]): int(item["fold"]) for item in split["development"]}
    if len(development) != 372 or len(set(split["confirmation"])) != 150:
        raise RuntimeError("Frozen SRO cohort sizes differ from the original 372/150 split")
    confirmation = set(map(str, split["confirmation"]))
    evidence: dict[str, Any] = {}
    for task in tasks:
        metadata = pd.read_parquet(paths["states"] / task / "metadata.parquet", columns=["participant", "fold"])
        item = validate_metadata_records(metadata.to_dict("records"), development, confirmation)
        states = np.load(paths["states"] / task / "layer_60.npy", mmap_mode="r")
        if states.ndim != 2 or states.shape != (len(metadata), 8192) or states.dtype != np.float16:
            raise RuntimeError(f"NF4 state shape/dtype/metadata mismatch for {task}")
        item.update(raw_dimension=8192, state_storage_dtype=str(states.dtype))
        evidence[task] = item
        del states, metadata
    if len(evidence) != 16:
        raise RuntimeError("All 16 frozen SRO tasks are required")
    return evidence


def load_original(snapshot: Path, filename: str) -> Any:
    source = snapshot / "scripts/original" / filename
    sys.path.insert(0, str(snapshot / "src"))
    name = "nf4_frozen_" + filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def attributed_rows(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    if model not in MODELS:
        raise RuntimeError("Cannot attribute control rows to an unsupported model")
    return [{**row, "model": model} for row in rows]


def label_manifest(path: Path, model: str, source: dict[str, Any]) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    payload.update(
        profile=PROFILE,
        model=model,
        precision="nf4_4bit",
        satisfies_original_exact_bf16_requirement=False,
        algorithms="unchanged_frozen_original_source",
        provenance=source,
    )
    if path.name.startswith("shared_controls_"):
        payload.update(core_sro_controls_complete=True, complete_original_control_battery=False)
    atomic_json(path, payload)


def configure_rank_module(module: Any, view: Path, paths: dict[str, Path], model: str, layer: int) -> None:
    module.ROOT = view
    module.CONFIG_PATH = paths["config"]
    module.CACHE = view / "cache/original"
    module.MODEL_CHOICES = MODELS
    module.parse_args = lambda: SimpleNamespace(
        model=model, layer=layer, overwrite=False, preprocess_only=False,
        pca_dim=512, remove_pcs=0, unit_normalize=False,
    )


def configure_control_module(module: Any, view: Path, paths: dict[str, Path], model: str) -> None:
    module.ROOT = view
    module.CONFIG = paths["config"]
    module.SELECTION = paths["selection"]
    module.SHARED = paths["shared"]
    module.CACHE = paths["control_cache"]
    module.RESULT = paths["control_result"]
    module.MANIFEST = paths["control_manifest"]

    def data() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        config = module.yaml.safe_load(module.CONFIG.read_text())
        selection = json.loads(module.SELECTION.read_text())
        if selection["model"] != model or int(selection["layer"]) != 60 or int(selection["pca_dim"]) != 512:
            raise RuntimeError("Control selection does not match the NF4 primary representation")
        if selection.get("config_hash") != sha256(module.CONFIG):
            raise RuntimeError("Shared selection was fitted under a different configuration")
        if selection.get("remove_first_global_pcs") != 0 or selection.get("unit_length_normalization") is not False:
            raise RuntimeError("Control selection is not the frozen primary preprocessing condition")
        return config, selection, module.load_task_arrays(paths["projected"], paths["states"], list(module.SPECS), 512)

    module._data = data
    original_write_cache = module._write_cache
    module._write_cache = lambda key, rows: original_write_cache(key, attributed_rows(rows, model))
    original_shared_rows = module.shared_behavioral_rows
    module.shared_behavioral_rows = lambda selection: attributed_rows(original_shared_rows(selection), model)
    original_null_rows = module.null_intercept_rows
    module.null_intercept_rows = lambda config, selection, data: attributed_rows(original_null_rows(config, selection, data), model)

    def unavailable_finetune_controls(_rows: list[dict[str, Any]]) -> None:
        atomic_json(paths["unavailable"], {
            "profile": PROFILE,
            "model": model,
            "state": "dependency_unavailable",
            "controls": ["shared_reproduced_intact", "shared_response_shuffled"],
            "reason": "Matched 70B NF4 reproduced-intact and response-shuffled models/states have not been produced.",
            "8b_substitution_used": False,
            "complete_original_control_battery": False,
        })

    # Frozen finalize() still verifies every full/task/Haar/held-out checkpoint.
    # The unavailable 70B fine-tuning-control dependency is explicitly recorded.
    module.add_finetune_controls = unavailable_finetune_controls


def run(args: argparse.Namespace) -> dict[str, Any]:
    snapshot, view = args.snapshot_root.resolve(), args.view_root.resolve()
    validate_roots(snapshot, view)
    paths = output_paths(view, args.model, args.layer)
    source = verify_sources(snapshot, view)
    filename = "04_fit_shared_manifold.py" if args.action == "rank-curve" else "04_fit_shared_controls.py"
    module = load_original(snapshot, filename)
    development = validate_development_inputs(view, paths, module.SPECS)
    receipt_path = checked_path(view, f"reports/original/stage_shared_{args.model}_{args.action}.json")
    receipt = {
        "profile": PROFILE, "model": args.model, "layer": args.layer,
        "action": args.action, "state": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source, "development_inputs": development,
        "algorithms": "unchanged_frozen_original_source",
        "held_out_estimator": "leave_one_task_family_out_shared_basis_then_target_readout",
        "satisfies_original_exact_bf16_requirement": False,
        "complete_original_control_battery": False,
    }
    atomic_json(receipt_path, receipt)
    try:
        if args.action == "rank-curve":
            configure_rank_module(module, view, paths, args.model, args.layer)
            module.main()
            label_manifest(paths["preprocessing_manifest"], args.model, source)
        else:
            configure_control_module(module, view, paths, args.model)
            module.parse_args = lambda: SimpleNamespace(action=args.action)
            module.main()
            if args.action == "finalize":
                label_manifest(paths["control_manifest"], args.model, source)
                receipt["core_sro_controls_complete"] = True
        receipt.update(state="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        receipt.update(state="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(receipt_path, receipt)
        raise
    atomic_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--layer", type=int, choices=(60,), default=60)
    parser.add_argument("--action", choices=ACTIONS, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    completed = run(parse_args())
    print(json.dumps({"state": completed["state"], "model": completed["model"], "action": completed["action"]}), flush=True)
