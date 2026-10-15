"""Build isolated NF4 analysis views and freeze existing SRO input lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import sys
from datetime import datetime, timezone

MODELS = ("centaur_70b_nf4", "llama_base_70b_nf4")
PROFILE = "exploratory_nf4_behavioral_replication"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def bounded(path, root):
    path, root = Path(os.path.abspath(path)), Path(root).resolve()
    if (not path.is_relative_to(root) or path == root
            or not path.parent.resolve().is_relative_to(root)):
        raise ValueError(f"Unsafe analysis-view write path: {path}")
    return path


def link(path, target, root):
    path = bounded(path, root)
    target = Path(target).resolve(strict=True)
    if path.is_symlink():
        if path.resolve(strict=True) != target:
            raise ValueError(f"Existing view link has a different target: {path}")
        return
    if path.exists():
        raise ValueError(f"Refusing to replace an existing view entry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=target.is_dir())


def split_contract(splits):
    development = {item["participant"]: item["fold"] for item in splits["development"]}
    confirmation = set(splits["confirmation"])
    if len(development) != 372 or len(confirmation) != 150 or set(development) & confirmation:
        raise ValueError("Frozen development/confirmation split is invalid")
    if set(development.values()) != set(range(5)):
        raise ValueError("Frozen development folds are invalid")
    return development, confirmation


def validate_participants(participants, folds, split, development, confirmation):
    participants = list(map(str, participants))
    if len(participants) != len(folds):
        raise ValueError("Participant/fold lengths differ")
    if split == "development":
        if not participants or not set(participants) <= set(development):
            raise ValueError("Development input contains empty or non-development participants")
        if any(int(f) != development[p] for p, f in zip(participants, folds)):
            raise ValueError("Development input disagrees with frozen participant folds")
    elif split in ("confirmation_t1", "confirmation_t2"):
        if not set(participants) <= confirmation or set(participants) & set(development):
            raise ValueError("Confirmation input contains non-confirmation participants")
    else:
        raise ValueError("Unknown split")


def extraction_inventory(plan):
    if (plan.get("models") != list(MODELS)
            or plan.get("splits") != ["development", "confirmation_t1", "confirmation_t2"]
            or plan.get("primary_layer") != 60):
        raise ValueError("Frozen extraction model/split/layer contract differs")
    records = {}
    for item in plan["input_files"]:
        key = (item["split"], item["task"])
        if key in records or item["rows"] <= 0 or item["sessions"] <= 0:
            raise ValueError("Invalid or duplicate extraction input")
        records[key] = item
    if len(records) != 48 or any(sum(s == split for s, _ in records) != 16 for split in plan["splits"]):
        raise ValueError("Expected 48 frozen task inputs")
    return records


def validate_identity(frame, model, split, task, expected):
    rows = expected["rows"]
    if any(len(values) != rows for values in frame.values()):
        raise ValueError("Metadata columns differ from frozen row count")
    for key, value in [("model", model), ("split", split), ("task", task)]:
        if set(frame[key]) != {value}:
            raise ValueError(f"Metadata {key} differs from frozen block identity")
    if len(set(frame["participant"])) != expected["sessions"]:
        raise ValueError("Metadata participant count differs from frozen task inventory")
    if frame["state_row"] != list(range(rows)):
        raise ValueError("Metadata/state row ordering differs")


def prepare(project_root, source_root, work_root):
    import numpy as np
    import pyarrow.parquet as pq

    project_root = Path(project_root).resolve(strict=True)
    source_root = Path(source_root).resolve(strict=True)
    work_root = Path(work_root).absolute()
    if work_root.exists() and work_root.is_symlink():
        raise ValueError("The work root cannot be a symlink")
    work_root.mkdir(parents=True, exist_ok=True)
    work_root = work_root.resolve()
    snapshot = Path(os.environ.get("CM_NF4_SNAPSHOT_ROOT", source_root / "original_source_snapshot")).resolve()
    inventory = json.loads((snapshot / "source_inventory.json").read_text())
    for relative, expected in inventory.items():
        path = bounded(snapshot / relative, snapshot)
        if sha256(path) != expected:
            raise ValueError(f"Original source snapshot changed: {relative}")
    splits_path = snapshot / "configs/original/splits.json"
    splits = json.loads(splits_path.read_text())
    development, confirmation = split_contract(splits)
    states_root = project_root / "EXTENSIONS/full_nf4_20260914/states"
    extraction = project_root / "EXTENSIONS/full_nf4_20260914"
    plan_path = extraction / "frozen_plan.json"
    plan = json.loads(plan_path.read_text())
    inputs = extraction_inventory(plan)
    extraction_status = json.loads((extraction / "status.json").read_text())
    if extraction_status.get("state") != "complete" or extraction_status.get("completed_trials") != 6268526:
        raise ValueError("The required full NF4 extraction has not completed")
    raw_root = project_root / "DATA_INPUTS/data/raw"
    derived_root = project_root / "DATA_INPUTS/data/derived/original"
    required_raw = [
        raw_root / "Self_Regulation_Ontology/Data/Complete_02-16-2019/taskdata_imputed.csv",
        raw_root / "Self_Regulation_Ontology/Data/Retest_02-16-2019/taskdata_imputed.csv",
        derived_root / "sro_retest_ids.txt",
    ]
    source_hashes = {str(path): sha256(path) for path in required_raw}
    source_hashes[str(plan_path)] = sha256(plan_path)
    blocks = []
    for model in MODELS:
        view = work_root / "views" / model
        view.mkdir(parents=True, exist_ok=True)
        config_root = bounded(view / "configs/original", work_root)
        if config_root.is_symlink():
            raise ValueError("Isolated configurations must be owned copies, not external links")
        config_root.mkdir(parents=True, exist_ok=True)
        for source_config in sorted((snapshot / "configs/original").iterdir()):
            if not source_config.is_file():
                continue
            destination_config = bounded(config_root / source_config.name, work_root)
            if destination_config.exists():
                if sha256(destination_config) != sha256(source_config):
                    raise ValueError(f"Isolated frozen configuration changed: {destination_config}")
            else:
                destination_config.write_bytes(source_config.read_bytes())
            source_hashes[str(destination_config)] = sha256(destination_config)
        link(view / "data/raw", raw_root, work_root)
        link(view / "data/derived/original", derived_root, work_root)
        link(view / "cache/original/8b_states" / model, states_root / model / "development", work_root)
        for timepoint, split in [("T1", "confirmation_t1"), ("T2", "confirmation_t2")]:
            link(view / "cache/original/confirmation_states" / timepoint, states_root / model / split, work_root)
        for split in ["development", "confirmation_t1", "confirmation_t2"]:
            folders = sorted(path for path in (states_root / model / split).iterdir() if path.is_dir())
            if {p.name for p in folders} != {task for s, task in inputs if s == split}:
                raise ValueError(f"Task blocks differ from frozen extraction inventory: {model}/{split}")
            for folder in folders:
                marker_path = folder / "complete.json"
                marker = json.loads(marker_path.read_text())
                expected = inputs[(split, folder.name)]
                if marker["rows"] != expected["rows"] or marker["source_sha256"] != expected["sha256"]:
                    raise ValueError(f"Block lineage differs from frozen extraction inventory: {folder}")
                metadata = folder / "metadata.parquet"
                if sha256(metadata) != marker["files"]["metadata.parquet"]:
                    raise ValueError(f"Metadata checksum mismatch: {folder}")
                frame = pq.read_table(metadata, columns=["participant", "fold", "model", "split", "task", "state_row"]).to_pydict()
                validate_identity(frame, model, split, folder.name, expected)
                validate_participants(frame["participant"], frame["fold"], split, development, confirmation)
                array = np.load(folder / "layer_60.npy", mmap_mode="r", allow_pickle=False)
                shape = array.shape
                if shape != (marker["rows"], 8192) or array.dtype != np.float16 or len(frame["participant"]) != marker["rows"]:
                    raise ValueError(f"Metadata/state alignment mismatch: {folder}")
                del array
                source_hashes[str(marker_path)] = sha256(marker_path)
                blocks.append(dict(model=model, split=split, task=folder.name, rows=marker["rows"],
                                   marker_sha256=source_hashes[str(marker_path)],
                                   metadata_sha256=marker["files"]["metadata.parquet"],
                                   layer60_payload_sha256_declared_by_extractor=marker["files"]["layer_60.npy"],
                                   independent_full_state_payload_rehash=False))
                if len(blocks) % 16 == 0:
                    print("INPUT_BLOCKS_VERIFIED", len(blocks), flush=True)
    report = dict(profile=PROFILE, exact_bf16=False, state="complete", models=list(MODELS),
                  utc=datetime.now(timezone.utc).isoformat(), blocks=blocks,
                  original_source_inventory_sha256=sha256(snapshot / "source_inventory.json"),
                  frozen_extraction_plan_sha256=source_hashes[str(plan_path)],
                  split_sha256=sha256(splits_path), source_hashes=source_hashes,
                  confirmation_outcome_columns_read=False,
                  compatibility_directory="8b_states is a legacy directory name; model identities are NF470B.",
                  psych201=dict(state="deferred", reason="Prespecified 70B subset definition and NF4 states absent"),
                  exact_bf16_original_completed=False)
    destination = source_root / "preflight/input_lineage.json"
    if destination.exists():
        existing = json.loads(destination.read_text())
        for key in ["blocks", "split_sha256", "original_source_inventory_sha256", "source_hashes"]:
            if existing[key] != report[key]:
                raise ValueError("Previously frozen input lineage changed; preserve prior analysis")
        return existing
    atomic_json(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--work-root", required=True)
    args = parser.parse_args()
    report = prepare(args.project_root, args.source_root, args.work_root)
    print(json.dumps({"state": report["state"], "blocks": len(report["blocks"]),
                      "confirmation_outcome_columns_read": report["confirmation_outcome_columns_read"]}), flush=True)


if __name__ == "__main__":
    main()
