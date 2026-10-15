#!/usr/bin/env python3
"""Resumable model-task extraction for original-profile 8B discovery models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from numpy.lib.format import open_memmap

from cognitive_manifold.models.choice_extractor import FrozenChoiceExtractor
from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/original/main.yaml"
CONTEXT_AUDIT = ROOT / "reports/original/context_length_audit.json"
SESSION_ROOT = ROOT / "data/derived/original/sro_sessions/development"
OUTPUT_ROOT = ROOT / "cache/original/8b_states"
MANIFEST = ROOT / "manifests/private/original/8b_state_extraction.json"
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
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"schema_version": 1, "profile": "original", "completed": {}}


def _save_manifest(payload: dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(MANIFEST)


def extract_task(
    extractor: FrozenChoiceExtractor,
    model_name: str,
    task: str,
    layers: list[int],
    model_revision: str,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    source = SESSION_ROOT / f"{task}.parquet"
    destination = OUTPUT_ROOT / model_name / task
    metadata_path = destination / "metadata.parquet"
    if metadata_path.exists() and not overwrite:
        metadata = pd.read_parquet(metadata_path)
        return {
            "status": "reused",
            "rows": len(metadata),
            "metadata_sha256": _sha256(metadata_path),
        }
    frame = pd.read_parquet(source)
    frame = (
        frame.loc[frame["validity"] == "primary"]
        .sort_values(["participant", "trial"], kind="stable")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise RuntimeError(f"No primary records for {task}")
    destination.mkdir(parents=True, exist_ok=True)
    split_payload = json.loads((ROOT / "configs/original/splits.json").read_text())
    folds = {item["participant"]: int(item["fold"]) for item in split_payload["development"]}
    hidden_size = int(extractor.model.config.hidden_size)
    arrays: dict[int, np.memmap] = {}
    partials: dict[int, Path] = {}
    for layer in layers:
        partial = destination / f"layer_{layer:02d}.partial.npy"
        partials[layer] = partial
        arrays[layer] = open_memmap(
            partial,
            mode="w+",
            dtype=np.float16,
            shape=(len(frame), hidden_size),
        )
    output_rows: list[dict[str, Any]] = []
    maximum_tokens = 0
    cache_reused = 0
    cache_crop_events = 0
    cache_prefix_tokens_reused = 0
    full_recomputations = 0
    completed = 0
    next_report = 1000
    for _participant, session in frame.groupby("participant", sort=False):
        results, cache_evidence = extractor.extract_session(
            session["prompt_prefix"].tolist(),
            session["candidate_responses"].tolist(),
            session["response"].tolist(),
        )
        cache_reused += cache_evidence["cache_reused"]
        cache_crop_events += cache_evidence["cache_crop_events"]
        cache_prefix_tokens_reused += cache_evidence["cache_prefix_tokens_reused"]
        full_recomputations += cache_evidence["full_recomputations"]
        for row, result in zip(session.itertuples(index=False), results, strict=True):
            index = completed
            maximum_tokens = max(maximum_tokens, result.token_count)
            for layer, state in result.hidden_states.items():
                arrays[layer][index] = state.numpy().astype(np.float16, copy=False)
            output_rows.append(
                {
                    "dataset": row.dataset,
                    "participant": row.participant,
                    "timepoint": row.timepoint,
                    "task": row.task,
                    "task_family": row.task_family,
                    "trial": row.trial,
                    "fold": folds[row.participant],
                    "model": model_name,
                    "model_revision": model_revision,
                    "config_hash": row.config_hash,
                    "dataset_revision": row.dataset_revision,
                    "response": row.response,
                    "candidate_responses": row.candidate_responses,
                    "candidate_logprobs": extractor.encode_scores(result.candidate_logprobs),
                    "response_logprob": result.response_logprob,
                    "nll": result.nll,
                    "token_count": result.token_count,
                    "state_row": index,
                }
            )
            completed += 1
        if completed >= next_report or completed == len(frame):
            print(
                json.dumps({"model": model_name, "task": task, "completed": completed}),
                flush=True,
            )
            next_report = completed + 1000
    for array in arrays.values():
        array.flush()
    for layer, partial in partials.items():
        partial.replace(destination / f"layer_{layer:02d}.npy")
    metadata = pd.DataFrame(output_rows)
    metadata.to_parquet(metadata_path, index=False, compression="zstd")
    return {
        "status": "completed",
        "rows": len(metadata),
        "max_tokens": maximum_tokens,
        "cache_reused": cache_reused,
        "cache_crop_events": cache_crop_events,
        "cache_prefix_tokens_reused": cache_prefix_tokens_reused,
        "full_recomputations": full_recomputations,
        "metadata_sha256": _sha256(metadata_path),
        "layer_files": {
            str(layer): _sha256(destination / f"layer_{layer:02d}.npy") for layer in layers
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(*MODEL_PATHS, "all"), default="all")
    parser.add_argument("--task", choices=(*SPECS, "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _workload_order() -> tuple[str, ...]:
    counts: list[tuple[int, str]] = []
    for task in SPECS:
        validity = pd.read_parquet(SESSION_ROOT / f"{task}.parquet", columns=["validity"])
        counts.append((int((validity["validity"] == "primary").sum()), task))
    return tuple(task for _count, task in sorted(counts))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if not config["execution"]["runnable"]:
        raise RuntimeError("Original profile is not runnable; O1/O2 gates must be frozen first")
    if not CONTEXT_AUDIT.exists():
        raise RuntimeError("Run 02_audit_context_lengths.py before state extraction")
    context_audit = json.loads(CONTEXT_AUDIT.read_text())
    if not context_audit["all_full_history_prefixes_fit"]:
        raise RuntimeError("Full-history SRO prefixes exceed the frozen model context")
    if set(context_audit["response_label_token_counts"].values()) != {1}:
        raise RuntimeError("Original response labels are not all single-token candidates")
    layers = config["likelihood_and_representation"]["layers_8b"]
    models = MODEL_PATHS if args.model == "all" else (args.model,)
    tasks = _workload_order() if args.task == "all" else (args.task,)
    manifest = _load_manifest()
    for model_name in models:
        extractor = FrozenChoiceExtractor(MODEL_PATHS[model_name], layers)
        manifest.setdefault("model_identity", {})[model_name] = extractor.identity()
        for task in tasks:
            result = extract_task(
                extractor,
                model_name,
                task,
                layers,
                MODEL_REVISIONS[model_name],
                overwrite=args.overwrite,
            )
            manifest["completed"][f"{model_name}/{task}"] = result
            _save_manifest(manifest)
        del extractor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
