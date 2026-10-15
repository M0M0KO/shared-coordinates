#!/usr/bin/env python3
"""Fit official Jacobian lenses and quantify original shared-space causal access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM

from cognitive_manifold.models.choice_extractor import load_local_tokenizer

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/original/main.yaml"
PSYCH101 = ROOT / "data/raw/Psych-101/prompts_training.jsonl"
JLENS_CODE = ROOT / "data/raw/jacobian-lens-code"
CACHE = ROOT / "cache/original/jlens"
PROMPTS = ROOT / "manifests/private/original/jlens_prompts.jsonl"
MANIFEST = ROOT / "manifests/private/original/jlens.json"
RESULT = ROOT / "results/original/jlens_access.parquet"
STATISTICS = ROOT / "results/original/jlens_statistics.json"
NULLS = ROOT / "results/original/jlens_permutation_nulls.npz"
SOURCE_REVISIONS = ROOT / "manifests/public/source_revisions.json"
MODEL_PATHS = {
    "llama_base_8b": ROOT / "models/Llama-3.1-8B",
    "minitaur_8b": ROOT / "models/Llama-3.1-Minitaur-8B",
    "response_shuffled_psych101_8b": ROOT / "models/Psych101-Response-Shuffled-8B",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_path(layer: int) -> Path:
    if layer == 24:
        return ROOT / "results/original/selected_shared.json"
    return ROOT / f"results/original/selected_shared_minitaur_8b_layer_{layer:02d}.json"


def _capacity_balanced_quotas(
    capacities: dict[str, int], total: int
) -> dict[str, int]:
    """Allocate an exact total as evenly as finite task capacities permit."""
    if total < 0:
        raise ValueError("Prompt total must be non-negative")
    if total > sum(capacities.values()):
        raise RuntimeError(
            f"Requested {total} prompts, but only {sum(capacities.values())} are available"
        )
    allocation = {experiment: 0 for experiment in sorted(capacities)}
    remaining = total
    while remaining:
        eligible = [
            experiment
            for experiment in sorted(capacities)
            if allocation[experiment] < capacities[experiment]
        ]
        if not eligible:
            raise RuntimeError("Prompt allocation exhausted all experiment capacities")
        minimum = min(allocation[experiment] for experiment in eligible)
        for experiment in eligible:
            if allocation[experiment] != minimum:
                continue
            allocation[experiment] += 1
            remaining -= 1
            if not remaining:
                break
    return allocation


def freeze_prompts(overwrite: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text())
    count = int(config["jlens"]["psych101_prompts"])
    if PROMPTS.exists() and not overwrite:
        lines = PROMPTS.read_text().splitlines()
        return {"status": "reused", "prompts": len(lines), "sha256": _sha256(PROMPTS)}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with PSYCH101.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            payload = json.loads(line)
            payload["source_line"] = line_index
            groups[str(payload["experiment"])].append(payload)
    experiments = sorted(groups)
    allocations = _capacity_balanced_quotas(
        {experiment: len(groups[experiment]) for experiment in experiments}, count
    )
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        take = allocations[experiment]
        ordered = sorted(
            groups[experiment],
            key=lambda row: hashlib.sha256(
                f"20260821|{experiment}|{row['participant']}|{row['source_line']}".encode()
            ).digest(),
        )
        rows.extend(ordered[:take])
    if len(rows) != count:
        raise RuntimeError(f"Expected {count} J-Lens prompts, selected {len(rows)}")
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS.write_text(
        "".join(
            json.dumps(
                {
                    "experiment": row["experiment"],
                    "source_line": row["source_line"],
                    "text": row["text"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        )
    )
    return {
        "status": "frozen",
        "prompts": len(rows),
        "experiments": len(experiments),
        "minimum_prompts_per_experiment": min(allocations.values()),
        "maximum_prompts_per_experiment": max(allocations.values()),
        "capacity_limited_experiments": {
            experiment: allocation
            for experiment, allocation in allocations.items()
            if allocation == len(groups[experiment])
            and allocation < max(allocations.values())
        },
        "sha256": _sha256(PROMPTS),
    }


def fit(model_name: str) -> dict[str, Any]:
    if not MODEL_PATHS["response_shuffled_psych101_8b"].exists():
        raise RuntimeError("O9 is sealed until the O8 response-shuffled model is complete")
    prompt_evidence = freeze_prompts()
    config = yaml.safe_load(CONFIG.read_text())["jlens"]
    prompts = [json.loads(line)["text"] for line in PROMPTS.read_text().splitlines()]
    sys.path.insert(0, str(JLENS_CODE))
    import jlens

    tokenizer = load_local_tokenizer(MODEL_PATHS[model_name])
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS[model_name],
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    wrapped = jlens.from_hf(hf_model, tokenizer, compile=False, force_bos=True)
    one_based = list(config["source_layers"])
    zero_based = [layer - 1 for layer in one_based]
    if wrapped.n_layers != 32 or zero_based != [7, 15, 23]:
        raise RuntimeError("J-Lens one-based to official zero-based layer mapping failed")
    destination = CACHE / model_name
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "checkpoint.pt"
    lens_path = destination / "lens.pt"
    lens = jlens.fit(
        wrapped,
        prompts=prompts,
        source_layers=zero_based,
        target_layer=wrapped.n_layers - 1,
        dim_batch=int(config["dim_batch"]),
        max_seq_len=int(config["max_sequence_length"]),
        checkpoint_path=str(checkpoint),
        checkpoint_every=10,
        resume=bool(config["checkpoint_and_resume"]),
    )
    lens.save(str(lens_path))
    evidence = {
        "status": "completed",
        "model": model_name,
        "prompts": prompt_evidence,
        "official_source_layers_zero_based": zero_based,
        "proposal_layers_one_based": one_based,
        "target_layer_zero_based": wrapped.n_layers - 1,
        "lens_sha256": _sha256(lens_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "official_code_revision": json.loads(SOURCE_REVISIONS.read_text())["sources"][
            "jacobian_lens"
        ]["revision"],
    }
    del lens, wrapped, hf_model
    torch.cuda.empty_cache()
    return evidence


def _raw_shared_basis(layer: int) -> tuple[np.ndarray, dict[str, Any]]:
    selection = json.loads(_selection_path(layer).read_text())
    preprocessing = np.load(
        ROOT / f"cache/original/preprocessing/minitaur_8b/layer_{layer:02d}/pca.npz"
    )
    shared = np.load(
        ROOT / f"cache/original/shared_final/minitaur_8b/layer_{layer:02d}/shared_basis.npy"
    )
    raw = preprocessing["components"][:512].T @ shared
    return raw.astype(np.float32), selection


def _access(jacobian: torch.Tensor, basis: torch.Tensor) -> float:
    transported = jacobian @ basis
    numerator = transported.square().sum() / basis.shape[1]
    denominator = jacobian.square().sum() / jacobian.shape[1]
    return float((numerator / denominator).cpu())


def analyze() -> None:
    sys.path.insert(0, str(JLENS_CODE))
    import jlens

    config = yaml.safe_load(CONFIG.read_text())
    random_count = int(config["jlens"]["random_subspaces_for_null"])
    rows: list[dict[str, Any]] = []
    for model_name in config["jlens"]["models"]:
        lens_path = CACHE / model_name / "lens.pt"
        if not lens_path.exists():
            raise RuntimeError(f"Missing fitted J-Lens for {model_name}")
        lens = jlens.JacobianLens.load(str(lens_path))
        for one_based in config["jlens"]["source_layers"]:
            basis, selection = _raw_shared_basis(int(one_based))
            basis_tensor = torch.as_tensor(basis, dtype=torch.float32)
            jacobian = lens.jacobians[int(one_based) - 1].float()
            observed = _access(jacobian, basis_tensor)
            rows.append(
                {
                    "model": model_name,
                    "layer": int(one_based),
                    "dimension": basis.shape[1],
                    "subspace": "behavioral",
                    "replicate": -1,
                    "causal_access": observed,
                    "selected_shared_config_hash": selection["config_hash"],
                }
            )
            generator = torch.Generator().manual_seed(
                int(config["execution"]["seed"]) + int(one_based)
            )
            for replicate in range(random_count):
                random = torch.randn(jacobian.shape[1], basis.shape[1], generator=generator)
                random = torch.linalg.qr(random, mode="reduced").Q
                rows.append(
                    {
                        "model": model_name,
                        "layer": int(one_based),
                        "dimension": basis.shape[1],
                        "subspace": "haar_random",
                        "replicate": replicate,
                        "causal_access": _access(jacobian, random),
                        "selected_shared_config_hash": selection["config_hash"],
                    }
                )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)
    output.to_parquet(RESULT, index=False, compression="zstd")
    permutations = int(config["statistics"]["permutation_replicates"])
    seed = int(config["execution"]["seed"])
    tests: dict[str, Any] = {}
    nulls: dict[str, np.ndarray] = {}
    for (model_name, layer), frame in output.groupby(["model", "layer"]):
        observed = float(frame.loc[frame["subspace"] == "behavioral", "causal_access"].iloc[0])
        random = frame.loc[frame["subspace"] == "haar_random"].sort_values("replicate")
        random_values = random["causal_access"].to_numpy(np.float64)
        tests[f"{model_name}_layer_{int(layer)}_behavioral_vs_haar"] = {
            "estimate": observed - float(random_values.mean()),
            "observed_causal_access": observed,
            "haar_mean": float(random_values.mean()),
            "empirical_one_sided_p": float(
                (1 + np.count_nonzero(random_values >= observed)) / (len(random_values) + 1)
            ),
            "unique_haar_subspaces": len(random_values),
        }
    models = ("llama_base_8b", "response_shuffled_psych101_8b")
    rng = np.random.default_rng(seed + 1901)
    for layer in map(int, config["jlens"]["source_layers"]):
        minitaur = output.loc[(output["model"] == "minitaur_8b") & (output["layer"] == layer)]
        minitaur_observed = float(
            minitaur.loc[minitaur["subspace"] == "behavioral", "causal_access"].iloc[0]
        )
        minitaur_random = (
            minitaur.loc[minitaur["subspace"] == "haar_random"]
            .sort_values("replicate")["causal_access"]
            .to_numpy(np.float64)
        )
        for comparison in models:
            other = output.loc[(output["model"] == comparison) & (output["layer"] == layer)]
            other_observed = float(
                other.loc[other["subspace"] == "behavioral", "causal_access"].iloc[0]
            )
            other_random = (
                other.loc[other["subspace"] == "haar_random"]
                .sort_values("replicate")["causal_access"]
                .to_numpy(np.float64)
            )
            if len(minitaur_random) != len(other_random):
                raise RuntimeError("Matched J-Lens Haar nulls have unequal replicate counts")
            unique_null = minitaur_random - other_random
            sampled = unique_null[rng.integers(0, len(unique_null), size=permutations)]
            observed_difference = minitaur_observed - other_observed
            name = f"minitaur_minus_{comparison}_layer_{layer}"
            tests[name] = {
                "estimate": observed_difference,
                "one_sided_random_subspace_permutation_p": float(
                    (1 + np.count_nonzero(sampled >= observed_difference)) / (permutations + 1)
                ),
                "permutations": permutations,
                "unique_matched_haar_differences": len(unique_null),
            }
            nulls[name] = sampled
    np.savez(NULLS, **nulls)
    statistics = {
        "schema_version": 1,
        "profile": "original",
        "permutation_replicates": permutations,
        "random_subspaces_per_model_layer": random_count,
        "tests": tests,
        "nulls_sha256": _sha256(NULLS),
    }
    STATISTICS.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(RESULT), "rows": len(rows), "statistics": statistics}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze-prompts", "fit", "analyze"))
    parser.add_argument("--model", choices=(*MODEL_PATHS, "all"), default="all")
    parser.add_argument("--overwrite-prompts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"completed": {}}
    if args.action == "freeze-prompts":
        manifest["prompt_freeze"] = freeze_prompts(args.overwrite_prompts)
    elif args.action == "fit":
        models = tuple(MODEL_PATHS) if args.model == "all" else (args.model,)
        for model_name in models:
            manifest["completed"][model_name] = fit(model_name)
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        analyze()
        manifest["analysis"] = {"result_sha256": _sha256(RESULT)}
    manifest.update({"schema_version": 1, "profile": "original"})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
