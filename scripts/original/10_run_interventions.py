#!/usr/bin/env python3
"""Run original projection-ablation and coordinate-swap residual interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from cognitive_manifold.models.residual_intervention import LlamaResidualSuffix
from cognitive_manifold.prompts.sro.specs import SPECS
from cognitive_manifold.statistics.inference import participant_sign_permutation

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/original/main.yaml"
INTERVENTION_CONFIG = ROOT / "configs/original/intervention_frozen.yaml"
SELECTION = ROOT / "results/original/selected_shared.json"
JLENS_RESULT = ROOT / "results/original/jlens_access.parquet"
MODEL_PATH = ROOT / "models/Llama-3.1-Minitaur-8B"
SESSION_ROOT = ROOT / "data/derived/original/sro_sessions/development"
STATE_ROOT = ROOT / "cache/original/8b_states/minitaur_8b"
CACHE = ROOT / "cache/original/interventions"
RESULT = ROOT / "results/original/causal_interventions.parquet"
CONTRASTS = ROOT / "results/original/causal_contrasts.parquet"
STATISTICS = ROOT / "results/original/causal_statistics.json"
NULLS = ROOT / "results/original/causal_permutation_nulls.npz"
MANIFEST = ROOT / "manifests/private/original/interventions.json"
OPTIONS_PATTERN = re.compile(r"OPTIONS\[(.*)\]\n<<$")
INTERVENTION_MODEL_DTYPE = torch.float32
DOWNSTREAM_CACHE_STRATEGY = "capture_downstream_kv_during_full_forward_then_crop_final_token"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_order(*values: object) -> bytes:
    payload = "|".join(map(str, values)).encode()
    return hashlib.sha256(payload).digest()


def _current_option_semantics(prompt_prefix: str) -> dict[str, str]:
    match = OPTIONS_PATTERN.search(prompt_prefix)
    if match is None:
        raise RuntimeError("Intervention prefix has no final OPTIONS mapping")
    mapping: dict[str, str] = {}
    for item in match.group(1).split(","):
        label, separator, semantic = item.partition("=")
        if not separator or not label or not semantic or label in mapping:
            raise RuntimeError(f"Invalid intervention OPTIONS entry: {item!r}")
        mapping[label] = semantic
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("Intervention OPTIONS semantics are not one-to-one")
    return mapping


def _recipient_label_for_donor_semantic(
    recipient_options: dict[str, str],
    donor_options: dict[str, str],
    donor_label: str,
) -> str | None:
    semantic = donor_options.get(donor_label)
    if semantic is None:
        raise RuntimeError("Donor response is absent from its OPTIONS mapping")
    return next(
        (label for label, value in recipient_options.items() if value == semantic),
        None,
    )


def _candidate_logprobs(
    logits: torch.Tensor,
    tokenizer: Any,
    candidates: list[str],
) -> dict[str, float]:
    ids = [tokenizer(value, add_special_tokens=False).input_ids for value in candidates]
    if any(len(value) != 1 for value in ids):
        raise RuntimeError("Intervention response labels must all be single-token")
    first = torch.log_softmax(logits.float(), dim=-1)[0]
    selected = torch.stack([first[value[0]] for value in ids])
    normalized = torch.log_softmax(selected.to(torch.float64), dim=0).cpu()
    return {key: float(value) for key, value in zip(candidates, normalized, strict=True)}


def _select_recipients(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    classes = sorted(frame["response"].unique())
    per_class = int(np.ceil(count / len(classes)))
    selected: list[int] = []
    for response in classes:
        indices = frame.index[frame["response"] == response].tolist()
        indices.sort(
            key=lambda index: _hash_order(
                seed,
                frame.at[index, "participant"],
                frame.at[index, "trial"],
                response,
            )
        )
        selected.extend(indices[:per_class])
    selected.sort(
        key=lambda index: _hash_order(
            seed, frame.at[index, "participant"], frame.at[index, "trial"]
        )
    )
    return frame.loc[selected[:count]].copy()


def _haar_subspaces(width: int, dimension: int, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    bases: list[torch.Tensor] = []
    for _ in range(count):
        sample = torch.randn(width, dimension, generator=generator, dtype=torch.float32)
        bases.append(torch.linalg.qr(sample.to("cuda:0"), mode="reduced").Q)
    return torch.stack(bases)


def _task_frame(task: str, layer: int) -> tuple[pd.DataFrame, np.ndarray]:
    sessions = pd.read_parquet(SESSION_ROOT / f"{task}.parquet")
    sessions = sessions.loc[sessions["validity"] == "primary"].copy()
    metadata = pd.read_parquet(STATE_ROOT / task / "metadata.parquet")
    joined = sessions.merge(
        metadata[["participant", "trial", "state_row", "response"]],
        on=["participant", "trial", "response"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(metadata):
        raise RuntimeError(f"Session/state join failed for {task}")
    joined["option_semantics"] = joined["prompt_prefix"].map(_current_option_semantics)
    joined["response_semantic"] = [
        options[response]
        for options, response in zip(joined["option_semantics"], joined["response"], strict=True)
    ]
    states = np.load(STATE_ROOT / task / f"layer_{layer:02d}.npy", mmap_mode="r")
    return joined, states


def run_task(
    task: str,
    tokenizer: Any,
    runner: LlamaResidualSuffix,
    mean: torch.Tensor,
    basis: torch.Tensor,
    random_bases: torch.Tensor,
    config: dict[str, Any],
    intervention: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    destination = CACHE / f"{task}.parquet"
    if destination.exists() and not overwrite:
        frame = pd.read_parquet(destination)
        return {
            "status": "reused",
            "task": task,
            "rows": len(frame),
            "model_dtype": str(next(runner.model.parameters()).dtype),
            "downstream_cache_strategy": DOWNSTREAM_CACHE_STRATEGY,
            "sha256": _sha256(destination),
        }
    layer = int(json.loads(SELECTION.read_text())["layer"])
    frame, cached_states = _task_frame(task, layer)
    recipient_count = int(intervention["ablation"]["recipients_per_task"])
    recipients = _select_recipients(
        frame,
        recipient_count,
        int(intervention["execution"]["seed"]),
    )
    random_count = int(intervention["ablation"]["haar_random_subspaces"])
    strengths = list(intervention["ablation"]["strengths"])
    thresholds = config["intervention"]["sufficiently_large_task"]
    counts = frame["response"].value_counts()
    sufficiently_large = bool(
        len(frame) >= int(thresholds["minimum_valid_recipient_trials"])
        and len(counts) >= int(thresholds["minimum_response_classes"])
        and counts.min() >= int(thresholds["minimum_trials_per_response_class"])
    )
    swap_pairs = (
        int(intervention["swap"]["pairs_per_sufficiently_large_task"]) if sufficiently_large else 0
    )
    pairs_per_recipient = int(np.ceil(swap_pairs / len(recipients))) if swap_pairs else 0
    rows: list[dict[str, Any]] = []
    maximum_baseline_difference = 0.0
    completed_pairs = 0
    for recipient_index, recipient in enumerate(recipients.itertuples(index=False)):
        encoded = tokenizer(
            recipient.prompt_prefix,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids.to("cuda:0")
        source_sequence, full_logits, full_forward_cache = runner.capture(encoded)
        prefix_length = source_sequence.shape[1] - 1
        downstream_cache = runner.build_cache(source_sequence, full_forward_cache)
        original = source_sequence[0, -1, :]
        suffix_logits = runner.logits_from_source_token(
            original,
            downstream_cache,
            prefix_length,
        )
        candidates = list(recipient.candidate_responses)
        full_scores = _candidate_logprobs(full_logits, tokenizer, candidates)
        suffix_scores = _candidate_logprobs(suffix_logits, tokenizer, candidates)
        baseline_difference = max(
            abs(full_scores[value] - suffix_scores[value]) for value in candidates
        )
        maximum_baseline_difference = max(maximum_baseline_difference, baseline_difference)
        baseline_nll = -full_scores[recipient.response]
        common = {
            "task": task,
            "task_family": SPECS[task].family,
            "participant": recipient.participant,
            "trial": recipient.trial,
            "recipient_response": recipient.response,
            "recipient_response_semantic": recipient.response_semantic,
            "baseline_nll": baseline_nll,
            "source_layer": layer,
            "intervention_config_sha256": _sha256(INTERVENTION_CONFIG),
        }
        rows.append(
            {
                **common,
                "analysis": "baseline",
                "strength": 0.0,
                "subspace": "none",
                "replicate": -1,
                "intervened_nll": baseline_nll,
                "delta_nll": 0.0,
                "donor_response": None,
                "donor_source_label": None,
                "donor_response_semantic": None,
                "donor_logprob_shift": None,
            }
        )
        centered = original.float() - mean
        behavioral_component = basis @ (basis.T @ centered)
        for strength in strengths:
            modified = (original.float() - float(strength) * behavioral_component).to(
                original.dtype
            )
            logits = runner.logits_from_source_token(
                modified,
                downstream_cache,
                prefix_length,
            )
            score = _candidate_logprobs(logits, tokenizer, candidates)
            nll = -score[recipient.response]
            rows.append(
                {
                    **common,
                    "analysis": "projection_ablation",
                    "strength": float(strength),
                    "subspace": "behavioral",
                    "replicate": -1,
                    "intervened_nll": nll,
                    "delta_nll": nll - baseline_nll,
                    "donor_response": None,
                    "donor_source_label": None,
                    "donor_response_semantic": None,
                    "donor_logprob_shift": None,
                }
            )
            for replicate in range(random_count):
                random = random_bases[replicate]
                random_component = random @ (random.T @ centered)
                modified = (original.float() - float(strength) * random_component).to(
                    original.dtype
                )
                logits = runner.logits_from_source_token(
                    modified,
                    downstream_cache,
                    prefix_length,
                )
                score = _candidate_logprobs(logits, tokenizer, candidates)
                nll = -score[recipient.response]
                rows.append(
                    {
                        **common,
                        "analysis": "projection_ablation",
                        "strength": float(strength),
                        "subspace": "haar_random",
                        "replicate": replicate,
                        "intervened_nll": nll,
                        "delta_nll": nll - baseline_nll,
                        "donor_response": None,
                        "donor_source_label": None,
                        "donor_response_semantic": None,
                        "donor_logprob_shift": None,
                    }
                )
        if completed_pairs < swap_pairs:
            recipient_options = recipient.option_semantics
            recipient_semantics = set(recipient_options.values())
            donor_pool = frame.loc[
                frame["response_semantic"].isin(recipient_semantics)
                & (frame["response_semantic"] != recipient.response_semantic)
                & (
                    (frame["participant"] != recipient.participant)
                    | (frame["trial"] != recipient.trial)
                )
            ].copy()
            donor_pool["_order"] = [
                _hash_order(
                    intervention["execution"]["seed"],
                    task,
                    recipient.participant,
                    recipient.trial,
                    value.participant,
                    value.trial,
                )
                for value in donor_pool.itertuples(index=False)
            ]
            donor_pool = donor_pool.sort_values("_order").head(pairs_per_recipient)
            for donor in donor_pool.itertuples(index=False):
                if completed_pairs >= swap_pairs:
                    break
                donor_state = torch.as_tensor(
                    np.array(cached_states[int(donor.state_row)], copy=True),
                    dtype=torch.float32,
                    device="cuda:0",
                )
                donor_centered = donor_state - mean
                swapped = (
                    original.float() - behavioral_component + basis @ (basis.T @ donor_centered)
                ).to(original.dtype)
                logits = runner.logits_from_source_token(
                    swapped,
                    downstream_cache,
                    prefix_length,
                )
                score = _candidate_logprobs(logits, tokenizer, candidates)
                donor_label = _recipient_label_for_donor_semantic(
                    recipient_options,
                    donor.option_semantics,
                    donor.response,
                )
                if donor_label is None:
                    raise RuntimeError("Filtered donor semantic is absent from recipient OPTIONS")
                shift = score[donor_label] - full_scores[donor_label]
                rows.append(
                    {
                        **common,
                        "analysis": "coordinate_swap",
                        "strength": 1.0,
                        "subspace": "behavioral",
                        "replicate": completed_pairs,
                        "intervened_nll": -score[recipient.response],
                        "delta_nll": -score[recipient.response] - baseline_nll,
                        "donor_response": donor_label,
                        "donor_source_label": donor.response,
                        "donor_response_semantic": donor.response_semantic,
                        "donor_logprob_shift": shift,
                    }
                )
                random = random_bases[completed_pairs % random_count]
                random_swapped = (
                    original.float()
                    - random @ (random.T @ centered)
                    + random @ (random.T @ donor_centered)
                ).to(original.dtype)
                logits = runner.logits_from_source_token(
                    random_swapped,
                    downstream_cache,
                    prefix_length,
                )
                score = _candidate_logprobs(logits, tokenizer, candidates)
                rows.append(
                    {
                        **common,
                        "analysis": "coordinate_swap",
                        "strength": 1.0,
                        "subspace": "haar_random",
                        "replicate": completed_pairs,
                        "intervened_nll": -score[recipient.response],
                        "delta_nll": -score[recipient.response] - baseline_nll,
                        "donor_response": donor_label,
                        "donor_source_label": donor.response,
                        "donor_response_semantic": donor.response_semantic,
                        "donor_logprob_shift": score[donor_label] - full_scores[donor_label],
                    }
                )
                completed_pairs += 1
        print(
            json.dumps(
                {
                    "task": task,
                    "recipient": recipient_index + 1,
                    "of": len(recipients),
                    "swap_pairs": completed_pairs,
                }
            ),
            flush=True,
        )
        del source_sequence, downstream_cache
        torch.cuda.empty_cache()
    if completed_pairs != swap_pairs:
        raise RuntimeError(f"Expected {swap_pairs} swaps for {task}, built {completed_pairs}")
    if maximum_baseline_difference > 0.1:
        raise RuntimeError(
            f"Downstream-only baseline differs from full forward by {maximum_baseline_difference}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)
    output.to_parquet(destination, index=False, compression="zstd")
    mapping_variants = {tuple(sorted(options.items())) for options in frame["option_semantics"]}
    return {
        "status": "completed",
        "task": task,
        "rows": len(output),
        "recipients": len(recipients),
        "swap_pairs": completed_pairs,
        "response_label_mapping_variants": len(mapping_variants),
        "swap_labels_translated_by_semantic_action": True,
        "maximum_baseline_logprob_difference": maximum_baseline_difference,
        "model_dtype": str(next(runner.model.parameters()).dtype),
        "downstream_cache_strategy": DOWNSTREAM_CACHE_STRATEGY,
        "sha256": _sha256(destination),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "finalize"))
    parser.add_argument("--task", choices=(*SPECS, "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _finalize_statistics(output: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    keys = ["participant", "task", "trial", "strength"]
    ablation = output.loc[output["analysis"] == "projection_ablation"]
    behavioral = ablation.loc[ablation["subspace"] == "behavioral", [*keys, "delta_nll"]]
    behavioral = behavioral.rename(columns={"delta_nll": "behavioral_delta_nll"})
    random = (
        ablation.loc[ablation["subspace"] == "haar_random"]
        .groupby(keys, as_index=False)["delta_nll"]
        .mean()
        .rename(columns={"delta_nll": "random_mean_delta_nll"})
    )
    ablation_contrast = behavioral.merge(random, on=keys, validate="one_to_one")
    ablation_contrast["contrast"] = (
        ablation_contrast["behavioral_delta_nll"] - ablation_contrast["random_mean_delta_nll"]
    )
    ablation_contrast["contrast_type"] = "projection_ablation_behavioral_minus_random"

    swap_keys = ["participant", "task", "trial", "replicate", "donor_response"]
    swaps = output.loc[output["analysis"] == "coordinate_swap"]
    swap_behavioral = swaps.loc[
        swaps["subspace"] == "behavioral", [*swap_keys, "donor_logprob_shift"]
    ].rename(columns={"donor_logprob_shift": "behavioral_donor_shift"})
    swap_random = swaps.loc[
        swaps["subspace"] == "haar_random", [*swap_keys, "donor_logprob_shift"]
    ].rename(columns={"donor_logprob_shift": "random_donor_shift"})
    swap_contrast = swap_behavioral.merge(swap_random, on=swap_keys, validate="one_to_one")
    swap_contrast["strength"] = 1.0
    swap_contrast["contrast"] = (
        swap_contrast["behavioral_donor_shift"] - swap_contrast["random_donor_shift"]
    )
    swap_contrast["contrast_type"] = "coordinate_swap_behavioral_minus_random"

    contrasts = pd.concat([ablation_contrast, swap_contrast], ignore_index=True, sort=False)
    contrasts.to_parquet(CONTRASTS, index=False, compression="zstd")
    replicates = int(config["statistics"]["permutation_replicates"])
    seed = int(config["execution"]["seed"])
    summaries: dict[str, Any] = {}
    nulls: dict[str, np.ndarray] = {}
    for index, (strength, frame) in enumerate(ablation_contrast.groupby("strength")):
        name = f"projection_ablation_strength_{float(strength):.2f}"
        observed, p_value, null = participant_sign_permutation(
            frame,
            "contrast",
            replicates=replicates,
            seed=seed + 2001 + index,
        )
        summaries[name] = {
            "estimate": observed,
            "one_sided_participant_permutation_p": p_value,
            "participant_clusters": int(frame["participant"].nunique()),
            "task_clusters": int(frame["task"].nunique()),
            "cells": len(frame),
        }
        nulls[name] = null
    if not swap_contrast.empty:
        observed, p_value, null = participant_sign_permutation(
            swap_contrast,
            "contrast",
            replicates=replicates,
            seed=seed + 2010,
        )
        summaries["coordinate_swap"] = {
            "estimate": observed,
            "one_sided_participant_permutation_p": p_value,
            "participant_clusters": int(swap_contrast["participant"].nunique()),
            "task_clusters": int(swap_contrast["task"].nunique()),
            "pairs": len(swap_contrast),
        }
        nulls["coordinate_swap"] = null
    np.savez(NULLS, **nulls)
    payload = {
        "schema_version": 1,
        "profile": "original",
        "null": "matched_Haar_random_subspace_then_participant_cluster_sign_permutation",
        "permutation_replicates": replicates,
        "contrasts_sha256": _sha256(CONTRASTS),
        "permutation_nulls_sha256": _sha256(NULLS),
        "tests": summaries,
    }
    STATISTICS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    args = parse_args()
    if not JLENS_RESULT.exists():
        raise RuntimeError("O10 is sealed until O9 J-Lens results are frozen")
    if args.action == "finalize":
        missing = [task for task in SPECS if not (CACHE / f"{task}.parquet").exists()]
        if missing:
            raise RuntimeError(f"Intervention tasks are incomplete: {missing}")
        frames = [pd.read_parquet(CACHE / f"{task}.parquet") for task in SPECS]
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        output = pd.concat(frames, ignore_index=True)
        output.to_parquet(RESULT, index=False, compression="zstd")
        statistics = _finalize_statistics(output, yaml.safe_load(CONFIG.read_text()))
        print(
            json.dumps({"result": str(RESULT), "sha256": _sha256(RESULT), "statistics": statistics})
        )
        return
    config = yaml.safe_load(CONFIG.read_text())
    intervention = yaml.safe_load(INTERVENTION_CONFIG.read_text())
    selection = json.loads(SELECTION.read_text())
    layer = int(selection["layer"])
    dimension = int(selection["dimension"])
    preprocessing = np.load(
        ROOT / f"cache/original/preprocessing/minitaur_8b/layer_{layer:02d}/pca.npz"
    )
    shared = np.load(
        ROOT / f"cache/original/shared_final/minitaur_8b/layer_{layer:02d}/shared_basis.npy"
    )
    basis = torch.as_tensor(
        preprocessing["components"][:512].T @ shared,
        dtype=torch.float32,
        device="cuda:0",
    )
    pca_center = (
        preprocessing["pca_center"]
        if "pca_center" in preprocessing.files
        else np.zeros(preprocessing["components"].shape[1], dtype=np.float32)
    )
    mean = torch.as_tensor(
        preprocessing["mean"] + pca_center,
        dtype=torch.float32,
        device="cuda:0",
    )
    random_bases = _haar_subspaces(
        basis.shape[0],
        dimension,
        int(intervention["ablation"]["haar_random_subspaces"]),
        int(intervention["execution"]["seed"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        dtype=INTERVENTION_MODEL_DTYPE,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    runner = LlamaResidualSuffix(
        model,
        layer,
        int(intervention["execution"]["downstream_chunk_size"]),
    )
    tasks = tuple(SPECS) if args.task == "all" else (args.task,)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"tasks": {}}
    for task in tasks:
        evidence = run_task(
            task,
            tokenizer,
            runner,
            mean,
            basis,
            random_bases,
            config,
            intervention,
            args.overwrite,
        )
        manifest["tasks"][task] = evidence
        manifest.update(
            {
                "schema_version": 1,
                "profile": "original",
                "intervention_config_sha256": _sha256(INTERVENTION_CONFIG),
                "model_dtype": str(next(model.parameters()).dtype),
                "downstream_cache_strategy": DOWNSTREAM_CACHE_STRATEGY,
            }
        )
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
