#!/usr/bin/env python3
"""Nested development-only rank selection for low-dimensional individuality."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from cognitive_manifold.data.splits import participant_pseudonym
from cognitive_manifold.individuality.baselines import (
    evaluate_fixed_feature_readout,
    fit_five_factor_scores,
    fit_fixed_feature_readout,
)
from cognitive_manifold.individuality.model import (
    evaluate_new_coordinates,
    fit_individual_population,
    infer_new_coordinates,
    load_individual_task_data,
)
from cognitive_manifold.individuality.recovery import orthogonal_procrustes_mapping
from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/original/main.yaml"
SELECTION = ROOT / "results/original/selected_shared.json"
PRIVATE_SELECTION = ROOT / "results/original/selected_private.json"
PRIVATE_RANK = ROOT / "results/original/private_rank.parquet"
CROSS_DOMAIN = ROOT / "results/original/cross_domain.parquet"
CACHE = ROOT / "cache/original"
BASELINE_CONFIG = ROOT / "configs/original/baselines_frozen.yaml"
PRIVATE_IDS = ROOT / "data/derived/original/sro_retest_ids.txt"
FACTOR_MODEL = CACHE / "original_baselines/factor_readout.pt"
FACTOR_SCORES = ROOT / "manifests/private/original/sro_five_factor_scores.npz"
RECOVERY_ROOT = CACHE / "original_coordinate_recovery"
RECOVERY_RESULT = ROOT / "results/original/coordinate_recovery_development.parquet"
BASELINE_MANIFEST = ROOT / "manifests/private/original/baselines.json"
SRO_ROOT = ROOT / "data/raw/Self_Regulation_Ontology"
RELEASED_FACTOR_CODE = (
    "dimensional_structure/generate_results.py",
    "dimensional_structure/EFA_test_retest.py",
    "dimensional_structure/results.py",
    "selfregulation/utils/result_utils.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mode(values: list[float]) -> float:
    counts = Counter(values)
    return min(value for value, count in counts.items() if count == max(counts.values()))


def _released_factor_audit(source_root: Path = SRO_ROOT) -> dict[str, Any]:
    code = {name: source_root / name for name in RELEASED_FACTOR_CODE}
    missing_code = sorted(name for name, path in code.items() if not path.is_file())
    saved_objects = sorted((source_root / "Results").rglob("*results.pkl"))
    if missing_code:
        status = "released_factor_source_code_incomplete"
    elif saved_objects:
        status = "released_fitted_objects_present_but_not_silently_substituted"
    else:
        status = "released_source_code_present_without_saved_fitted_objects"
    return {
        "status": status,
        "source_code_sha256": {
            name: _sha256(path) for name, path in code.items() if path.is_file()
        },
        "missing_source_code": missing_code,
        "saved_result_objects": [str(path.relative_to(source_root)) for path in saved_objects],
        "exact_released_reconstruction_completed": False,
        "development_only_refit_completed": True,
        "boundary": (
            "The frozen public repository contains the released analysis source but no fitted "
            "task-factor result object. The original-profile mandatory comparison therefore "
            "uses the separately specified development-only five-factor refit and does not "
            "mislabel it as an exact recovered released solution."
        ),
    }


def _load_population(
    checkpoint: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, tuple[str, ...]]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    weights = {
        key.split(".", 1)[1]: value for key, value in state.items() if key.startswith("weights.")
    }
    biases = {
        key.split(".", 1)[1]: value for key, value in state.items() if key.startswith("biases.")
    }
    classes = {task: tuple(values) for task, values in payload["classes"].items()}
    return weights, biases, classes


def _raw_split_ids() -> tuple[set[str], set[str]]:
    confirmation = {line.strip() for line in PRIVATE_IDS.read_text().splitlines() if line.strip()}
    complete = pd.read_csv(
        ROOT / "data/raw/Self_Regulation_Ontology/Data/Complete_02-16-2019/taskdata_imputed.csv",
        usecols=[0],
    ).iloc[:, 0]
    complete_ids = set(complete.astype(str).str.strip())
    development = complete_ids - confirmation
    if len(development) != 372 or len(confirmation) != 150:
        raise RuntimeError(
            f"Unexpected factor split: development={len(development)}, "
            f"confirmation={len(confirmation)}"
        )
    return development, confirmation


def _save_auxiliary_original_models(
    config: dict[str, Any],
    data: dict[str, Any],
    participant_names: list[str],
    weights: dict[str, torch.Tensor],
    biases: dict[str, torch.Tensor],
    selected_rank: int,
    selected_lr: float,
    selected_regularization: float,
) -> list[dict[str, Any]]:
    baseline = yaml.safe_load(BASELINE_CONFIG.read_text())
    development_ids, confirmation_ids = _raw_split_ids()
    factor_config = baseline["sro_factor"]
    scores = fit_five_factor_scores(
        ROOT / factor_config["source_complete"],
        ROOT / factor_config["source_retest"],
        development_ids,
        confirmation_ids,
        components=int(factor_config["components"]),
        random_state=int(factor_config["random_state"]),
        rotation=str(factor_config["rotation"]),
    )
    development_scores = scores.development.copy()
    development_scores.index = [participant_pseudonym(value) for value in development_scores.index]
    if set(participant_names) != set(development_scores.index):
        raise RuntimeError("Development factor IDs do not match extracted-state participants")
    features = development_scores.loc[participant_names].to_numpy(np.float32)
    factor_model = fit_fixed_feature_readout(
        data,
        features,
        weights,
        biases,
        rank=selected_rank,
        learning_rate=selected_lr,
        regularization=selected_regularization,
        epochs=int(config["individuality"]["optimization"]["max_epochs"]),
        batch_per_task=256,
        steps_per_epoch=10,
        seed=int(config["execution"]["seed"]) + 700,
    )
    factor_rows = evaluate_fixed_feature_readout(
        factor_model,
        data,
        list(range(len(participant_names))),
        features,
        set(config["individuality"]["preregistered_family_split"]["B"]),
    )
    for row in factor_rows:
        row["participant"] = participant_names[int(row.pop("participant_index"))]
        row["individual_nll"] = row["personalized_nll"]
        row["outer_fold"] = -1
        row["rank"] = selected_rank
        row["learning_rate"] = selected_lr
        row["regularization"] = selected_regularization
        row["method"] = "sro_factor"
        row["factor_variant"] = "development_only_five_factor_refit"
        row["analysis"] = "development_T1_factor_to_T1_B"
        row["config_hash"] = _sha256(CONFIG_PATH)
    FACTOR_MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in factor_model.state_dict().items()
            },
            "shared_dim": int(next(iter(data.values())).states.shape[1]),
            "feature_dim": features.shape[1],
            "rank": selected_rank,
        },
        FACTOR_MODEL,
    )
    confirmation_t1 = scores.confirmation_t1.copy()
    confirmation_t1.index = [participant_pseudonym(value) for value in confirmation_t1.index]
    FACTOR_SCORES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        FACTOR_SCORES,
        development_participant=np.asarray(participant_names, dtype=str),
        development_features=features,
        confirmation_participant=np.asarray(confirmation_t1.index, dtype=str),
        confirmation_t1=confirmation_t1.to_numpy(np.float32),
        loadings=scores.loadings,
        mean=scores.mean,
        scale=scores.scale,
        variables=np.asarray(scores.variables, dtype=str),
    )

    split = config["individuality"]["preregistered_family_split"]
    recovery_models: dict[str, Any] = {}
    recovery_coordinates: dict[str, np.ndarray] = {}
    recovery_remaps: dict[str, dict[int, int]] = {}
    for label in ("A", "B"):
        families = set(split[label])
        subset = {task: values for task, values in data.items() if values.family in families}
        model, remap = fit_individual_population(
            subset,
            weights,
            biases,
            rank=selected_rank,
            included_folds=set(range(config["data"]["sro"]["development_folds"])),
            learning_rate=selected_lr,
            regularization=selected_regularization,
            epochs=int(config["individuality"]["optimization"]["max_epochs"]),
            batch_per_task=256,
            steps_per_epoch=10,
            seed=int(config["execution"]["seed"]) + 800 + ord(label),
        )
        coordinates = model.coordinates().detach().cpu().numpy()
        ordered = np.stack([coordinates[remap[index]] for index in range(len(participant_names))])
        recovery_models[label] = model
        recovery_coordinates[label] = ordered
        recovery_remaps[label] = remap
    alignment = orthogonal_procrustes_mapping(recovery_coordinates["A"], recovery_coordinates["B"])
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    for label, model in recovery_models.items():
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "participant_remap": recovery_remaps[label],
                "rank": selected_rank,
                "families": split[label],
            },
            RECOVERY_ROOT / f"model_{label}.pt",
        )
    np.savez(
        RECOVERY_ROOT / "alignment.npz",
        participant=np.asarray(participant_names),
        coordinates_A=recovery_coordinates["A"],
        coordinates_B=recovery_coordinates["B"],
        R_align=alignment,
    )
    aligned_b = recovery_coordinates["B"] @ alignment
    pd.DataFrame(
        {
            "participant": participant_names,
            "set_A_norm": np.linalg.norm(recovery_coordinates["A"], axis=1),
            "set_B_norm": np.linalg.norm(recovery_coordinates["B"], axis=1),
            "aligned_distance": np.linalg.norm(recovery_coordinates["A"] - aligned_b, axis=1),
            "rank": selected_rank,
            "config_hash": _sha256(CONFIG_PATH),
        }
    ).to_parquet(RECOVERY_RESULT, index=False, compression="zstd")
    BASELINE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    released_factor_audit = _released_factor_audit()
    BASELINE_MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "original",
                "released_factor_reconstruction": released_factor_audit,
                "baseline_config_sha256": _sha256(BASELINE_CONFIG),
                "factor_model_sha256": _sha256(FACTOR_MODEL),
                "factor_scores_sha256": _sha256(FACTOR_SCORES),
                "recovery_models": {
                    label: _sha256(RECOVERY_ROOT / f"model_{label}.pt") for label in ("A", "B")
                },
                "alignment_sha256": _sha256(RECOVERY_ROOT / "alignment.npz"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return factor_rows


def main() -> None:
    config: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())
    shared = json.loads(SELECTION.read_text())
    model_name = shared["model"]
    layer = int(shared["layer"])
    shared_root = CACHE / "shared_final" / model_name / f"layer_{layer:02d}"
    weights, biases, classes = _load_population(shared_root / "model.pt")
    shared_basis = np.load(shared_root / "shared_basis.npy")
    projected_root = CACHE / "preprocessing" / model_name / f"layer_{layer:02d}" / "projected"
    state_root = CACHE / "8b_states" / model_name
    data = load_individual_task_data(
        projected_root,
        state_root,
        shared_basis,
        {task: spec.family for task, spec in SPECS.items()},
        classes,
    )
    participant_names = sorted(
        {
            value
            for task in SPECS
            for value in pd.read_parquet(
                state_root / task / "metadata.parquet", columns=["participant"]
            )["participant"].unique()
        }
    )
    optimization = config["individuality"]["optimization"]
    family_split = config["individuality"]["preregistered_family_split"]
    source_families = set(family_split["A"])
    target_families = set(family_split["B"])
    rank_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    folds = set(range(config["data"]["sro"]["development_folds"]))
    cache_root = CACHE / "individuality_cv"
    cache_root.mkdir(parents=True, exist_ok=True)
    feasible_ranks = [
        rank for rank in config["individuality"]["private_ranks"] if rank <= shared_basis.shape[1]
    ]
    for outer_fold in sorted(folds):
        inner_fold = (outer_fold + 1) % len(folds)
        inner_train_folds = folds - {outer_fold, inner_fold}
        outer_train_folds = folds - {outer_fold}
        inner_participants = sorted(
            {
                int(participant)
                for values in data.values()
                for participant, fold in zip(values.participants, values.folds, strict=True)
                if int(fold) == inner_fold
            }
        )
        outer_participants = sorted(
            {
                int(participant)
                for values in data.values()
                for participant, fold in zip(values.participants, values.folds, strict=True)
                if int(fold) == outer_fold
            }
        )
        for rank in feasible_ranks:
            candidates: list[dict[str, float]] = []
            for learning_rate in optimization["learning_rates"]:
                for regularization in optimization["l2_regularization"]:
                    evidence_path = cache_root / (
                        f"outer{outer_fold}_inner{inner_fold}_r{rank}_"
                        f"lr{learning_rate:.0e}_l2{regularization:.0e}.json"
                    )
                    if evidence_path.exists():
                        evidence = json.loads(evidence_path.read_text())
                    else:
                        model, _ = fit_individual_population(
                            data,
                            weights,
                            biases,
                            rank=rank,
                            included_folds=inner_train_folds,
                            learning_rate=learning_rate,
                            regularization=regularization,
                            epochs=optimization["max_epochs"],
                            batch_per_task=256,
                            steps_per_epoch=10,
                            seed=config["execution"]["seed"] + outer_fold + rank,
                        )
                        coordinates = infer_new_coordinates(
                            model,
                            data,
                            participant_ids=inner_participants,
                            source_families=source_families,
                            regularization=regularization,
                        )
                        evaluation = evaluate_new_coordinates(
                            model,
                            data,
                            inner_participants,
                            coordinates,
                            target_families,
                        )
                        evidence = {
                            "learning_rate": learning_rate,
                            "regularization": regularization,
                            "inner_individual_nll": float(
                                np.mean([row["individual_nll"] for row in evaluation])
                            ),
                        }
                        evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n")
                    candidates.append(evidence)
            selected = min(candidates, key=lambda item: item["inner_individual_nll"])
            final_model, _ = fit_individual_population(
                data,
                weights,
                biases,
                rank=rank,
                included_folds=outer_train_folds,
                learning_rate=selected["learning_rate"],
                regularization=selected["regularization"],
                epochs=optimization["max_epochs"],
                batch_per_task=256,
                steps_per_epoch=10,
                seed=config["execution"]["seed"] + 100 + outer_fold + rank,
            )
            outer_coordinates = infer_new_coordinates(
                final_model,
                data,
                participant_ids=outer_participants,
                source_families=source_families,
                regularization=selected["regularization"],
            )
            evaluation = evaluate_new_coordinates(
                final_model,
                data,
                outer_participants,
                outer_coordinates,
                target_families,
            )
            for row in evaluation:
                row["personalized_nll"] = row["individual_nll"]
                row.update(
                    {
                        "participant": participant_names[row.pop("participant_index")],
                        "outer_fold": outer_fold,
                        "rank": rank,
                        "learning_rate": selected["learning_rate"],
                        "regularization": selected["regularization"],
                        "method": "manifold",
                        "analysis": "development_T1_A_to_T1_B",
                        "config_hash": _sha256(CONFIG_PATH),
                    }
                )
                cross_rows.append(row)
            rank_rows.append(
                {
                    "outer_fold": outer_fold,
                    "rank": rank,
                    "learning_rate": selected["learning_rate"],
                    "regularization": selected["regularization"],
                    "mean_individual_nll": float(
                        np.mean([row["individual_nll"] for row in evaluation])
                    ),
                    "mean_delta_nll": float(np.mean([row["delta_nll"] for row in evaluation])),
                    "config_hash": _sha256(CONFIG_PATH),
                }
            )
            print(json.dumps(rank_rows[-1]), flush=True)
    rank_frame = pd.DataFrame(rank_rows)
    cross_frame = pd.DataFrame(cross_rows)
    PRIVATE_RANK.parent.mkdir(parents=True, exist_ok=True)
    rank_frame.to_parquet(PRIVATE_RANK, index=False, compression="zstd")
    cross_frame.to_parquet(CROSS_DOMAIN, index=False, compression="zstd")
    summary = rank_frame.groupby("rank")["mean_individual_nll"].agg(["mean", "std", "count"])
    best_rank = int(summary["mean"].idxmin())
    best_se = float(summary.loc[best_rank, "std"] / np.sqrt(summary.loc[best_rank, "count"]))
    selected_rank = int(
        summary.loc[summary["mean"] <= summary.loc[best_rank, "mean"] + best_se].index.min()
    )
    selected_rows = rank_frame.loc[rank_frame["rank"] == selected_rank]
    selected_lr = _mode(selected_rows["learning_rate"].tolist())
    selected_regularization = _mode(selected_rows["regularization"].tolist())
    final_model, remap = fit_individual_population(
        data,
        weights,
        biases,
        rank=selected_rank,
        included_folds=folds,
        learning_rate=selected_lr,
        regularization=selected_regularization,
        epochs=optimization["max_epochs"],
        batch_per_task=256,
        steps_per_epoch=10,
        seed=config["execution"]["seed"],
    )
    final_root = CACHE / "individuality_final"
    final_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in final_model.state_dict().items()
            },
            "participant_remap": remap,
            "rank": selected_rank,
            "regularization": selected_regularization,
            "learning_rate": selected_lr,
        },
        final_root / "model.pt",
    )
    selection = {
        "rank": selected_rank,
        "learning_rate": selected_lr,
        "regularization": selected_regularization,
        "one_standard_error_threshold": float(summary.loc[best_rank, "mean"] + best_se),
        "best_rank_point_estimate": best_rank,
        "config_hash": _sha256(CONFIG_PATH),
    }
    PRIVATE_SELECTION.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    factor_rows = _save_auxiliary_original_models(
        config,
        data,
        participant_names,
        weights,
        biases,
        selected_rank,
        selected_lr,
        selected_regularization,
    )
    combined_cross = pd.concat(
        [pd.read_parquet(CROSS_DOMAIN), pd.DataFrame(factor_rows)],
        ignore_index=True,
        sort=False,
    )
    combined_cross.to_parquet(CROSS_DOMAIN, index=False, compression="zstd")
    print(json.dumps({"selected": selection}, sort_keys=True))


if __name__ == "__main__":
    main()
