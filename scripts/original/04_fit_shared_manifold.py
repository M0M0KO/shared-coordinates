#!/usr/bin/env python3
"""Fit original task-balanced PCA and prepare shared-manifold state matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cognitive_manifold.geometry.preprocessing import (
    fit_task_balanced_randomized_pca,
    project_state_file,
)
from cognitive_manifold.geometry.shared import (
    evaluate_task_balanced_nll,
    fit_shared_all,
    fit_shared_fold,
    load_task_arrays,
)
from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/original/main.yaml"
CACHE = ROOT / "cache/original"
MODEL_CHOICES = (
    "llama_base_8b",
    "minitaur_8b",
    "reproduced_intact_psych101_8b",
    "response_shuffled_psych101_8b",
)


def _condition_name(pca_dim: int, remove_pcs: int, unit_normalize: bool) -> str:
    return f"q{pca_dim}_rm{remove_pcs}_unit{int(unit_normalize)}"


def _output_paths(
    model: str,
    layer: int,
    primary_layer: int,
    pca_dim: int,
    remove_pcs: int,
    unit_normalize: bool,
) -> tuple[Path, Path, Path]:
    primary_condition = pca_dim == 512 and remove_pcs == 0 and not unit_normalize
    if model == "minitaur_8b" and layer == primary_layer and primary_condition:
        results = ROOT / "results/original/shared_manifold.parquet"
        selection = ROOT / "results/original/selected_shared.json"
    else:
        suffix = f"{model}_layer_{layer:02d}"
        if not primary_condition:
            suffix += f"_{_condition_name(pca_dim, remove_pcs, unit_normalize)}"
        results = ROOT / f"results/original/shared_manifold_{suffix}.parquet"
        selection = ROOT / f"results/original/selected_shared_{suffix}.json"
    manifest_suffix = f"{model}_layer_{layer:02d}"
    if not primary_condition:
        manifest_suffix += f"_{_condition_name(pca_dim, remove_pcs, unit_normalize)}"
    manifest = ROOT / f"manifests/private/original/preprocessing_{manifest_suffix}.json"
    return manifest, results, selection


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CHOICES, default="minitaur_8b")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--pca-dim", type=int, choices=(256, 512, 1024), default=512)
    parser.add_argument("--remove-pcs", type=int, choices=(0, 1, 3, 5), default=0)
    parser.add_argument("--unit-normalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if not config["execution"]["runnable"]:
        raise RuntimeError("Original profile is not runnable")
    layer = args.layer or config["likelihood_and_representation"]["primary_layer_8b"]
    if args.unit_normalize and args.remove_pcs:
        raise RuntimeError("Unit normalization and PC removal are separate frozen sensitivities")
    allowed_q = {
        int(config["likelihood_and_representation"]["pca"]["primary_dim"]),
        *map(int, config["likelihood_and_representation"]["pca"]["sensitivity_dims"]),
    }
    if args.pca_dim not in allowed_q:
        raise RuntimeError(f"PCA dimension {args.pca_dim} is outside the frozen sensitivity set")
    manifest_path, results_path, selection_path = _output_paths(
        args.model,
        layer,
        int(config["likelihood_and_representation"]["primary_layer_8b"]),
        args.pca_dim,
        args.remove_pcs,
        args.unit_normalize,
    )
    state_files = {
        task: CACHE / "8b_states" / args.model / task / f"layer_{layer:02d}.npy" for task in SPECS
    }
    missing = [str(path) for path in state_files.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"State extraction is incomplete: {missing[:3]}")
    pca_config = config["likelihood_and_representation"]["pca"]
    maximum_components = max(pca_config["primary_dim"], *pca_config["sensitivity_dims"])
    standard_output = CACHE / "preprocessing" / args.model / f"layer_{layer:02d}"
    condition = _condition_name(args.pca_dim, args.remove_pcs, args.unit_normalize)
    sensitivity_output = (
        CACHE / "preprocessing_sensitivity" / args.model / f"layer_{layer:02d}" / condition
    )
    basis_output = sensitivity_output if args.unit_normalize else standard_output
    projection_output = (
        sensitivity_output if args.unit_normalize or args.remove_pcs else standard_output
    )
    basis_path = basis_output / "pca.npz"
    if basis_path.exists() and not args.overwrite:
        payload = np.load(basis_path)
        mean = payload["mean"]
        pca_center = (
            payload["pca_center"]
            if "pca_center" in payload.files
            else np.zeros(payload["components"].shape[1], dtype=np.float32)
        )
        components = payload["components"]
        evidence = json.loads(str(payload["evidence"]))
    else:
        pca, mean, evidence = fit_task_balanced_randomized_pca(
            state_files,
            components=maximum_components,
            maximum_per_task=pca_config["max_states_per_task_for_basis_fit"],
            seed=config["execution"]["seed"],
            unit_length_normalization=args.unit_normalize,
        )
        mean = mean.astype(np.float32)
        pca_center = pca.mean_.astype(np.float32)
        components = pca.components_.astype(np.float32)
        basis_output.mkdir(parents=True, exist_ok=True)
        np.savez(
            basis_path,
            mean=mean,
            pca_center=pca_center,
            components=components,
            explained_variance=pca.explained_variance_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            evidence=json.dumps(evidence, sort_keys=True),
        )
    projected_files: dict[str, str] = {}
    for task, source in state_files.items():
        destination = projection_output / "projected" / f"{task}.npy"
        if not destination.exists() or args.overwrite:
            project_state_file(
                source,
                destination,
                mean,
                components,
                pca_center=pca_center,
                unit_length_normalization=args.unit_normalize,
                remove_first_components=args.remove_pcs,
            )
        projected_files[task] = _sha256(destination)
    manifest = {
        "schema_version": 1,
        "profile": "original",
        "model": args.model,
        "layer": layer,
        "config_hash": _sha256(CONFIG_PATH),
        "basis_sha256": _sha256(basis_path),
        "pca_dim": args.pca_dim,
        "remove_first_global_pcs": args.remove_pcs,
        "unit_length_normalization": args.unit_normalize,
        "evidence": evidence,
        "projected_files": projected_files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    if args.preprocess_only:
        return

    input_dim = args.pca_dim
    state_root = CACHE / "8b_states" / args.model
    data = load_task_arrays(projection_output / "projected", state_root, list(SPECS), input_dim)
    optimization = config["shared_manifold"]["optimization"]
    cv_root = CACHE / "shared_cv" / args.model / f"layer_{layer:02d}" / condition
    cv_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    dimensions = [
        int(value) for value in config["shared_manifold"]["dimensions"] if int(value) <= input_dim
    ]
    for dimension in dimensions:
        for learning_rate in optimization["learning_rates"]:
            for weight_decay in optimization["weight_decays"]:
                for fold in range(config["data"]["sro"]["development_folds"]):
                    stem = f"d{dimension}_lr{learning_rate:.0e}_wd{weight_decay:.0e}_fold{fold}"
                    evidence_path = cv_root / f"{stem}.json"
                    if evidence_path.exists() and not args.overwrite:
                        evidence = json.loads(evidence_path.read_text())
                    else:
                        model, training = fit_shared_fold(
                            data,
                            input_dim=input_dim,
                            shared_dim=dimension,
                            validation_fold=fold,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            max_epochs=optimization["max_epochs"],
                            patience=optimization["early_stopping_patience"],
                            minimum_improvement=optimization["minimum_validation_improvement"],
                            batch_per_task=optimization["task_balanced_trials_per_task_per_step"],
                            steps_per_epoch=optimization["steps_per_epoch"],
                            seed=config["execution"]["seed"] + fold,
                        )
                        validation = {
                            task: np.flatnonzero(values.folds == fold)
                            for task, values in data.items()
                        }
                        nll, task_nll = evaluate_task_balanced_nll(
                            model, data, validation, torch.device("cuda:0")
                        )
                        evidence = {
                            "dimension": dimension,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "fold": fold,
                            "validation_nll": nll,
                            "task_nll": task_nll,
                            "best_epoch": training["best_epoch"],
                        }
                        evidence_path.write_text(
                            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
                        )
                    rows.append(
                        {
                            **evidence,
                            "model": args.model,
                            "layer": layer,
                            "config_hash": _sha256(CONFIG_PATH),
                            "pca_dim": input_dim,
                            "remove_first_global_pcs": args.remove_pcs,
                            "unit_length_normalization": args.unit_normalize,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "dimension": dimension,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                                "fold": fold,
                                "validation_nll": evidence["validation_nll"],
                            }
                        ),
                        flush=True,
                    )
    results = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_path, index=False, compression="zstd")
    summary = (
        results.groupby(["dimension", "learning_rate", "weight_decay"])["validation_nll"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    best_by_dimension = summary.loc[summary.groupby("dimension")["mean"].idxmin()].copy()
    best_row = best_by_dimension.loc[best_by_dimension["mean"].idxmin()]
    best_se = float(best_row["std"] / np.sqrt(best_row["count"]))
    eligible = best_by_dimension.loc[best_by_dimension["mean"] <= best_row["mean"] + best_se]
    selected = eligible.sort_values("dimension").iloc[0]
    selected_dimension = int(selected["dimension"])
    selected_lr = float(selected["learning_rate"])
    selected_wd = float(selected["weight_decay"])
    selected_folds = results.loc[
        (results["dimension"] == selected_dimension)
        & (results["learning_rate"] == selected_lr)
        & (results["weight_decay"] == selected_wd)
    ]
    final_epochs = int(np.median(selected_folds["best_epoch"])) + 1
    final_model = fit_shared_all(
        data,
        input_dim=input_dim,
        shared_dim=selected_dimension,
        learning_rate=selected_lr,
        weight_decay=selected_wd,
        epochs=final_epochs,
        batch_per_task=optimization["task_balanced_trials_per_task_per_step"],
        steps_per_epoch=optimization["steps_per_epoch"],
        seed=config["execution"]["seed"],
    )
    primary_condition = input_dim == 512 and args.remove_pcs == 0 and not args.unit_normalize
    final_root = (
        CACHE / "shared_final" / args.model / f"layer_{layer:02d}"
        if primary_condition
        else CACHE / "shared_final_sensitivity" / args.model / f"layer_{layer:02d}" / condition
    )
    final_root.mkdir(parents=True, exist_ok=True)
    np.save(final_root / "shared_basis.npy", final_model.basis().detach().cpu().numpy())
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in final_model.state_dict().items()
            },
            "classes": {task: values.classes for task, values in data.items()},
            "input_dim": input_dim,
            "shared_dim": selected_dimension,
        },
        final_root / "model.pt",
    )
    selection = {
        "model": args.model,
        "layer": layer,
        "dimension": selected_dimension,
        "learning_rate": selected_lr,
        "weight_decay": selected_wd,
        "final_epochs": final_epochs,
        "one_standard_error_threshold": float(best_row["mean"] + best_se),
        "best_mean_validation_nll": float(best_row["mean"]),
        "config_hash": _sha256(CONFIG_PATH),
        "pca_dim": input_dim,
        "remove_first_global_pcs": args.remove_pcs,
        "unit_length_normalization": args.unit_normalize,
    }
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selection}, sort_keys=True))


if __name__ == "__main__":
    main()
