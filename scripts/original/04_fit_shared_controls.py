#!/usr/bin/env python3
"""Fit original full, task-specific, Haar and held-out-family shared controls."""

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

from cognitive_manifold.geometry.controls import (
    evaluate_fixed_basis,
    fit_fixed_basis_fold,
)
from cognitive_manifold.geometry.shared import (
    evaluate_task_balanced_nll,
    fit_shared_fold,
    load_task_arrays,
)
from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/original/main.yaml"
SELECTION = ROOT / "results/original/selected_shared.json"
SHARED = ROOT / "results/original/shared_manifold.parquet"
CACHE = ROOT / "cache/original/shared_controls"
RESULT = ROOT / "results/original/shared_controls.parquet"
MANIFEST = ROOT / "manifests/private/original/shared_controls.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _data() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(CONFIG.read_text())
    selection = json.loads(SELECTION.read_text())
    layer = int(selection["layer"])
    projected = ROOT / f"cache/original/preprocessing/minitaur_8b/layer_{layer:02d}/projected"
    states = ROOT / "cache/original/8b_states/minitaur_8b"
    data = load_task_arrays(projected, states, list(SPECS), 512)
    return config, selection, data


def _write_cache(key: str, rows: list[dict[str, Any]]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{key}.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def _write_basis(path: Path, basis: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, basis)


def _read_or_fit_fixed(
    key: str,
    control: str,
    data: dict[str, Any],
    basis: np.ndarray,
    fold: int,
    config: dict[str, Any],
    selection: dict[str, Any],
    *,
    replicate: int,
    held_out_family: str | None = None,
) -> list[dict[str, Any]]:
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    optimization = config["shared_manifold"]["optimization"]
    model, evidence = fit_fixed_basis_fold(
        data,
        basis,
        validation_fold=fold,
        learning_rate=float(selection["learning_rate"]),
        weight_decay=float(selection["weight_decay"]),
        max_epochs=int(optimization["max_epochs"]),
        patience=int(optimization["early_stopping_patience"]),
        minimum_improvement=float(optimization["minimum_validation_improvement"]),
        batch_per_task=int(optimization["task_balanced_trials_per_task_per_step"]),
        steps_per_epoch=int(optimization["steps_per_epoch"]),
        seed=int(config["execution"]["seed"]) + fold + max(replicate, 0) * 17,
    )
    validation = {task: np.flatnonzero(values.folds == fold) for task, values in data.items()}
    task_nll = evaluate_fixed_basis(model, data, validation)
    rows = [
        {
            "control": control,
            "dimension": int(basis.shape[1]),
            "fold": fold,
            "replicate": replicate,
            "task": task,
            "held_out_family": held_out_family,
            "validation_nll": value,
            "best_epoch": evidence["best_epoch"],
            "model": "minitaur_8b",
            "layer": int(selection["layer"]),
            "config_hash": _sha256(CONFIG),
        }
        for task, value in task_nll.items()
    ]
    _write_cache(key, rows)
    return rows


def shared_behavioral_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    shared = pd.read_parquet(SHARED)
    selected = shared.loc[
        (shared["dimension"] == int(selection["dimension"]))
        & (shared["learning_rate"] == float(selection["learning_rate"]))
        & (shared["weight_decay"] == float(selection["weight_decay"]))
    ]
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        task_nll = row.task_nll
        if not isinstance(task_nll, dict):
            task_nll = dict(task_nll)
        for task, value in task_nll.items():
            rows.append(
                {
                    "control": "shared_behavioral",
                    "dimension": int(selection["dimension"]),
                    "fold": int(row.fold),
                    "replicate": -1,
                    "task": task,
                    "held_out_family": None,
                    "validation_nll": float(value),
                    "best_epoch": int(row.best_epoch),
                    "model": "minitaur_8b",
                    "layer": int(selection["layer"]),
                    "config_hash": _sha256(CONFIG),
                }
            )
    return rows


def null_intercept_rows(
    config: dict[str, Any], selection: dict[str, Any], data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fold-specific task intercepts needed for the frozen retained-information ratio."""
    rows: list[dict[str, Any]] = []
    for fold in range(int(config["data"]["sro"]["development_folds"])):
        for task, values in data.items():
            training = values.labels[values.folds != fold]
            validation = values.labels[values.folds == fold]
            counts = np.bincount(training, minlength=len(values.classes)).astype(np.float64)
            if not len(validation) or np.any(counts == 0):
                raise RuntimeError(f"Cannot fit an unsmoothed null intercept for {task}/fold{fold}")
            probabilities = counts / counts.sum()
            nll = float(-np.log(probabilities[validation]).mean())
            rows.append(
                {
                    "control": "null_intercept",
                    "dimension": 0,
                    "fold": fold,
                    "replicate": -1,
                    "task": task,
                    "held_out_family": None,
                    "validation_nll": nll,
                    "best_epoch": -1,
                    "model": "minitaur_8b",
                    "layer": int(selection["layer"]),
                    "config_hash": _sha256(CONFIG),
                }
            )
    return rows


def fit_standard_controls(control: str) -> list[dict[str, Any]]:
    config, selection, data = _data()
    dimension = int(selection["dimension"])
    folds = range(int(config["data"]["sro"]["development_folds"]))
    rows: list[dict[str, Any]] = []
    if control == "full_representation":
        basis = np.eye(512, dtype=np.float32)
        for fold in folds:
            rows.extend(
                _read_or_fit_fixed(
                    f"full_fold{fold}", control, data, basis, fold, config, selection, replicate=-1
                )
            )
    elif control == "haar_random":
        count = int(config["shared_manifold"]["controls"]["haar_random_subspaces"])
        for replicate in range(count):
            rng = np.random.default_rng(int(config["execution"]["seed"]) + replicate)
            basis = np.linalg.qr(rng.normal(size=(512, dimension)))[0].astype(np.float32)
            for fold in folds:
                rows.extend(
                    _read_or_fit_fixed(
                        f"haar_r{replicate:03d}_fold{fold}",
                        control,
                        data,
                        basis,
                        fold,
                        config,
                        selection,
                        replicate=replicate,
                    )
                )
    elif control == "task_specific":
        optimization = config["shared_manifold"]["optimization"]
        for task, values in data.items():
            for fold in folds:
                key = f"task_specific_{task}_fold{fold}"
                path = CACHE / f"{key}.json"
                basis_path = CACHE / f"{key}_basis.npy"
                if path.exists() and basis_path.exists():
                    rows.extend(json.loads(path.read_text()))
                    continue
                model, evidence = fit_shared_fold(
                    {task: values},
                    input_dim=512,
                    shared_dim=dimension,
                    validation_fold=fold,
                    learning_rate=float(selection["learning_rate"]),
                    weight_decay=float(selection["weight_decay"]),
                    max_epochs=int(optimization["max_epochs"]),
                    patience=int(optimization["early_stopping_patience"]),
                    minimum_improvement=float(optimization["minimum_validation_improvement"]),
                    batch_per_task=int(optimization["task_balanced_trials_per_task_per_step"]),
                    steps_per_epoch=int(optimization["steps_per_epoch"]),
                    seed=int(config["execution"]["seed"]) + fold,
                )
                validation = {task: np.flatnonzero(values.folds == fold)}
                _mean, task_nll = evaluate_task_balanced_nll(
                    model, {task: values}, validation, torch.device("cuda:0")
                )
                item = [
                    {
                        "control": control,
                        "dimension": dimension,
                        "fold": fold,
                        "replicate": -1,
                        "task": task,
                        "held_out_family": None,
                        "validation_nll": task_nll[task],
                        "best_epoch": evidence["best_epoch"],
                        "model": "minitaur_8b",
                        "layer": int(selection["layer"]),
                        "config_hash": _sha256(CONFIG),
                    }
                ]
                _write_basis(
                    basis_path,
                    model.basis().detach().cpu().numpy().astype(np.float32),
                )
                _write_cache(key, item)
                rows.extend(item)
    else:
        raise ValueError(control)
    return rows


def fit_held_out_family() -> list[dict[str, Any]]:
    config, selection, data = _data()
    optimization = config["shared_manifold"]["optimization"]
    dimension = int(selection["dimension"])
    families = sorted({spec.family for spec in SPECS.values()})
    rows: list[dict[str, Any]] = []
    for family in families:
        source = {task: values for task, values in data.items() if SPECS[task].family != family}
        target = {task: values for task, values in data.items() if SPECS[task].family == family}
        for fold in range(int(config["data"]["sro"]["development_folds"])):
            key = f"heldout_{family}_fold{fold}"
            path = CACHE / f"{key}.json"
            if path.exists():
                rows.extend(json.loads(path.read_text()))
                continue
            source_model, source_evidence = fit_shared_fold(
                source,
                input_dim=512,
                shared_dim=dimension,
                validation_fold=fold,
                learning_rate=float(selection["learning_rate"]),
                weight_decay=float(selection["weight_decay"]),
                max_epochs=int(optimization["max_epochs"]),
                patience=int(optimization["early_stopping_patience"]),
                minimum_improvement=float(optimization["minimum_validation_improvement"]),
                batch_per_task=int(optimization["task_balanced_trials_per_task_per_step"]),
                steps_per_epoch=int(optimization["steps_per_epoch"]),
                seed=int(config["execution"]["seed"]) + fold + 500,
            )
            basis = source_model.basis().detach().cpu().numpy()
            target_model, target_evidence = fit_fixed_basis_fold(
                target,
                basis,
                validation_fold=fold,
                learning_rate=float(selection["learning_rate"]),
                weight_decay=float(selection["weight_decay"]),
                max_epochs=int(optimization["max_epochs"]),
                patience=int(optimization["early_stopping_patience"]),
                minimum_improvement=float(optimization["minimum_validation_improvement"]),
                batch_per_task=int(optimization["task_balanced_trials_per_task_per_step"]),
                steps_per_epoch=int(optimization["steps_per_epoch"]),
                seed=int(config["execution"]["seed"]) + fold + 600,
            )
            validation = {
                task: np.flatnonzero(values.folds == fold) for task, values in target.items()
            }
            task_nll = evaluate_fixed_basis(target_model, target, validation)
            current = [
                {
                    "control": "held_out_family_transfer",
                    "dimension": dimension,
                    "fold": fold,
                    "replicate": -1,
                    "task": task,
                    "held_out_family": family,
                    "validation_nll": value,
                    "best_epoch": target_evidence["best_epoch"],
                    "source_best_epoch": source_evidence["best_epoch"],
                    "model": "minitaur_8b",
                    "layer": int(selection["layer"]),
                    "config_hash": _sha256(CONFIG),
                }
                for task, value in task_nll.items()
            ]
            _write_cache(key, current)
            rows.extend(current)
    return rows


def add_finetune_controls(rows: list[dict[str, Any]]) -> None:
    selection = json.loads(SELECTION.read_text())
    dimension = int(selection["dimension"])
    layer = int(selection["layer"])
    names = {
        "reproduced_intact_psych101_8b": "shared_reproduced_intact",
        "response_shuffled_psych101_8b": "shared_response_shuffled",
    }
    for model, control in names.items():
        path = ROOT / f"results/original/shared_manifold_{model}_layer_{layer:02d}.parquet"
        if not path.exists():
            raise RuntimeError(f"Missing frozen O8 SRO control curve: {path}")
        frame = pd.read_parquet(path)
        dimension_rows = frame.loc[frame["dimension"] == dimension]
        best = (
            dimension_rows.groupby(["learning_rate", "weight_decay"])["validation_nll"]
            .mean()
            .idxmin()
        )
        selected = dimension_rows.loc[
            (dimension_rows["learning_rate"] == best[0])
            & (dimension_rows["weight_decay"] == best[1])
        ]
        for row in selected.itertuples(index=False):
            task_nll = row.task_nll if isinstance(row.task_nll, dict) else dict(row.task_nll)
            for task, value in task_nll.items():
                rows.append(
                    {
                        "control": control,
                        "dimension": dimension,
                        "fold": int(row.fold),
                        "replicate": -1,
                        "task": task,
                        "held_out_family": None,
                        "validation_nll": float(value),
                        "best_epoch": int(row.best_epoch),
                        "model": model,
                        "layer": layer,
                        "config_hash": _sha256(CONFIG),
                    }
                )


def finalize() -> None:
    config, selection, data = _data()
    rows = shared_behavioral_rows(selection)
    rows.extend(null_intercept_rows(config, selection, data))
    fold_count = int(config["data"]["sro"]["development_folds"])
    task_count = len(SPECS)
    family_count = len({spec.family for spec in SPECS.values()})
    haar_count = int(config["shared_manifold"]["controls"]["haar_random_subspaces"])
    expected_groups = {
        "full": ("full_fold*.json", fold_count, fold_count * task_count),
        "task_specific": (
            "task_specific_*_fold*.json",
            fold_count * task_count,
            fold_count * task_count,
        ),
        "heldout": (
            "heldout_*_fold*.json",
            fold_count * family_count,
            fold_count * task_count,
        ),
        **{
            f"haar_r{replicate:03d}": (
                f"haar_r{replicate:03d}_fold*.json",
                fold_count,
                fold_count * task_count,
            )
            for replicate in range(haar_count)
        },
    }
    incomplete: list[str] = []
    for name, (pattern, expected_files, expected_rows) in expected_groups.items():
        paths = sorted(CACHE.glob(pattern))
        current_rows: list[dict[str, Any]] = []
        for path in paths:
            current_rows.extend(json.loads(path.read_text()))
        if len(paths) != expected_files or len(current_rows) != expected_rows:
            incomplete.append(
                f"{name}: files={len(paths)}/{expected_files}, "
                f"rows={len(current_rows)}/{expected_rows}"
            )
        rows.extend(current_rows)
    if incomplete:
        raise RuntimeError(f"Shared controls are incomplete: {incomplete[:10]}")
    add_finetune_controls(rows)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(RESULT, index=False, compression="zstd")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "original",
                "rows": len(rows),
                "result_sha256": _sha256(RESULT),
                "config_sha256": _sha256(CONFIG),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("full", "task-specific", "haar", "held-out", "finalize"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "full":
        rows = fit_standard_controls("full_representation")
    elif args.action == "task-specific":
        rows = fit_standard_controls("task_specific")
    elif args.action == "haar":
        rows = fit_standard_controls("haar_random")
    elif args.action == "held-out":
        rows = fit_held_out_family()
    else:
        finalize()
        rows = []
    print(json.dumps({"action": args.action, "rows": len(rows)}))


if __name__ == "__main__":
    main()
