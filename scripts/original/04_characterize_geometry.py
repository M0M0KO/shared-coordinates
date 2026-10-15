#!/usr/bin/env python3
"""Produce the original independent geometry, sensitivity and trajectory tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors

from cognitive_manifold.geometry.characterization import (
    levina_bickel_dimension,
    participation_ratio,
    subspace_angles_and_alignment,
    trajectory_summary,
    two_nn_dimension,
)
from cognitive_manifold.geometry.preprocessing import task_balanced_indices
from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/original/main.yaml"
GEOMETRY_CONFIG = ROOT / "configs/original/geometry_frozen.yaml"
RESULTS = ROOT / "results/original"
CACHE = ROOT / "cache/original"
MANIFEST = ROOT / "manifests/private/original/geometry_characterization.json"
INTRINSIC = RESULTS / "intrinsic_dimension.parquet"
ANGLES = RESULTS / "task_subspace_angles.parquet"
TRAJECTORIES = RESULTS / "shared_trajectories.parquet"
SENSITIVITIES = RESULTS / "representation_sensitivities.parquet"
RETAINED = RESULTS / "retained_information.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_name(pca_dim: int, remove_pcs: int, unit_normalize: bool) -> Path:
    if pca_dim == 512 and remove_pcs == 0 and not unit_normalize:
        return RESULTS / "selected_shared.json"
    condition = f"q{pca_dim}_rm{remove_pcs}_unit{int(unit_normalize)}"
    return RESULTS / f"selected_shared_minitaur_8b_layer_24_{condition}.json"


def run_intrinsic(frozen: dict[str, Any]) -> None:
    settings = frozen["intrinsic_dimension"]
    rows: list[dict[str, Any]] = []
    maximum_neighbors = max(map(int, settings["levina_bickel_neighbors"]))
    for model in settings["models"]:
        root = CACHE / "preprocessing" / model / f"layer_{int(settings['layer']):02d}"
        pca = np.load(root / "pca.npz")
        eigenvalues = pca["explained_variance"][: int(settings["pca_dimension"])]
        rows.append(
            {
                "model": model,
                "layer": int(settings["layer"]),
                "estimator": "participation_ratio",
                "parameter": None,
                "estimate": participation_ratio(eigenvalues),
                "sample_size": int(sum(json.loads(str(pca["evidence"]))["sample_counts"].values())),
            }
        )
        samples: list[np.ndarray] = []
        for task in SPECS:
            states = np.load(root / "projected" / f"{task}.npy", mmap_mode="r")
            indices = task_balanced_indices(
                len(states),
                int(settings["local_sample_per_task"]),
                int(settings["sampling_seed"]),
                f"{model}|{task}",
            )
            samples.append(
                np.asarray(states[indices, : int(settings["pca_dimension"])], np.float32)
            )
        sample = np.concatenate(samples)
        if len(sample) != int(settings["task_balanced_total_local_sample"]):
            raise RuntimeError(f"Local geometry sample size changed for {model}: {len(sample)}")
        neighbors = NearestNeighbors(
            n_neighbors=maximum_neighbors + 1,
            metric=str(settings["distance"]),
            n_jobs=-1,
        ).fit(sample)
        distances, _indices = neighbors.kneighbors(sample, return_distance=True)
        rows.append(
            {
                "model": model,
                "layer": int(settings["layer"]),
                "estimator": "two_nn",
                "parameter": 2,
                "estimate": two_nn_dimension(distances),
                "sample_size": len(sample),
            }
        )
        for count in settings["levina_bickel_neighbors"]:
            rows.append(
                {
                    "model": model,
                    "layer": int(settings["layer"]),
                    "estimator": "levina_bickel",
                    "parameter": int(count),
                    "estimate": levina_bickel_dimension(distances, int(count)),
                    "sample_size": len(sample),
                }
            )
    pd.DataFrame(rows).to_parquet(INTRINSIC, index=False, compression="zstd")


def run_angles(config: dict[str, Any]) -> None:
    selection = json.loads((RESULTS / "selected_shared.json").read_text())
    dimension = int(selection["dimension"])
    rows: list[dict[str, Any]] = []
    for fold in range(int(config["data"]["sro"]["development_folds"])):
        bases = {
            task: np.load(CACHE / "shared_controls" / f"task_specific_{task}_fold{fold}_basis.npy")[
                :, :dimension
            ]
            for task in SPECS
        }
        for first, second in combinations(sorted(bases), 2):
            angles, alignment = subspace_angles_and_alignment(bases[first], bases[second])
            rows.append(
                {
                    "fold": fold,
                    "task_a": first,
                    "task_b": second,
                    "family_a": SPECS[first].family,
                    "family_b": SPECS[second].family,
                    "dimension": dimension,
                    "principal_angles_radians": json.dumps(angles.tolist()),
                    "mean_squared_cosine_alignment": alignment,
                }
            )
    pd.DataFrame(rows).to_parquet(ANGLES, index=False, compression="zstd")


def run_trajectories(frozen: dict[str, Any]) -> None:
    settings = frozen["trajectories"]
    selection = json.loads((RESULTS / "selected_shared.json").read_text())
    layer = int(selection["layer"])
    projected_root = CACHE / "preprocessing/minitaur_8b" / f"layer_{layer:02d}/projected"
    shared = np.load(CACHE / "shared_final/minitaur_8b" / f"layer_{layer:02d}/shared_basis.npy")
    rows: list[dict[str, Any]] = []
    for task in settings["tasks"]:
        metadata = pd.read_parquet(
            CACHE / "8b_states/minitaur_8b" / task / "metadata.parquet",
            columns=["participant", "trial", "state_row"],
        )
        projected = np.load(projected_root / f"{task}.npy", mmap_mode="r")[:, :512]
        latent = np.asarray(projected, np.float32) @ shared
        for participant, frame in metadata.groupby("participant", sort=False):
            ordered = frame.sort_values("trial", kind="stable")
            if len(ordered) < int(settings["minimum_states"]):
                continue
            summary = trajectory_summary(
                latent[ordered["state_row"].to_numpy(np.int64)],
                epsilon=float(settings["epsilon"]),
            )
            rows.append(
                {
                    "participant": participant,
                    "task": task,
                    "task_family": SPECS[task].family,
                    "layer": layer,
                    "dimension": int(selection["dimension"]),
                    **summary,
                }
            )
    pd.DataFrame(rows).to_parquet(TRAJECTORIES, index=False, compression="zstd")


def run_sensitivities(frozen: dict[str, Any]) -> None:
    settings = frozen["representation_sensitivities"]
    conditions = [(512, 0, False)]
    conditions.extend((int(value), 0, False) for value in settings["pca_dimensions"])
    conditions.extend((512, int(value), False) for value in settings["remove_first_global_pcs"])
    conditions.append((512, 0, True))
    rows: list[dict[str, Any]] = []
    for pca_dim, remove_pcs, unit_normalize in conditions:
        path = _selection_name(pca_dim, remove_pcs, unit_normalize)
        if not path.is_file():
            raise RuntimeError(f"Missing frozen representation sensitivity: {path}")
        selection = json.loads(path.read_text())
        rows.append(
            {
                "pca_dim": pca_dim,
                "remove_first_global_pcs": remove_pcs,
                "unit_length_normalization": unit_normalize,
                "selected_dimension": int(selection["dimension"]),
                "best_mean_validation_nll": float(selection["best_mean_validation_nll"]),
                "one_standard_error_threshold": float(selection["one_standard_error_threshold"]),
                "selection_sha256": _sha256(path),
            }
        )
    pd.DataFrame(rows).to_parquet(SENSITIVITIES, index=False, compression="zstd")


def run_retained_information() -> None:
    selection = json.loads((RESULTS / "selected_shared.json").read_text())
    controls = pd.read_parquet(RESULTS / "shared_controls.parquet")
    columns = ["fold", "task", "validation_nll"]
    null = controls.loc[controls["control"] == "null_intercept", columns].rename(
        columns={"validation_nll": "null_nll"}
    )
    full = controls.loc[controls["control"] == "full_representation", columns].rename(
        columns={"validation_nll": "full_nll"}
    )
    shared_curve = pd.read_parquet(RESULTS / "shared_manifold.parquet")
    expanded: list[dict[str, Any]] = []
    for dimension, dimension_rows in shared_curve.groupby("dimension"):
        best = (
            dimension_rows.groupby(["learning_rate", "weight_decay"])["validation_nll"]
            .mean()
            .idxmin()
        )
        chosen = dimension_rows.loc[
            (dimension_rows["learning_rate"] == best[0])
            & (dimension_rows["weight_decay"] == best[1])
        ]
        for row in chosen.itertuples(index=False):
            task_nll = row.task_nll if isinstance(row.task_nll, dict) else dict(row.task_nll)
            for task, value in task_nll.items():
                expanded.append(
                    {
                        "dimension": int(dimension),
                        "fold": int(row.fold),
                        "task": task,
                        "shared_nll": float(value),
                        "learning_rate": float(best[0]),
                        "weight_decay": float(best[1]),
                    }
                )
    shared = pd.DataFrame(expanded)
    output = shared.merge(null, on=["fold", "task"], validate="many_to_one").merge(
        full, on=["fold", "task"], validate="many_to_one"
    )
    denominator = output["null_nll"] - output["full_nll"]
    if (denominator <= 0).any():
        raise RuntimeError("Full representation fails to improve over the frozen null intercept")
    output["retained_information_fraction"] = (
        output["null_nll"] - output["shared_nll"]
    ) / denominator
    output["selected_dimension"] = output["dimension"] == int(selection["dimension"])
    output.to_parquet(RETAINED, index=False, compression="zstd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("intrinsic", "angles", "trajectories", "sensitivities", "retained", "all"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(CONFIG.read_text())
    frozen = yaml.safe_load(GEOMETRY_CONFIG.read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    actions = (
        ("intrinsic", lambda: run_intrinsic(frozen)),
        ("angles", lambda: run_angles(config)),
        ("trajectories", lambda: run_trajectories(frozen)),
        ("sensitivities", lambda: run_sensitivities(frozen)),
        ("retained", run_retained_information),
    )
    completed: dict[str, Any] = {}
    for name, function in actions:
        if args.action not in {name, "all"}:
            continue
        function()
        path = {
            "intrinsic": INTRINSIC,
            "angles": ANGLES,
            "trajectories": TRAJECTORIES,
            "sensitivities": SENSITIVITIES,
            "retained": RETAINED,
        }[name]
        completed[name] = {"rows": len(pd.read_parquet(path)), "sha256": _sha256(path)}
    payload = {
        "schema_version": 1,
        "profile": "original",
        "config_sha256": _sha256(CONFIG),
        "geometry_config_sha256": _sha256(GEOMETRY_CONFIG),
        "completed": completed,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
