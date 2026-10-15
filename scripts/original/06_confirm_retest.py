#!/usr/bin/env python3
"""Build, extract and evaluate the untouched original SRO confirmation cohort."""

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

from cognitive_manifold.data.splits import participant_pseudonym
from cognitive_manifold.individuality.baselines import (
    FixedFeatureReadout,
    evaluate_fixed_feature_readout,
)
from cognitive_manifold.individuality.model import (
    IndividualReadout,
    IndividualTaskData,
    evaluate_new_coordinates,
    infer_new_coordinates,
)
from cognitive_manifold.individuality.recovery import (
    canonical_correlations,
    global_correlation_permutation_test,
    identity_permutation_test,
)
from cognitive_manifold.models.choice_extractor import FrozenChoiceExtractor
from cognitive_manifold.prompts.sro import build_session
from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/original/main.yaml"
PRIVATE_IDS = ROOT / "data/derived/original/sro_retest_ids.txt"
SESSION_ROOT = ROOT / "data/derived/original/sro_sessions"
STATE_ROOT = ROOT / "cache/original/confirmation_states"
PRIVATE_SELECTION = ROOT / "results/original/selected_private.json"
SHARED_SELECTION = ROOT / "results/original/selected_shared.json"
RESULT = ROOT / "results/original/retest_transfer.parquet"
COORDINATES = ROOT / "manifests/private/original/retest_coordinates.npz"
MANIFEST = ROOT / "manifests/private/original/retest_confirmation.json"
BASELINE_CONFIG = ROOT / "configs/original/baselines_frozen.yaml"
FACTOR_MODEL = ROOT / "cache/original/original_baselines/factor_readout.pt"
FACTOR_SCORES = ROOT / "manifests/private/original/sro_five_factor_scores.npz"
RECOVERY_ROOT = ROOT / "cache/original/original_coordinate_recovery"
RECOVERY_RESULT = ROOT / "results/original/coordinate_recovery_confirmation.parquet"
RECOVERY_SUMMARY = ROOT / "results/original/coordinate_recovery_summary.json"
ELIGIBILITY = ROOT / "results/original/confirmation_eligibility.json"
PERSONA_ROOT = ROOT / "cache/original/persona"
PERSONA_RESULT = ROOT / "results/original/persona_comparison.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _guard() -> dict[str, Any]:
    if not SHARED_SELECTION.exists() or not PRIVATE_SELECTION.exists():
        raise RuntimeError(
            "Confirmation behavior is sealed until shared dimension, layer, private rank and "
            "regularization are frozen"
        )
    return yaml.safe_load(CONFIG_PATH.read_text())


def _raw_ids() -> set[str]:
    values = {line.strip() for line in PRIVATE_IDS.read_text().splitlines() if line.strip()}
    if len(values) != 150:
        raise RuntimeError(f"Expected 150 frozen confirmation IDs, found {len(values)}")
    return values


def build_sessions(
    task: str,
    timepoint: str,
    config: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    destination = SESSION_ROOT / f"confirmation_{timepoint.lower()}" / f"{task}.parquet"
    if destination.exists() and not overwrite:
        frame = pd.read_parquet(destination, columns=["participant", "validity"])
        return {
            "task": task,
            "timepoint": timepoint,
            "status": "reused",
            "rows": len(frame),
            "participants": frame["participant"].nunique(),
            "primary": int((frame["validity"] == "primary").sum()),
            "sha256": _sha256(destination),
        }
    release = (
        config["data"]["sro"]["complete_path"]
        if timepoint == "T1"
        else config["data"]["sro"]["retest_path"]
    )
    source = ROOT / "data/raw/Self_Regulation_Ontology" / release / f"{task}.csv.gz"
    raw = pd.read_csv(source, compression="gzip", low_memory=False)
    raw["worker_id"] = raw["worker_id"].astype(str).str.strip()
    raw = raw.loc[raw["worker_id"].isin(_raw_ids())]
    rows: list[dict[str, Any]] = []
    for participant, frame in raw.groupby("worker_id", sort=True):
        rows.extend(build_session(task, participant, frame, timepoint))
    records = pd.DataFrame.from_records(rows)
    if records.empty:
        raise RuntimeError(f"No confirmation rows reconstructed for {timepoint}/{task}")
    records["config_hash"] = _sha256(CONFIG_PATH)
    records["dataset_revision"] = config["data"]["sro"]["revision"]
    records["model_revision"] = "not_applicable_parser_stage"
    if not records["prompt_prefix"].str.endswith("\n<<").all():
        raise RuntimeError(f"Open-response marker gate failed for {timepoint}/{task}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(destination, index=False, compression="zstd")
    return {
        "task": task,
        "timepoint": timepoint,
        "status": "built",
        "rows": len(records),
        "participants": records["participant"].nunique(),
        "primary": int((records["validity"] == "primary").sum()),
        "sha256": _sha256(destination),
    }


def extract_task(
    task: str,
    timepoint: str,
    config: dict[str, Any],
    extractor: FrozenChoiceExtractor,
    model_name: str,
    layer: int,
    overwrite: bool,
) -> dict[str, Any]:
    source = SESSION_ROOT / f"confirmation_{timepoint.lower()}" / f"{task}.parquet"
    destination = STATE_ROOT / timepoint / task
    metadata_path = destination / "metadata.parquet"
    state_path = destination / f"layer_{layer:02d}.npy"
    if metadata_path.exists() and state_path.exists() and not overwrite:
        metadata = pd.read_parquet(metadata_path)
        return {
            "task": task,
            "timepoint": timepoint,
            "status": "reused",
            "rows": len(metadata),
            "metadata_sha256": _sha256(metadata_path),
            "state_sha256": _sha256(state_path),
        }
    frame = pd.read_parquet(source)
    frame = (
        frame.loc[frame["validity"] == "primary"]
        .sort_values(["participant", "trial"], kind="stable")
        .reset_index(drop=True)
    )
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / f"layer_{layer:02d}.partial.npy"
    states = open_memmap(
        partial,
        mode="w+",
        dtype=np.float16,
        shape=(len(frame), int(extractor.model.config.hidden_size)),
    )
    rows: list[dict[str, Any]] = []
    completed = 0
    cache_evidence = {"cache_reused": 0, "cache_crop_events": 0, "full_recomputations": 0}
    for _participant, session in frame.groupby("participant", sort=False):
        results, evidence = extractor.extract_session(
            session["prompt_prefix"].tolist(),
            session["candidate_responses"].tolist(),
            session["response"].tolist(),
        )
        for key in cache_evidence:
            cache_evidence[key] += evidence[key]
        for row, result in zip(session.itertuples(index=False), results, strict=True):
            states[completed] = result.hidden_states[layer].numpy().astype(np.float16, copy=False)
            rows.append(
                {
                    "participant": row.participant,
                    "timepoint": timepoint,
                    "task": task,
                    "task_family": row.task_family,
                    "trial": row.trial,
                    "response": row.response,
                    "candidate_responses": row.candidate_responses,
                    "baseline_model_nll": result.nll,
                    "token_count": result.token_count,
                    "state_row": completed,
                    "model": model_name,
                    "layer": layer,
                    "config_hash": _sha256(CONFIG_PATH),
                }
            )
            completed += 1
        if completed and completed % 1000 < len(session):
            print(
                json.dumps({"timepoint": timepoint, "task": task, "completed": completed}),
                flush=True,
            )
    states.flush()
    partial.replace(state_path)
    metadata = pd.DataFrame(rows)
    metadata.to_parquet(metadata_path, index=False, compression="zstd")
    return {
        "task": task,
        "timepoint": timepoint,
        "status": "completed",
        "rows": len(metadata),
        **cache_evidence,
        "metadata_sha256": _sha256(metadata_path),
        "state_sha256": _sha256(state_path),
    }


def _fixed_model() -> tuple[
    IndividualReadout,
    dict[str, tuple[str, ...]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    shared = json.loads(SHARED_SELECTION.read_text())
    private = json.loads(PRIVATE_SELECTION.read_text())
    model_name = shared["model"]
    layer = int(shared["layer"])
    shared_root = ROOT / "cache/original/shared_final" / model_name / f"layer_{layer:02d}"
    population = torch.load(shared_root / "model.pt", map_location="cpu", weights_only=True)
    state = population["state_dict"]
    weights = {
        key.split(".", 1)[1]: value for key, value in state.items() if key.startswith("weights.")
    }
    biases = {
        key.split(".", 1)[1]: value for key, value in state.items() if key.startswith("biases.")
    }
    individual_payload = torch.load(
        ROOT / "cache/original/individuality_final/model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = IndividualReadout(
        int(population["shared_dim"]),
        int(private["rank"]),
        len(individual_payload["participant_remap"]),
        weights,
        biases,
    ).to("cuda:0")
    model.load_state_dict(individual_payload["state_dict"])
    model.eval()
    preprocessing = (
        ROOT / "cache/original/preprocessing" / model_name / f"layer_{layer:02d}" / "pca.npz"
    )
    pca = np.load(preprocessing)
    shared_basis = np.load(shared_root / "shared_basis.npy")
    classes = {task: tuple(values) for task, values in population["classes"].items()}
    pca_center = (
        pca["pca_center"]
        if "pca_center" in pca.files
        else np.zeros(pca["components"].shape[1], dtype=np.float32)
    )
    projection_origin = pca["mean"] + pca_center
    return model, classes, projection_origin, pca["components"], shared_basis


def _load_factor_model(
    population: IndividualReadout,
) -> FixedFeatureReadout:
    payload = torch.load(FACTOR_MODEL, map_location="cpu", weights_only=True)
    model = FixedFeatureReadout(
        int(payload["shared_dim"]),
        int(payload["feature_dim"]),
        int(payload["rank"]),
        {task: value.detach().cpu() for task, value in population.population_weights.items()},
        {task: value.detach().cpu() for task, value in population.population_biases.items()},
    ).to("cuda:0")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _confirmation_factor_features(participant_names: list[str], timepoint: str) -> np.ndarray:
    with np.load(FACTOR_SCORES, allow_pickle=False) as payload:
        names = payload["confirmation_participant"].astype(str).tolist()
        values = payload[f"confirmation_{timepoint.lower()}"].copy()
    index = {value: row for row, value in enumerate(names)}
    if set(participant_names) != set(names):
        raise RuntimeError("Confirmation factor-score IDs do not match state participants")
    return np.stack([values[index[value]] for value in participant_names]).astype(np.float32)


def _load_recovery_model(
    label: str,
    population: IndividualReadout,
) -> IndividualReadout:
    payload = torch.load(RECOVERY_ROOT / f"model_{label}.pt", map_location="cpu", weights_only=True)
    model = IndividualReadout(
        int(population.raw_directions.shape[0]),
        int(payload["rank"]),
        len(payload["participant_remap"]),
        {task: value.detach().cpu() for task, value in population.population_weights.items()},
        {task: value.detach().cpu() for task, value in population.population_biases.items()},
    ).to("cuda:0")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _icc_3_1(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    values = np.stack([first, second], axis=1)
    output = np.empty(values.shape[2], dtype=np.float64)
    for dimension in range(values.shape[2]):
        matrix = values[:, :, dimension]
        subject_mean = matrix.mean(axis=1)
        rater_mean = matrix.mean(axis=0)
        grand = matrix.mean()
        ms_subject = matrix.shape[1] * np.var(subject_mean, ddof=1)
        residual = matrix - subject_mean[:, None] - rater_mean[None, :] + grand
        ms_error = np.square(residual).sum() / ((matrix.shape[0] - 1) * (matrix.shape[1] - 1))
        output[dimension] = (ms_subject - ms_error) / (
            ms_subject + (matrix.shape[1] - 1) * ms_error
        )
    return output


def _eligible_confirmation_participants(
    t1: dict[str, IndividualTaskData],
    t2: dict[str, IndividualTaskData],
    source_families: set[str],
    target_families: set[str],
    minimum_source_families: int,
    minimum_target_families: int,
) -> list[int]:
    participants = sorted(
        {
            int(participant)
            for values in (*t1.values(), *t2.values())
            for participant in np.unique(values.participants)
        }
    )
    eligible: list[int] = []
    for participant in participants:
        observed_source = {
            values.family
            for values in t1.values()
            if values.family in source_families and np.any(values.participants == participant)
        }
        observed_target = {
            values.family
            for values in t2.values()
            if values.family in target_families and np.any(values.participants == participant)
        }
        if (
            len(observed_source) >= minimum_source_families
            and len(observed_target) >= minimum_target_families
        ):
            eligible.append(participant)
    return eligible


def _coordinate_recovery(
    population: IndividualReadout,
    t1: dict[str, IndividualTaskData],
    t2: dict[str, IndividualTaskData],
    names: list[str],
    participant_ids: list[int],
    config: dict[str, Any],
    regularization: float,
) -> dict[str, Any]:
    split = config["individuality"]["preregistered_family_split"]
    coordinates: dict[str, np.ndarray] = {}
    for label in ("A", "B"):
        model = _load_recovery_model(label, population)
        inferred = infer_new_coordinates(
            model,
            t1,
            participant_ids=participant_ids,
            source_families=set(split[label]),
            regularization=regularization,
        )
        coordinates[label] = inferred.detach().cpu().numpy()
    alignment = np.load(RECOVERY_ROOT / "alignment.npz")["R_align"]
    aligned_b = coordinates["B"] @ alignment
    permutations = int(config["statistics"]["permutation_replicates"])
    observed, p_value, null = identity_permutation_test(
        coordinates["A"],
        aligned_b,
        replicates=permutations,
        seed=int(config["execution"]["seed"]) + 901,
    )
    global_correlation, global_p, global_null = global_correlation_permutation_test(
        coordinates["A"],
        aligned_b,
        replicates=permutations,
        seed=int(config["execution"]["seed"]) + 903,
    )
    canonical = canonical_correlations(coordinates["A"], aligned_b)

    primary_t1 = (
        infer_new_coordinates(
            population,
            t1,
            participant_ids=participant_ids,
            source_families=set(split["A"]),
            regularization=regularization,
        )
        .detach()
        .cpu()
        .numpy()
    )
    primary_t2 = (
        infer_new_coordinates(
            population,
            t2,
            participant_ids=participant_ids,
            source_families=set(split["A"]),
            regularization=regularization,
        )
        .detach()
        .cpu()
        .numpy()
    )
    time_retrieval, time_p, time_null = identity_permutation_test(
        primary_t1,
        primary_t2,
        replicates=permutations,
        seed=int(config["execution"]["seed"]) + 902,
    )
    within = np.linalg.norm(primary_t1 - primary_t2, axis=1)
    between_matrix = np.linalg.norm(primary_t1[:, None, :] - primary_t2[None, :, :], axis=2)
    between = between_matrix[~np.eye(len(names), dtype=bool)]
    frame = pd.DataFrame(
        {
            "participant": names,
            "independent_A_B_aligned_distance": np.linalg.norm(
                coordinates["A"] - aligned_b, axis=1
            ),
            "T1_T2_coordinate_distance": within,
            "rank": coordinates["A"].shape[1],
            "config_hash": _sha256(CONFIG_PATH),
        }
    )
    RECOVERY_RESULT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(RECOVERY_RESULT, index=False, compression="zstd")
    np.savez(
        RECOVERY_ROOT / "confirmation_statistics.npz",
        participant=np.asarray(names),
        independent_A=coordinates["A"],
        independent_B=coordinates["B"],
        aligned_B=aligned_b,
        retrieval_null=null,
        global_correlation_null=global_null,
        primary_T1_A=primary_t1,
        primary_T2_A=primary_t2,
        time_retrieval_null=time_null,
    )
    summary = {
        "independent_global_procrustes_correlation": global_correlation,
        "independent_global_procrustes_permutation_p": global_p,
        "independent_canonical_correlations": canonical.tolist(),
        "independent_top1_retrieval": observed,
        "independent_top1_chance": 1 / len(names),
        "independent_top1_permutation_p": p_value,
        "time_coordinate_correlations": [
            float(np.corrcoef(primary_t1[:, index], primary_t2[:, index])[0, 1])
            for index in range(primary_t1.shape[1])
        ],
        "time_coordinate_icc_3_1": _icc_3_1(primary_t1, primary_t2).tolist(),
        "time_top1_retrieval": time_retrieval,
        "time_top1_permutation_p": time_p,
        "mean_within_person_distance": float(within.mean()),
        "mean_between_person_distance": float(between.mean()),
        "participants": len(names),
        "permutations": permutations,
        "alignment_fit_cohort": "development_only",
    }
    RECOVERY_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _persona_templates() -> dict[str, str]:
    frozen = yaml.safe_load(BASELINE_CONFIG.read_text())["persona"]
    source = pd.read_csv(ROOT / frozen["source_complete"], index_col=0)
    source.index = source.index.astype(str).str.strip()
    confirmation_ids = _raw_ids()
    development_ids = sorted(set(source.index) - confirmation_ids)
    columns = [value for value in source.columns if "_survey." in str(value)]
    if not columns:
        raise RuntimeError("No SRO questionnaire columns matched the frozen persona rule")
    development = source.loc[development_ids, columns]
    mean = development.mean(axis=0, skipna=True)
    scale = development.std(axis=0, skipna=True, ddof=0).replace(0, 1.0)
    standardized = (source.loc[:, columns] - mean) / scale
    instruments: dict[str, list[str]] = {}
    for column in columns:
        instruments.setdefault(str(column).split(".", 1)[0], []).append(column)
    templates: dict[str, str] = {}
    decimals = int(frozen["rounding_decimals"])
    for raw_id in sorted(confirmation_ids):
        scores: list[str] = []
        for instrument in sorted(instruments):
            value = standardized.loc[raw_id, instruments[instrument]].mean(skipna=True)
            if pd.notna(value):
                scores.append(f"{instrument}={float(value):.{decimals}f}")
        if not scores:
            raise RuntimeError(f"No questionnaire persona values for {raw_id}")
        templates[participant_pseudonym(raw_id)] = str(frozen["fixed_template"]).format(
            scores=", ".join(scores)
        )
    return templates


def _run_persona_task(
    task: str,
    extractor: FrozenChoiceExtractor,
    population: IndividualReadout,
    classes: dict[str, tuple[str, ...]],
    mean: np.ndarray,
    components: np.ndarray,
    shared_basis: np.ndarray,
    templates: dict[str, str],
    layer: int,
    overwrite: bool,
) -> dict[str, Any]:
    destination = PERSONA_ROOT / f"{task}.parquet"
    if destination.exists() and not overwrite:
        frame = pd.read_parquet(destination)
        return {
            "task": task,
            "status": "reused",
            "rows": len(frame),
            "sha256": _sha256(destination),
        }
    source = pd.read_parquet(SESSION_ROOT / "confirmation_t2" / f"{task}.parquet")
    source = (
        source.loc[source["validity"] == "primary"]
        .sort_values(["participant", "trial"], kind="stable")
        .reset_index(drop=True)
    )
    class_index = {value: index for index, value in enumerate(classes[task])}
    weights = population.population_weights[task].detach().cpu().numpy()
    bias = population.population_biases[task].detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    completed = 0
    for participant, session in source.groupby("participant", sort=False):
        persona = templates.get(str(participant))
        if persona is None:
            raise RuntimeError(f"Missing persona for confirmation participant {participant}")
        prefixes = [f"{persona}\n\n{value}" for value in session["prompt_prefix"]]
        extracted, _evidence = extractor.extract_session(
            prefixes,
            session["candidate_responses"].tolist(),
            session["response"].tolist(),
        )
        for source_row, result in zip(session.itertuples(index=False), extracted, strict=True):
            raw = result.hidden_states[layer].numpy().astype(np.float32, copy=False)
            projected = (raw - mean) @ components[:512].T @ shared_basis
            logits = projected @ weights.T + bias
            log_probabilities = logits - np.logaddexp.reduce(logits)
            label = class_index[source_row.response]
            rows.append(
                {
                    "participant": participant,
                    "task": task,
                    "task_family": SPECS[task].family,
                    "trial": source_row.trial,
                    "persona_nll": float(-log_probabilities[label]),
                    "layer": layer,
                    "config_hash": _sha256(CONFIG_PATH),
                    "baseline_config_sha256": _sha256(BASELINE_CONFIG),
                }
            )
            completed += 1
        if completed and completed % 1000 < len(session):
            print(json.dumps({"persona_task": task, "completed": completed}), flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(destination, index=False, compression="zstd")
    return {"task": task, "status": "completed", "rows": len(rows), "sha256": _sha256(destination)}


def _finalize_persona(config: dict[str, Any]) -> None:
    target_families = set(config["individuality"]["preregistered_family_split"]["B"])
    tasks = [task for task, spec in SPECS.items() if spec.family in target_families]
    missing = [task for task in tasks if not (PERSONA_ROOT / f"{task}.parquet").exists()]
    if missing:
        raise RuntimeError(f"Persona task outputs are incomplete: {missing}")
    trials = pd.concat(
        [pd.read_parquet(PERSONA_ROOT / f"{task}.parquet") for task in tasks],
        ignore_index=True,
    )
    persona = trials.groupby(["participant", "task", "task_family"], as_index=False).agg(
        personalized_nll=("persona_nll", "mean"), trials=("persona_nll", "size")
    )
    retest = pd.read_parquet(RESULT)
    baseline = retest.loc[
        (retest["method"] == "manifold") & (retest["analysis"] == "primary_T1_A_to_T2_B"),
        ["participant", "task", "population_nll"],
    ]
    persona = persona.merge(
        baseline,
        on=["participant", "task"],
        how="inner",
        validate="one_to_one",
    )
    persona["delta_nll"] = persona["population_nll"] - persona["personalized_nll"]
    persona["individual_nll"] = persona["personalized_nll"]
    persona["analysis"] = "primary_T1_questionnaire_persona_to_T2_B"
    persona["method"] = "persona"
    persona["config_hash"] = _sha256(CONFIG_PATH)
    if len(persona) != len(baseline):
        raise RuntimeError("Persona and primary target participant-task cells do not align")
    PERSONA_RESULT.parent.mkdir(parents=True, exist_ok=True)
    persona.to_parquet(PERSONA_RESULT, index=False, compression="zstd")
    combined = pd.concat(
        [retest.loc[retest["method"] != "persona"], persona],
        ignore_index=True,
        sort=False,
    )
    combined.to_parquet(RESULT, index=False, compression="zstd")


def _analysis_data(
    timepoint: str,
    classes: dict[str, tuple[str, ...]],
    mean: np.ndarray,
    components: np.ndarray,
    shared_basis: np.ndarray,
    layer: int,
) -> tuple[dict[str, IndividualTaskData], list[str]]:
    metadata_by_task: dict[str, pd.DataFrame] = {}
    participants: set[str] = set()
    for task in SPECS:
        metadata = pd.read_parquet(STATE_ROOT / timepoint / task / "metadata.parquet")
        metadata_by_task[task] = metadata
        participants.update(metadata["participant"].unique())
    names = sorted(participants)
    participant_index = {value: index for index, value in enumerate(names)}
    output: dict[str, IndividualTaskData] = {}
    for task, spec in SPECS.items():
        metadata = metadata_by_task[task]
        raw = np.load(
            STATE_ROOT / timepoint / task / f"layer_{layer:02d}.npy",
            mmap_mode="r",
        )
        projected = (np.asarray(raw, dtype=np.float32) - mean) @ components[:512].T
        states = projected @ shared_basis
        class_index = {value: index for index, value in enumerate(classes[task])}
        labels = metadata["response"].map(class_index)
        if labels.isna().any():
            raise RuntimeError(f"Unknown confirmation response class in {timepoint}/{task}")
        output[task] = IndividualTaskData(
            states=states.astype(np.float32),
            labels=labels.to_numpy(np.int64),
            participants=metadata["participant"].map(participant_index).to_numpy(np.int32),
            folds=np.full(len(metadata), -1, dtype=np.int8),
            family=spec.family,
            classes=classes[task],
        )
    return output, names


@torch.inference_mode()
def _evaluate_response_bias(
    model: IndividualReadout,
    t1: dict[str, IndividualTaskData],
    t2: dict[str, IndividualTaskData],
    participant_ids: list[int],
    target_families: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, target in t2.items():
        if target.family not in target_families:
            continue
        source = t1[task]
        for participant in participant_ids:
            source_rows = np.flatnonzero(source.participants == participant)
            target_rows = np.flatnonzero(target.participants == participant)
            if not len(source_rows) or not len(target_rows):
                continue
            source_labels = (
                source.labels.detach().cpu().numpy()
                if isinstance(source.labels, torch.Tensor)
                else np.asarray(source.labels)
            )
            counts = np.bincount(source_labels[source_rows], minlength=len(source.classes))
            probabilities = (counts + 1.0) / (counts.sum() + len(counts))
            intercept = torch.as_tensor(
                np.log(probabilities) - np.log(probabilities).mean(),
                dtype=torch.float32,
                device="cuda:0",
            )
            target_index = torch.as_tensor(target_rows, dtype=torch.long, device="cuda:0")
            states = (
                torch.index_select(target.states, 0, target_index)
                if isinstance(target.states, torch.Tensor)
                else torch.as_tensor(
                    target.states[target_rows], dtype=torch.float32, device="cuda:0"
                )
            )
            labels = (
                torch.index_select(target.labels, 0, target_index)
                if isinstance(target.labels, torch.Tensor)
                else torch.as_tensor(
                    target.labels[target_rows], dtype=torch.long, device="cuda:0"
                )
            )
            population_logits = (
                states @ model.population_weights[task].T + model.population_biases[task]
            )
            population = torch.nn.functional.cross_entropy(population_logits, labels)
            biased = torch.nn.functional.cross_entropy(population_logits + intercept, labels)
            rows.append(
                {
                    "participant_index": participant,
                    "task": task,
                    "task_family": target.family,
                    "population_nll": float(population.cpu()),
                    "personalized_nll": float(biased.cpu()),
                    "individual_nll": float(biased.cpu()),
                    "delta_nll": float((population - biased).cpu()),
                    "trials": len(target_rows),
                    "method": "response_bias",
                    "analysis": "secondary_T1_same_task_bias_to_T2_B",
                    "config_hash": _sha256(CONFIG_PATH),
                }
            )
    return rows


def analyze(config: dict[str, Any]) -> None:
    model, classes, mean, components, shared_basis = _fixed_model()
    layer = int(json.loads(SHARED_SELECTION.read_text())["layer"])
    t1, t1_names = _analysis_data("T1", classes, mean, components, shared_basis, layer)
    t2, t2_names = _analysis_data("T2", classes, mean, components, shared_basis, layer)
    if t1_names != t2_names:
        raise RuntimeError("T1 and T2 confirmation participant pseudonyms do not match")
    split = config["individuality"]["preregistered_family_split"]
    source_families = set(split["A"])
    target_families = set(split["B"])
    minimum_source = int(config["individuality"]["confirmation_minimum_families_for_inference"])
    minimum_target = int(
        config["individuality"]["confirmation_minimum_distinct_families_for_evaluation"]
    )
    participant_ids = _eligible_confirmation_participants(
        t1,
        t2,
        source_families,
        target_families,
        minimum_source,
        minimum_target,
    )
    if not participant_ids:
        raise RuntimeError("No confirmation participants satisfy the frozen family gate")
    eligibility = {
        "schema_version": 1,
        "profile": "original",
        "total_confirmation_participants": len(t1_names),
        "eligible_confirmation_participants": len(participant_ids),
        "excluded_confirmation_participants": len(t1_names) - len(participant_ids),
        "minimum_source_families": minimum_source,
        "minimum_distinct_target_families": minimum_target,
        "source_families": sorted(source_families),
        "target_families": sorted(target_families),
        "config_hash": _sha256(CONFIG_PATH),
    }
    ELIGIBILITY.parent.mkdir(parents=True, exist_ok=True)
    ELIGIBILITY.write_text(json.dumps(eligibility, indent=2, sort_keys=True) + "\n")
    regularization = float(json.loads(PRIVATE_SELECTION.read_text())["regularization"])
    rows: list[dict[str, Any]] = []
    coordinates = infer_new_coordinates(
        model,
        t1,
        participant_ids=participant_ids,
        source_families=source_families,
        regularization=regularization,
    )
    for row in evaluate_new_coordinates(
        model,
        t2,
        participant_ids,
        coordinates,
        target_families,
    ):
        row["participant"] = t1_names[row.pop("participant_index")]
        row["analysis"] = "primary_T1_A_to_T2_B"
        row["method"] = "manifold"
        row["personalized_nll"] = row["individual_nll"]
        row["config_hash"] = _sha256(CONFIG_PATH)
        rows.append(row)
    all_families = {spec.family for spec in SPECS.values()}
    for held_out in sorted(all_families):
        held_coordinates = infer_new_coordinates(
            model,
            t1,
            participant_ids=participant_ids,
            source_families=all_families - {held_out},
            regularization=regularization,
        )
        for row in evaluate_new_coordinates(
            model,
            t1,
            participant_ids,
            held_coordinates,
            {held_out},
        ):
            row["participant"] = t1_names[row.pop("participant_index")]
            row["analysis"] = "primary_cross_domain_T1_leave_one_family_out"
            row["method"] = "manifold"
            row["personalized_nll"] = row["individual_nll"]
            row["config_hash"] = _sha256(CONFIG_PATH)
            rows.append(row)
        for row in evaluate_new_coordinates(
            model,
            t2,
            participant_ids,
            held_coordinates,
            {held_out},
        ):
            row["participant"] = t1_names[row.pop("participant_index")]
            row["analysis"] = f"supplement_T1_minus_{held_out}_to_T2_{held_out}"
            row["method"] = "manifold"
            row["personalized_nll"] = row["individual_nll"]
            row["config_hash"] = _sha256(CONFIG_PATH)
            rows.append(row)
    factor_model = _load_factor_model(model)
    factor_features = _confirmation_factor_features(t1_names, "T1")
    for row in evaluate_fixed_feature_readout(
        factor_model,
        t2,
        participant_ids,
        factor_features,
        set(split["B"]),
    ):
        row["participant"] = t1_names[int(row.pop("participant_index"))]
        row["analysis"] = "primary_T1_factor_to_T2_B"
        row["method"] = "sro_factor"
        row["individual_nll"] = row["personalized_nll"]
        row["config_hash"] = _sha256(CONFIG_PATH)
        rows.append(row)
    for row in _evaluate_response_bias(
        model,
        t1,
        t2,
        participant_ids,
        set(split["B"]),
    ):
        row["participant"] = t1_names[int(row.pop("participant_index"))]
        rows.append(row)
    _coordinate_recovery(
        model,
        t1,
        t2,
        [t1_names[index] for index in participant_ids],
        participant_ids,
        config,
        regularization,
    )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(RESULT, index=False, compression="zstd")
    COORDINATES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        COORDINATES,
        participant=np.asarray([t1_names[index] for index in participant_ids]),
        primary_t1_a=coordinates.detach().cpu().numpy(),
    )
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "rows": len(rows),
                "eligible_participants": len(participant_ids),
                "total_confirmation_participants": len(t1_names),
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "extract", "analyze", "persona"))
    parser.add_argument("--task", choices=(*SPECS, "all"), default="all")
    parser.add_argument("--timepoint", choices=("T1", "T2", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _guard()
    tasks = tuple(SPECS) if args.task == "all" else (args.task,)
    timepoints = ("T1", "T2") if args.timepoint == "all" else (args.timepoint,)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"steps": {}}
    if args.action == "build":
        results = [
            build_sessions(task, timepoint, config, args.overwrite)
            for timepoint in timepoints
            for task in tasks
        ]
        manifest["steps"]["build"] = results
    elif args.action == "extract":
        shared = json.loads(SHARED_SELECTION.read_text())
        model_name = shared["model"]
        layer = int(shared["layer"])
        model_path = ROOT / (
            "models/Llama-3.1-Minitaur-8B" if model_name == "minitaur_8b" else "models/Llama-3.1-8B"
        )
        extractor = FrozenChoiceExtractor(model_path, [layer])
        results = [
            extract_task(
                task,
                timepoint,
                config,
                extractor,
                model_name,
                layer,
                args.overwrite,
            )
            for timepoint in timepoints
            for task in tasks
        ]
        manifest["steps"]["extract"] = results
    elif args.action == "analyze":
        analyze(config)
        manifest["steps"]["analyze"] = {
            "result": str(RESULT),
            "sha256": _sha256(RESULT),
        }
    else:
        if not RESULT.exists():
            raise RuntimeError("Persona is sealed until the primary retest analysis is frozen")
        model, classes, mean, components, shared_basis = _fixed_model()
        shared = json.loads(SHARED_SELECTION.read_text())
        layer = int(shared["layer"])
        model_path = ROOT / "models/Llama-3.1-Minitaur-8B"
        extractor = FrozenChoiceExtractor(model_path, [layer])
        templates = _persona_templates()
        target_families = set(config["individuality"]["preregistered_family_split"]["B"])
        persona_tasks = [task for task in tasks if SPECS[task].family in target_families]
        evidence = [
            _run_persona_task(
                task,
                extractor,
                model,
                classes,
                mean,
                components,
                shared_basis,
                templates,
                layer,
                args.overwrite,
            )
            for task in persona_tasks
        ]
        if args.task == "all":
            _finalize_persona(config)
        manifest["steps"]["persona"] = {
            "tasks": evidence,
            "result": str(PERSONA_RESULT) if PERSONA_RESULT.exists() else None,
        }
    manifest.update(
        {
            "schema_version": 1,
            "profile": "original",
            "config_hash": _sha256(CONFIG_PATH),
        }
    )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
