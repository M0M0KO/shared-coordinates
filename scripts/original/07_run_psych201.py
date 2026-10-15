#!/usr/bin/env python3
"""Extract and analyze non-Psych-101 Psych-201-discrete original-profile sessions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.lib.format import open_memmap
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cognitive_manifold.data.marked_sessions import parse_bool_flag, parse_marked_choices
from cognitive_manifold.models.choice_extractor import FrozenChoiceExtractor

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/Psych-201-discrete/data"
CONFIG = ROOT / "configs/original/psych201_frozen.yaml"
CONTEXT_QC_CONFIG = ROOT / "configs/original/psych201_context_qc.yaml"
SHARED_SELECTION = ROOT / "results/original/selected_shared.json"
RETEST_RESULT = ROOT / "results/original/retest_transfer.parquet"
CACHE = ROOT / "cache/original/psych201"
RESULT = ROOT / "results/original/psych201_generalization.parquet"
STATISTICS = ROOT / "results/original/psych201_mixed_effects.json"
MANIFEST = ROOT / "manifests/private/original/psych201.json"
MODEL_PATHS = {
    "llama_base_8b": ROOT / "models/Llama-3.1-8B",
    "minitaur_8b": ROOT / "models/Llama-3.1-Minitaur-8B",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _pseudonym(study: object, participant: object) -> str:
    value = f"psych201|{study}|{participant}".encode()
    return "p201_" + hashlib.sha256(value).hexdigest()[:20]


def _source_files() -> list[Path]:
    files = sorted(SOURCE.glob("*.parquet"))
    if len(files) != 4:
        raise RuntimeError(f"Expected four Psych-201 shards, found {len(files)}")
    return files


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        not parse_bool_flag(psych101) and not parse_bool_flag(psych101_test)
        for psych101, psych101_test in zip(
            frame["is_psych101"], frame["is_psych101_test"], strict=True
        )
    ]
    return frame.loc[keep].reset_index(drop=True)


def _apply_context_qc(
    frame: pd.DataFrame,
    shard: int,
    qc: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply outcome-blind, source-hash-pinned model-context exclusions."""
    retained = frame.copy()
    retained["frozen_source_session_row"] = np.arange(len(retained), dtype=np.int64)
    evidence: list[dict[str, Any]] = []
    excluded_rows: set[int] = set()
    for exclusion in qc.get("exclusions", []):
        if int(exclusion["source_shard"]) != shard:
            continue
        source_row = int(exclusion["source_session_row"])
        if source_row < 0 or source_row >= len(retained):
            raise RuntimeError(f"Psych-201 context-QC row is absent: shard={shard}, row={source_row}")
        text = str(retained.iloc[source_row]["text"])
        observed_sha256 = hashlib.sha256(text.encode()).hexdigest()
        if observed_sha256 != str(exclusion["text_sha256"]):
            raise RuntimeError(
                f"Psych-201 context-QC source hash changed: shard={shard}, row={source_row}"
            )
        responses = len(parse_marked_choices(text))
        if responses != int(exclusion["response_count"]):
            raise RuntimeError(
                f"Psych-201 context-QC response count changed: shard={shard}, row={source_row}"
            )
        if int(exclusion["observed_tokens"]) <= int(qc["maximum_model_tokens"]):
            raise RuntimeError("Psych-201 context-QC exclusion does not exceed the frozen limit")
        excluded_rows.add(source_row)
        evidence.append(
            {
                "source_shard": shard,
                "source_session_row": source_row,
                "text_sha256": observed_sha256,
                "observed_tokens": int(exclusion["observed_tokens"]),
                "response_count": responses,
                "reason": str(exclusion["reason"]),
            }
        )
    retained = retained.loc[
        ~retained["frozen_source_session_row"].isin(excluded_rows)
    ].reset_index(drop=True)
    return retained, evidence


def extract_shard(model_name: str, shard: int, overwrite: bool) -> dict[str, Any]:
    if not RETEST_RESULT.exists():
        raise RuntimeError("O7 is sealed until O6 confirmation results are frozen")
    selection = json.loads(SHARED_SELECTION.read_text())
    layer = int(selection["layer"])
    source = _source_files()[shard]
    frame = _eligible(pd.read_parquet(source))
    context_qc = yaml.safe_load(CONTEXT_QC_CONFIG.read_text())
    frame, context_exclusions = _apply_context_qc(frame, shard, context_qc)
    response_counts = [len(parse_marked_choices(str(text))) for text in frame["text"]]
    if not all(response_counts):
        raise RuntimeError(f"A retained Psych-201 session in shard {shard} has no responses")
    total_responses = int(sum(response_counts))
    destination = CACHE / model_name / f"shard_{shard:02d}"
    metadata_path = destination / "metadata.parquet"
    state_path = destination / f"layer_{layer:02d}.npy"
    if metadata_path.exists() and state_path.exists() and not overwrite:
        metadata = pd.read_parquet(metadata_path)
        return {
            "status": "reused",
            "model": model_name,
            "shard": shard,
            "sessions": len(frame),
            "responses": len(metadata),
            "excluded_sessions": len(context_exclusions),
            "excluded_responses": sum(row["response_count"] for row in context_exclusions),
            "context_qc_sha256": _sha256(CONTEXT_QC_CONFIG),
            "metadata_sha256": _sha256(metadata_path),
            "state_sha256": _sha256(state_path),
        }
    extractor = FrozenChoiceExtractor(MODEL_PATHS[model_name], [layer])
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / f"layer_{layer:02d}.partial.npy"
    states = open_memmap(
        partial,
        mode="w+",
        dtype=np.float16,
        shape=(total_responses, int(extractor.model.config.hidden_size)),
    )
    rows: list[dict[str, Any]] = []
    completed_responses = 0
    maximum_tokens = 0
    for session_index, row in enumerate(frame.itertuples(index=False)):
        extracted, evidence = extractor.extract_marked_likelihood_session(str(row.text))
        maximum_tokens = max(maximum_tokens, evidence["session_tokens"])
        participant = _pseudonym(row.study, row.participant)
        for result in extracted:
            states[completed_responses] = (
                result.hidden_states[layer].numpy().astype(np.float16, copy=False)
            )
            rows.append(
                {
                    "source_shard": shard,
                    "source_session_row": int(row.frozen_source_session_row),
                    "study": str(row.study),
                    "experiment": str(row.experiment),
                    "participant": participant,
                    "response_index": result.response_index,
                    "response": result.response,
                    "response_tokens": result.response_tokens,
                    "nll": result.nll,
                    "state_token_count": result.state_token_count,
                    "session_token_count": evidence["session_tokens"],
                    "state_row": completed_responses,
                    "model": model_name,
                    "layer": layer,
                    "dataset_revision": "060062064d00766ea0b5666c73268329d0f556b6",
                    "psych201_config_sha256": _sha256(CONFIG),
                }
            )
            completed_responses += 1
        completed_sessions = session_index + 1
        if completed_sessions % 100 == 0:
            print(
                json.dumps(
                    {
                        "model": model_name,
                        "shard": shard,
                        "sessions": completed_sessions,
                        "responses": completed_responses,
                    }
                ),
                flush=True,
            )
    if completed_responses != total_responses:
        raise RuntimeError("Psych-201 response count changed between indexing and extraction")
    states.flush()
    partial.replace(state_path)
    metadata = pd.DataFrame(rows)
    metadata.to_parquet(metadata_path, index=False, compression="zstd")
    return {
        "status": "completed",
        "model": model_name,
        "shard": shard,
        "sessions": len(frame),
        "responses": len(metadata),
        "excluded_sessions": len(context_exclusions),
        "excluded_responses": sum(row["response_count"] for row in context_exclusions),
        "context_qc_sha256": _sha256(CONTEXT_QC_CONFIG),
        "maximum_tokens": maximum_tokens,
        "metadata_sha256": _sha256(metadata_path),
        "state_sha256": _sha256(state_path),
    }


def _experiment_basis(
    features: np.ndarray,
    dimension: int,
    maximum: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    if len(features) > maximum:
        selected = np.sort(rng.choice(len(features), maximum, replace=False))
        features = features[selected]
    centered = features - features.mean(axis=0, keepdims=True)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:dimension].T
    variance = np.square(singular_values)
    participation_ratio = float(variance.sum() ** 2 / np.square(variance).sum())
    return basis.astype(np.float32), participation_ratio


def analyze() -> None:
    if not all(
        (CACHE / model / f"shard_{shard:02d}/metadata.parquet").exists()
        for model in MODEL_PATHS
        for shard in range(4)
    ):
        raise RuntimeError("Psych-201 extraction is incomplete")
    selection = json.loads(SHARED_SELECTION.read_text())
    model_name = selection["model"]
    layer = int(selection["layer"])
    dimension = int(selection["dimension"])
    preprocessing = np.load(
        ROOT / "cache/original/preprocessing" / model_name / f"layer_{layer:02d}/pca.npz"
    )
    shared_basis = np.load(
        ROOT / "cache/original/shared_final" / model_name / f"layer_{layer:02d}/shared_basis.npy"
    )
    meta: dict[str, pd.DataFrame] = {}
    projected: dict[str, np.ndarray] = {}
    for model in MODEL_PATHS:
        model_frames: list[pd.DataFrame] = []
        feature_frames: list[np.ndarray] = []
        for shard in range(4):
            root = CACHE / model / f"shard_{shard:02d}"
            metadata = pd.read_parquet(root / "metadata.parquet")
            raw = np.load(root / f"layer_{layer:02d}.npy", mmap_mode="r")
            if len(metadata) != len(raw):
                raise RuntimeError(f"Psych-201 state mismatch for {model}/shard {shard}")
            model_frames.append(metadata)
            pca_center = (
                preprocessing["pca_center"]
                if "pca_center" in preprocessing.files
                else np.zeros(preprocessing["components"].shape[1], dtype=np.float32)
            )
            feature_frames.append(
                (np.asarray(raw, dtype=np.float32) - preprocessing["mean"] - pca_center)
                @ preprocessing["components"][:512].T
            )
        meta[model] = pd.concat(model_frames, ignore_index=True)
        projected[model] = np.concatenate(feature_frames)
    keys = ["source_shard", "source_session_row", "response_index", "experiment"]
    if not meta["llama_base_8b"][keys].equals(meta["minitaur_8b"][keys]):
        raise RuntimeError("Base and Minitaur Psych-201 response rows do not align")
    frozen = yaml.safe_load(CONFIG.read_text())
    maximum = int(frozen["task_specific_basis"]["maximum_responses_per_experiment"])
    minimum = int(frozen["task_specific_basis"]["minimum_responses"])
    seed = int(frozen["task_specific_basis"]["sampling_seed"])
    rows: list[dict[str, Any]] = []
    reference = meta["minitaur_8b"]
    for experiment, indices in reference.groupby("experiment", sort=True).indices.items():
        selected = np.asarray(indices, dtype=np.int64)
        if len(selected) < max(minimum, dimension):
            continue
        bases: dict[str, np.ndarray] = {}
        intrinsic = float("nan")
        for model in MODEL_PATHS:
            basis, model_intrinsic = _experiment_basis(
                projected[model][selected], dimension, maximum, seed
            )
            bases[model] = basis
            if model == "minitaur_8b":
                intrinsic = model_intrinsic
        alignment = {
            model: float(np.linalg.norm(shared_basis.T @ basis, ord="fro") ** 2 / dimension)
            for model, basis in bases.items()
        }
        base_nll = float(meta["llama_base_8b"].iloc[selected]["nll"].mean())
        minitaur_nll = float(meta["minitaur_8b"].iloc[selected]["nll"].mean())
        rows.append(
            {
                "study": str(reference.iloc[selected[0]]["study"]),
                "experiment": experiment,
                "responses": len(selected),
                "participants": int(reference.iloc[selected]["participant"].nunique()),
                "llama_nll": base_nll,
                "minitaur_nll": minitaur_nll,
                "fine_tuning_gain": base_nll - minitaur_nll,
                "llama_alignment": alignment["llama_base_8b"],
                "minitaur_alignment": alignment["minitaur_8b"],
                "behavioral_alignment_increment": (
                    alignment["minitaur_8b"] - alignment["llama_base_8b"]
                ),
                "intrinsic_dimension": intrinsic,
                "mean_log_context_length": float(
                    np.log1p(reference.iloc[selected]["state_token_count"]).mean()
                ),
                "response_set_complexity": int(reference.iloc[selected]["response"].nunique()),
                "layer": layer,
                "dimension": dimension,
                "psych201_config_sha256": _sha256(CONFIG),
            }
        )
    output = pd.DataFrame(rows)
    if output["study"].nunique() < 3:
        raise RuntimeError("Psych-201 leave-one-study-out analysis requires at least three studies")
    predictors = [
        "behavioral_alignment_increment",
        "intrinsic_dimension",
        "mean_log_context_length",
        "response_set_complexity",
    ]
    predictions = np.full(len(output), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for study in sorted(output["study"].unique()):
        test = output["study"] == study
        train = ~test
        estimator = make_pipeline(StandardScaler(), LinearRegression())
        estimator.fit(output.loc[train, predictors], output.loc[train, "fine_tuning_gain"])
        predictions[test] = estimator.predict(output.loc[test, predictors])
        fold_rows.append(
            {
                "study": study,
                "training_experiments": int(train.sum()),
                "test_experiments": int(test.sum()),
            }
        )
    output["loso_prediction"] = predictions
    calibration = LinearRegression().fit(predictions.reshape(-1, 1), output["fine_tuning_gain"])
    from statsmodels.regression.mixed_linear_model import MixedLM

    exogenous = output[predictors].copy()
    exogenous = (exogenous - exogenous.mean()) / exogenous.std(ddof=0).replace(0, 1.0)
    exogenous.insert(0, "intercept", 1.0)
    mixed = MixedLM(
        output["fine_tuning_gain"].to_numpy(np.float64),
        exogenous.to_numpy(np.float64),
        groups=output["study"].astype(str).to_numpy(),
    ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    parameter_names = exogenous.columns.tolist()
    beta_index = parameter_names.index("behavioral_alignment_increment")
    rho = spearmanr(output["fine_tuning_gain"], predictions)
    statistics = {
        "model": "G ~ behavioral_alignment_increment + intrinsic_dimension + "
        "log_context_length + response_set_complexity + (1|study)",
        "predictors_standardized_for_numerical_stability": True,
        "fixed_effects": {
            name: {
                "estimate": float(mixed.fe_params[index]),
                "standard_error": float(mixed.bse_fe[index]),
                "p_value": float(mixed.pvalues[index]),
            }
            for index, name in enumerate(parameter_names)
        },
        "primary_beta_1": float(mixed.fe_params[beta_index]),
        "primary_beta_1_p": float(mixed.pvalues[beta_index]),
        "random_intercept_variance": float(mixed.cov_re[0, 0]),
        "loso_r2": float(r2_score(output["fine_tuning_gain"], predictions)),
        "loso_spearman": float(rho.statistic),
        "loso_spearman_p": float(rho.pvalue),
        "loso_calibration_slope": float(calibration.coef_[0]),
        "loso_calibration_intercept": float(calibration.intercept_),
        "folds": fold_rows,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(RESULT, index=False, compression="zstd")
    STATISTICS.write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(RESULT), "experiments": len(rows), **statistics}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("extract", "analyze"))
    parser.add_argument("--model", choices=(*MODEL_PATHS, "all"), default="all")
    parser.add_argument("--shard", choices=("0", "1", "2", "3", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"completed": {}}
    manifest.update(
        {
            "schema_version": 1,
            "profile": "original",
            "psych201_config_sha256": _sha256(CONFIG),
            "context_qc_sha256": _sha256(CONTEXT_QC_CONFIG),
        }
    )
    if args.action == "extract":
        models = tuple(MODEL_PATHS) if args.model == "all" else (args.model,)
        shards = tuple(range(4)) if args.shard == "all" else (int(args.shard),)
        for model in models:
            for shard in shards:
                evidence = extract_shard(model, shard, args.overwrite)
                manifest["completed"][f"{model}/shard_{shard:02d}"] = evidence
                MANIFEST.parent.mkdir(parents=True, exist_ok=True)
                MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        analyze()
        manifest["analysis"] = {"result_sha256": _sha256(RESULT)}
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
