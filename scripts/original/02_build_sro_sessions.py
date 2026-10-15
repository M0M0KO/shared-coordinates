#!/usr/bin/env python3
"""Build development-only SRO natural-language sessions and acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cognitive_manifold.data.splits import participant_fold, participant_pseudonym
from cognitive_manifold.prompts.sro import build_session
from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/original/main.yaml"
AUDIT_PATH = ROOT / "configs/original/task_audit.csv"
DERIVED = ROOT / "data/derived/original"
SESSION_ROOT = DERIVED / "sro_sessions/development"
PRIVATE_RETEST = DERIVED / "sro_retest_ids.txt"
SPLITS = ROOT / "configs/original/splits.json"
REPORT_ROOT = ROOT / "manifests/private/original/parser_review"
MANIFEST = ROOT / "manifests/private/original/sro_parser_acceptance.json"


def _config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash() -> str:
    return _sha256(CONFIG_PATH)


def _load_ids(path: Path) -> set[str]:
    frame = pd.read_csv(path, compression="gzip", usecols=["worker_id"])
    return set(frame["worker_id"].dropna().astype(str).str.strip())


def freeze_splits(config: dict[str, Any]) -> tuple[set[str], set[str]]:
    sro = ROOT / "data/raw/Self_Regulation_Ontology"
    complete_root = sro / config["data"]["sro"]["complete_path"]
    retest_root = sro / config["data"]["sro"]["retest_path"]
    reference = next(iter(SPECS))
    complete_ids = _load_ids(complete_root / f"{reference}.csv.gz")
    retest_ids: set[str] = set()
    for task in SPECS:
        retest_ids.update(_load_ids(retest_root / f"{task}.csv.gz"))
    confirmation = complete_ids & retest_ids
    development = complete_ids - confirmation
    expected = config["data"]["sro"]
    if len(complete_ids) != expected["expected_total_participants"]:
        raise RuntimeError(f"Expected 522 complete IDs, found {len(complete_ids)}")
    if len(confirmation) != expected["expected_confirmation_participants"]:
        raise RuntimeError(f"Expected 150 confirmation IDs, found {len(confirmation)}")
    if len(development) != expected["expected_development_participants"]:
        raise RuntimeError(f"Expected 372 development IDs, found {len(development)}")
    DERIVED.mkdir(parents=True, exist_ok=True)
    PRIVATE_RETEST.write_text("\n".join(sorted(confirmation)) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "profile": "original",
        "hash_contract": config["participant_split"],
        "counts": {
            "complete": len(complete_ids),
            "development": len(development),
            "confirmation": len(confirmation),
        },
        "development": [
            {"participant": participant_pseudonym(item), "fold": participant_fold(item)}
            for item in sorted(development)
        ],
        "confirmation": [participant_pseudonym(item) for item in sorted(confirmation)],
    }
    SPLITS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return development, confirmation


def _review_rows(records: pd.DataFrame, sample_size: int, seed: int) -> list[dict[str, Any]]:
    participants = sorted(records["participant"].unique())
    rng = np.random.default_rng(seed)
    selected = rng.choice(participants, size=min(sample_size, len(participants)), replace=False)
    output: list[dict[str, Any]] = []
    for participant in selected:
        session = records.loc[records["participant"] == participant].sort_values("trial")
        primary = session.loc[session["validity"] == "primary"]
        output.append(
            {
                "participant": participant,
                "events": len(session),
                "primary_events": len(primary),
                "invalid_events": int((session["validity"] == "invalid").sum()),
                "first_prefix": session.iloc[0]["prompt_prefix"] if len(session) else "",
                "middle_prefix": session.iloc[len(session) // 2]["prompt_prefix"]
                if len(session)
                else "",
                "last_prefix": session.iloc[-1]["prompt_prefix"] if len(session) else "",
                "response_set": sorted(
                    {item for items in session["candidate_responses"] for item in items}
                ),
                "review_checks": {
                    "chronological": bool(
                        list(zip(session["source_row"], session["event_index"], strict=True))
                        == sorted(zip(session["source_row"], session["event_index"], strict=True))
                    ),
                    "prefix_ends_open_response": bool(
                        session["prompt_prefix"].str.endswith("\n<<").all()
                    ),
                    "observed_in_candidates_when_valid": bool(
                        all(
                            row.response in row.candidate_responses
                            for row in session.itertuples()
                            if row.validity != "invalid"
                        )
                    ),
                },
            }
        )
    return output


def build_task(
    task: str,
    config: dict[str, Any],
    development: set[str],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    destination = SESSION_ROOT / f"{task}.parquet"
    evidence_path = REPORT_ROOT / f"{task}.jsonl"
    if destination.exists() and evidence_path.exists() and not overwrite:
        existing = pd.read_parquet(destination, columns=["participant", "validity"])
        return {
            "task": task,
            "status": "reused",
            "rows": len(existing),
            "participants": existing["participant"].nunique(),
            "primary": int((existing["validity"] == "primary").sum()),
            "parquet_sha256": _sha256(destination),
        }
    sro = ROOT / "data/raw/Self_Regulation_Ontology"
    source = sro / config["data"]["sro"]["complete_path"] / f"{task}.csv.gz"
    raw = pd.read_csv(source, compression="gzip", low_memory=False)
    raw["worker_id"] = raw["worker_id"].astype(str).str.strip()
    raw = raw.loc[raw["worker_id"].isin(development)].copy()
    sessions: list[dict[str, Any]] = []
    for participant, frame in raw.groupby("worker_id", sort=True):
        sessions.extend(build_session(task, participant, frame, "T1"))
    records = pd.DataFrame.from_records(sessions)
    if records.empty:
        raise RuntimeError(f"Parser emitted no records for {task}")
    config_hash = _config_hash()
    records["config_hash"] = config_hash
    records["dataset_revision"] = config["data"]["sro"]["revision"]
    records["model_revision"] = "not_applicable_parser_stage"
    structural = {
        "all_prefixes_end_open_marker": bool(records["prompt_prefix"].str.endswith("\n<<").all()),
        "all_valid_observed_in_candidates": bool(
            all(
                row.response in row.candidate_responses
                for row in records.itertuples()
                if row.validity != "invalid"
            )
        ),
        "no_confirmation_pseudonyms": True,
        "chronological_within_participant": True,
    }
    for _, session in records.groupby("participant", sort=False):
        pairs = list(zip(session["source_row"], session["event_index"], strict=True))
        if pairs != sorted(pairs):
            structural["chronological_within_participant"] = False
            break
    confirmation_pseudonyms = set(json.loads(SPLITS.read_text(encoding="utf-8"))["confirmation"])
    if set(records["participant"]) & confirmation_pseudonyms:
        structural["no_confirmation_pseudonyms"] = False
    if not all(structural.values()):
        raise RuntimeError(f"Acceptance failure for {task}: {structural}")
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    records.to_parquet(destination, index=False, compression="zstd")
    review = _review_rows(
        records,
        config["sro_prompt"]["acceptance"]["manual_complete_sessions_per_task"],
        config["execution"]["seed"],
    )
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in review),
        encoding="utf-8",
    )
    if len(review) != 50 or not all(all(item["review_checks"].values()) for item in review):
        raise RuntimeError(f"Fifty-session review gate failed for {task}")
    return {
        "task": task,
        "status": "built",
        "source_rows_development": len(raw),
        "rows": len(records),
        "participants": records["participant"].nunique(),
        "primary": int((records["validity"] == "primary").sum()),
        "history_only": int((records["validity"] == "history_only").sum()),
        "invalid": int((records["validity"] == "invalid").sum()),
        "reviewed_sessions": len(review),
        "structural_checks": structural,
        "parquet_sha256": _sha256(destination),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(*SPECS, "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _config()
    audit = pd.read_csv(AUDIT_PATH)
    if set(audit["task"]) != set(SPECS) or not (audit["lossless_encoding"] >= 1).all():
        raise RuntimeError("Final Q_k audit is not frozen")
    development, confirmation = freeze_splits(config)
    tasks = SPECS if args.task == "all" else (args.task,)
    results = [build_task(task, config, development, overwrite=args.overwrite) for task in tasks]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"tasks": {}}
    previous.update(
        {
            "schema_version": 1,
            "profile": "original",
            "config_hash": _config_hash(),
            "development_participants": len(development),
            "confirmation_participants": len(confirmation),
        }
    )
    for result in results:
        previous["tasks"][result["task"]] = result
    MANIFEST.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tasks": results, "manifest": str(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
