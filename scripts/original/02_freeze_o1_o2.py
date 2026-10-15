#!/usr/bin/env python3
"""Freeze O1/O2 only after every parser and provenance gate passes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/original/main.yaml"
AUDIT = ROOT / "configs/original/task_audit.csv"
SESSIONS = ROOT / "data/derived/original/sro_sessions/development"
REVIEWS = ROOT / "manifests/private/original/parser_review"
SPLITS = ROOT / "configs/original/splits.json"
OUTPUT = ROOT / "manifests/public/original_o1_o2_freeze.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    if not config["execution"]["runnable"] or config["execution"]["blocked_by_decisions"]:
        raise RuntimeError("D01-D05 are not frozen")
    config_hash = _sha256(CONFIG)
    audit = pd.read_csv(AUDIT)
    if set(audit["task"]) != set(SPECS) or set(audit["lossless_encoding"]) != {1}:
        raise RuntimeError("Q_k audit is incomplete")
    splits = json.loads(SPLITS.read_text())
    if splits["counts"] != {"complete": 522, "confirmation": 150, "development": 372}:
        raise RuntimeError("Participant split counts differ from the frozen contract")
    task_evidence: dict[str, dict[str, object]] = {}
    for task in SPECS:
        session_path = SESSIONS / f"{task}.parquet"
        review_path = REVIEWS / f"{task}.jsonl"
        sessions = pd.read_parquet(
            session_path,
            columns=["participant", "validity", "config_hash"],
        )
        hashes = set(sessions["config_hash"])
        if hashes != {config_hash}:
            raise RuntimeError(f"{task} does not use the final configuration hash: {hashes}")
        reviews = [json.loads(line) for line in review_path.read_text().splitlines() if line]
        if len(reviews) != 50 or not all(all(item["review_checks"].values()) for item in reviews):
            raise RuntimeError(f"Fifty-session review failed for {task}")
        task_evidence[task] = {
            "rows": len(sessions),
            "primary_rows": int((sessions["validity"] == "primary").sum()),
            "participants": int(sessions["participant"].nunique()),
            "session_sha256": _sha256(session_path),
            "review_sha256": _sha256(review_path),
            "reviewed_sessions": 50,
        }
    audit["audit_status"] = "Q1_parser_accepted_frozen"
    temporary_audit = AUDIT.with_suffix(".csv.tmp")
    audit.to_csv(temporary_audit, index=False)
    temporary_audit.replace(AUDIT)
    payload = {
        "schema_version": 1,
        "profile": "original",
        "config_sha256": config_hash,
        "task_audit_sha256": _sha256(AUDIT),
        "split_manifest_sha256": _sha256(SPLITS),
        "counts": splits["counts"],
        "tasks": task_evidence,
        "model_performance_inspected_before_freeze": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "frozen", "tasks": len(task_evidence), "config": config_hash}))


if __name__ == "__main__":
    main()
