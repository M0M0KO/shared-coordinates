#!/usr/bin/env python3
"""Create a source-only SRO task-audit draft for the original profile.

This step inspects file presence and CSV headers. It deliberately does not assign Q_k,
read retest behavior rows, or inspect model performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "original" / "main.yaml"
DEFAULT_DRAFT = ROOT / "configs" / "original" / "task_audit_draft.csv"
DEFAULT_REPORT = ROOT / "reports" / "original" / "task_audit_inventory.md"

RESPONSE_NAMES = {
    "choice",
    "chosen_boxes_clicked",
    "color_clicked",
    "key_press",
    "key_press_first",
    "key_press_second",
    "mouse_click",
    "responses",
    "stim_chosen",
    "stim_selected_first",
    "stim_selected_second",
}
STIMULUS_TOKENS = (
    "amount",
    "condition",
    "cue",
    "delay",
    "location",
    "position",
    "possible_responses",
    "problem",
    "stage",
    "stim",
    "target",
    "task",
    "text",
)
FEEDBACK_TOKENS = ("correct", "feedback", "gain", "loss", "optimal", "points", "reward")
ORDER_TOKENS = ("block", "round", "stage", "time_elapsed", "trial", "which_click")
PARTICIPANT_NAMES = ("worker_id", "participant_id", "subject_id")

SOURCE_COLUMNS = [
    "task",
    "source_file",
    "required_columns",
    "response_column",
    "stimulus_columns",
    "feedback_columns",
    "trial_order_columns",
    "lossless_encoding",
    "exclusion_reason",
    "task_family",
]
EVIDENCE_COLUMNS = [
    "audit_status",
    "complete_bytes",
    "complete_sha256",
    "participant_column",
    "complete_columns",
    "retest_columns",
    "retest_header_matches_complete",
    "complete_only_columns",
    "retest_only_columns",
    "response_candidates",
    "stimulus_candidates",
    "feedback_candidates",
    "trial_order_candidates",
]


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["analysis_profile"] != "original":
        raise RuntimeError("Task audit must use the original profile")
    return config


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def select_exact(columns: list[str], names: set[str]) -> list[str]:
    return [column for column in columns if column.lower() in names]


def select_tokens(columns: list[str], tokens: tuple[str, ...]) -> list[str]:
    return [column for column in columns if any(token in column.lower() for token in tokens)]


def participant_column(columns: list[str]) -> str:
    lowered = {column.lower(): column for column in columns}
    matches = [lowered[name] for name in PARTICIPANT_NAMES if name in lowered]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one participant column, found {matches}")
    return matches[0]


def encode_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    sro_root = ROOT / "data" / "raw" / "Self_Regulation_Ontology"
    complete_root = sro_root / config["data"]["sro"]["complete_path"]
    retest_root = sro_root / config["data"]["sro"]["retest_path"]
    rows: list[dict[str, Any]] = []

    for family, tasks in config["task_audit"]["task_families"].items():
        for task in tasks:
            complete = complete_root / f"{task}.csv.gz"
            retest = retest_root / f"{task}.csv.gz"
            if not complete.is_file() or not retest.is_file():
                raise RuntimeError(f"Missing complete or retest source for {task}")
            complete_columns = header(complete)
            retest_columns = header(retest)
            participant = participant_column(complete_columns)

            response_candidates = select_exact(complete_columns, RESPONSE_NAMES)
            stimulus_candidates = select_tokens(complete_columns, STIMULUS_TOKENS)
            feedback_candidates = select_tokens(complete_columns, FEEDBACK_TOKENS)
            order_candidates = select_tokens(complete_columns, ORDER_TOKENS)
            rows.append(
                {
                    "task": task,
                    "source_file": str(complete.relative_to(ROOT)),
                    "required_columns": "PENDING_MANUAL_DECISION",
                    "response_column": "PENDING_MANUAL_DECISION",
                    "stimulus_columns": "PENDING_MANUAL_DECISION",
                    "feedback_columns": "PENDING_MANUAL_DECISION",
                    "trial_order_columns": "PENDING_MANUAL_DECISION",
                    "lossless_encoding": "PENDING_QK",
                    "exclusion_reason": "",
                    "task_family": family,
                    "audit_status": "header_inventory_only",
                    "complete_bytes": complete.stat().st_size,
                    "complete_sha256": sha256_file(complete),
                    "participant_column": participant,
                    "complete_columns": encode_list(complete_columns),
                    "retest_columns": encode_list(retest_columns),
                    "retest_header_matches_complete": complete_columns == retest_columns,
                    "complete_only_columns": encode_list(
                        sorted(set(complete_columns) - set(retest_columns))
                    ),
                    "retest_only_columns": encode_list(
                        sorted(set(retest_columns) - set(complete_columns))
                    ),
                    "response_candidates": encode_list(response_candidates),
                    "stimulus_candidates": encode_list(stimulus_candidates),
                    "feedback_candidates": encode_list(feedback_candidates),
                    "trial_order_candidates": encode_list(order_candidates),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SOURCE_COLUMNS + EVIDENCE_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def write_report(rows: list[dict[str, Any]], destination: Path) -> None:
    mismatch = [row for row in rows if not row["retest_header_matches_complete"]]
    no_response = [row["task"] for row in rows if row["response_candidates"] == "[]"]
    lines = [
        "# Original SRO task-audit inventory",
        "",
        "Status: **header inventory only; no Q_k decisions and no model output**",
        "",
        f"- Candidate tasks found in both releases: {len(rows)}",
        f"- Complete/retest header mismatches: {len(mismatch)}",
        f"- Tasks with no automatically recognized response candidate: {len(no_response)}",
        "- Retest access in this step: CSV header only (`nrows=0`); no retest behavior rows read.",
        "- Participant IDs or response values are not written to this report or the draft CSV.",
        "",
        "The candidate columns are mechanical hints, not scientific decisions. Each task still",
        "requires source-code/instruction review, Q_k assignment, parser tests, summary-statistic",
        "reproduction, and manual inspection of 50 complete sessions before inference.",
        "",
    ]
    if mismatch:
        lines.extend(["## Header differences", ""])
        for row in mismatch:
            lines.append(
                f"- `{row['task']}`: complete-only {row['complete_only_columns']}; "
                f"retest-only {row['retest_only_columns']}"
            )
        lines.append("")
    if no_response:
        lines.extend(["No-response-candidate tasks: " + ", ".join(no_response), ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(load_config())
    write_csv(rows, args.draft)
    write_report(rows, args.report)
    print(json.dumps({"tasks": len(rows), "draft": str(args.draft), "report": str(args.report)}))


if __name__ == "__main__":
    main()
