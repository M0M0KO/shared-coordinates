#!/usr/bin/env python3
"""Freeze Q_k decisions from recorded fields before any model-performance output."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "original" / "main.yaml"
OUTPUT = ROOT / "configs" / "original" / "task_audit.csv"
REPORT = ROOT / "reports" / "original" / "task_audit.md"

CONTRACTS: dict[str, dict[str, list[str] | str]] = {
    "adaptive_n_back": {
        "response": "key_press",
        "stimulus": ["stim", "load", "exp_stage"],
        "feedback": ["correct"],
        "order": ["time_elapsed", "trial_id"],
    },
    "attention_network_task": {
        "response": "key_press",
        "stimulus": ["cue", "flanker_location", "flanker_middle_direction", "flanker_type"],
        "feedback": ["correct"],
        "order": ["time_elapsed", "trial_num", "trial_id"],
    },
    "columbia_card_task_hot": {
        "response": "mouse_click",
        "stimulus": [
            "which_round",
            "num_click_in_round",
            "gain_amount",
            "loss_amount",
            "num_loss_cards",
            "total_cards",
        ],
        "feedback": ["clicked_on_loss_card", "round_points"],
        "order": ["time_elapsed", "which_round", "num_click_in_round", "trial_id"],
    },
    "discount_titrate": {
        "response": "key_press",
        "stimulus": [
            "stimulus",
            "smaller_amount",
            "sooner_delay",
            "larger_amount",
            "later_delay",
        ],
        "feedback": [],
        "order": ["time_elapsed", "trial_id"],
    },
    "dot_pattern_expectancy": {
        "response": "key_press",
        "stimulus": ["condition", "stimulus"],
        "feedback": ["correct"],
        "order": ["time_elapsed", "trial_num", "trial_id"],
    },
    "go_nogo": {
        "response": "key_press",
        "stimulus": ["condition", "stimulus"],
        "feedback": ["correct"],
        "order": ["time_elapsed", "trial_num", "trial_id"],
    },
    "information_sampling_task": {
        "response": "mouse_click",
        "stimulus": ["exp_stage", "which_click_in_round", "color_clicked"],
        "feedback": ["points", "reward"],
        "order": ["time_elapsed", "trial_num", "which_click_in_round", "trial_id"],
    },
    "probabilistic_selection": {
        "response": "key_press",
        "stimulus": ["stimulus", "exp_stage"],
        "feedback": ["feedback"],
        "order": ["time_elapsed", "trial_num", "trial_id"],
    },
    "shift_task": {
        "response": "key_press",
        "stimulus": ["stims"],
        "feedback": ["feedback", "total_points"],
        "order": ["time_elapsed", "trial_num", "trial_id"],
    },
    "simon": {
        "response": "key_press",
        "stimulus": ["stim_color", "stim_side"],
        "feedback": ["correct"],
        "order": ["time_elapsed", "trial_num", "trial_id"],
    },
    "stim_selective_stop_signal": {
        "response": "key_press",
        "stimulus": ["stimulus", "condition", "SS_trial_type", "SS_delay", "SS_stimulus"],
        "feedback": ["correct", "stopped"],
        "order": ["time_elapsed", "trial_num", "trial_id"],
    },
    "stop_signal": {
        "response": "key_press",
        "stimulus": ["stimulus", "SS_trial_type", "SS_delay", "SS_stimulus"],
        "feedback": ["correct", "stopped"],
        "order": ["time_elapsed", "trial_num", "trial_id"],
    },
    "stroop": {
        "response": "key_press",
        "stimulus": ["stimulus"],
        "feedback": ["correct"],
        "order": ["time_elapsed", "trial_id"],
    },
    "threebytwo": {
        "response": "key_press",
        "stimulus": ["cue", "task", "stim_color", "stim_number"],
        "feedback": ["correct"],
        "order": ["time_elapsed", "trial_num", "trial_id"],
    },
    "tower_of_london": {
        "response": "mouse_click",
        "stimulus": ["problem_id", "current_position", "target", "min_moves"],
        "feedback": ["correct"],
        "order": ["time_elapsed", "trial_id"],
    },
    "two_stage_decision": {
        "response": "key_press_first|key_press_second",
        "stimulus": ["stim_order_first", "stage_second", "stim_order_second"],
        "feedback": ["stage_transition", "feedback"],
        "order": ["time_elapsed", "trial_num", "trial_id"],
    },
}


def _encode(values: list[str]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command, cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def main() -> None:
    config: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())
    sro = ROOT / "data/raw/Self_Regulation_Ontology"
    complete_root = sro / config["data"]["sro"]["complete_path"]
    retest_root = sro / config["data"]["sro"]["retest_path"]
    source_repo = ROOT / config["task_audit"]["source_experiment_repository"]["local_dir"]
    source_head = _git(["git", "rev-parse", "HEAD"], source_repo)
    source_window_commits = _git(
        [
            "git",
            "log",
            "--format=%H",
            "--since=2016-07-21",
            "--until=2016-09-23",
            "--",
            *SPECS,
        ],
        source_repo,
    ).splitlines()
    rows: list[dict[str, Any]] = []
    for task, spec in SPECS.items():
        contract = CONTRACTS[task]
        complete = complete_root / f"{task}.csv.gz"
        retest = retest_root / f"{task}.csv.gz"
        complete_columns = list(pd.read_csv(complete, nrows=0).columns)
        retest_columns = list(pd.read_csv(retest, nrows=0).columns)
        required = [
            "worker_id",
            "finishtime",
            "exp_stage",
            *contract["stimulus"],
            *contract["feedback"],
            *contract["order"],
            *str(contract["response"]).split("|"),
        ]
        required = list(dict.fromkeys(required))
        missing = sorted(set(required) - set(complete_columns))
        if missing:
            raise RuntimeError(f"{task} is missing required columns: {missing}")
        rows.append(
            {
                "task": task,
                "source_file": str(complete.relative_to(ROOT)),
                "required_columns": _encode(required),
                "response_column": contract["response"],
                "stimulus_columns": _encode(contract["stimulus"]),
                "feedback_columns": _encode(contract["feedback"]),
                "trial_order_columns": _encode(contract["order"]),
                "lossless_encoding": 1,
                "exclusion_reason": "",
                "task_family": spec.family,
                "audit_status": "Q1_frozen_parser_acceptance_pending",
                "complete_sha256": _sha256(complete),
                "complete_retest_header_match": complete_columns == retest_columns,
                "source_code_head": source_head,
                "study_window_source_commits": _encode(source_window_commits),
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT.with_suffix(".csv.tmp").replace(OUTPUT)
    REPORT.write_text(
        "\n".join(
            [
                "# Original SRO reconstruction audit",
                "",
                "Status: **all 16 tasks frozen at Q=1 before model-performance inspection; "
                "parser acceptance pending**",
                "",
                "- Recorded fields, rather than a single retrospective task-code snapshot, "
                "are authoritative.",
                "- The collection window crossed task-code changes; the source commit list is "
                "stored per row.",
                "- All tasks require structured reconstruction, so none is labelled Q=2.",
                "- No task was selected or excluded using model performance.",
                "- Retest access in this step was restricted to CSV headers.",
                "",
                f"Source-code HEAD used for instruction audit: `{source_head}`",
                f"Study-window candidate source commits: {len(source_window_commits)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"tasks": len(rows), "q1": len(rows), "source_head": source_head}))


if __name__ == "__main__":
    main()
