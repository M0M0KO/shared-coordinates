from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "configs" / "original" / "task_audit_draft.csv"
REPORT = ROOT / "reports" / "original" / "task_audit_inventory.md"
FINAL = ROOT / "configs" / "original" / "task_audit.csv"


def rows() -> list[dict[str, str]]:
    with DRAFT.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_original_audit_draft_contains_all_candidate_tasks() -> None:
    content = rows()
    assert len(content) == 16
    assert len({row["task"] for row in content}) == 16
    assert len({row["task_family"] for row in content}) == 6


def test_original_audit_draft_does_not_pretend_qk_is_decided() -> None:
    content = rows()
    assert {row["lossless_encoding"] for row in content} == {"PENDING_QK"}
    assert {row["audit_status"] for row in content} == {"header_inventory_only"}
    assert {row["response_column"] for row in content} == {"PENDING_MANUAL_DECISION"}


def test_original_audit_artifacts_do_not_contain_participant_values() -> None:
    text = DRAFT.read_text(encoding="utf-8") + REPORT.read_text(encoding="utf-8")
    assert "participant ids or response values" in text.lower()
    assert {row["participant_column"] for row in rows()} == {"worker_id"}


def test_original_audit_sources_are_public_complete_files() -> None:
    if not (ROOT / "data/raw/Self_Regulation_Ontology").exists():
        pytest.skip("The public SRO checkout is needed for this integration check")
    for row in rows():
        source = ROOT / row["source_file"]
        assert source.is_file()
        assert "Complete_02-16-2019/Individual_Measures" in row["source_file"]
        assert len(row["complete_sha256"]) == 64


def test_final_original_audit_is_frozen_only_after_parser_acceptance() -> None:
    if not FINAL.exists():
        return
    with FINAL.open(encoding="utf-8", newline="") as handle:
        content = list(csv.DictReader(handle))
    assert len(content) == 16
    assert {row["lossless_encoding"] for row in content} == {"1"}
    assert {row["audit_status"] for row in content} == {"Q1_parser_accepted_frozen"}
