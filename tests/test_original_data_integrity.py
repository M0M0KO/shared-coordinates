from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cognitive_manifold.data.splits import participant_pseudonym

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "configs/original/splits.json"
DEVELOPMENT = ROOT / "data/derived/original/sro_sessions/development"
PRIVATE_RETEST = ROOT / "data/derived/original/sro_retest_ids.txt"
RECOVERY_ALIGNMENT = ROOT / "cache/original/original_coordinate_recovery/alignment.npz"


def test_frozen_development_and_retest_confirmation_splits_are_disjoint() -> None:
    if not SPLITS.exists() or not PRIVATE_RETEST.exists():
        pytest.skip("Private frozen SRO split artifacts are not available in this checkout")
    payload = json.loads(SPLITS.read_text())
    development = {item["participant"] for item in payload["development"]}
    confirmation = set(payload["confirmation"])
    raw_retest_ids = {
        item.strip() for item in PRIVATE_RETEST.read_text().splitlines() if item.strip()
    }
    assert len(development) == 372
    assert len(confirmation) == 150
    assert development.isdisjoint(confirmation)
    assert {participant_pseudonym(item) for item in raw_retest_ids} == confirmation


def test_materialized_development_artifacts_exclude_confirmation_participants() -> None:
    if not SPLITS.exists() or not DEVELOPMENT.exists():
        pytest.skip("Private materialized SRO artifacts are not available in this checkout")
    confirmation = set(json.loads(SPLITS.read_text())["confirmation"])
    tables = sorted(DEVELOPMENT.glob("*.parquet"))
    if not tables:
        pytest.skip("Development Parquet artifacts have not been materialized")
    for table in tables:
        observed = set(pd.read_parquet(table, columns=["participant"])["participant"])
        assert observed.isdisjoint(confirmation), table.name


def test_procrustes_alignment_fit_never_uses_confirmation_participants() -> None:
    if not SPLITS.exists() or not RECOVERY_ALIGNMENT.exists():
        pytest.skip("Development-only Procrustes artifacts have not been materialized")
    payload = json.loads(SPLITS.read_text())
    development = {item["participant"] for item in payload["development"]}
    confirmation = set(payload["confirmation"])
    with np.load(RECOVERY_ALIGNMENT) as alignment:
        observed = set(alignment["participant"].astype(str))
    assert observed == development
    assert observed.isdisjoint(confirmation)
