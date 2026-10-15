from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _module() -> object:
    path = Path(__file__).parents[1] / "scripts/original/07_run_psych201.py"
    spec = importlib.util.spec_from_file_location("original_psych201_context_qc_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["prompt <<A>>", "long prompt <<B>> <<C>>", "prompt <<D>>"],
            "study": ["s", "s", "s"],
            "experiment": ["e", "e", "e"],
            "participant": ["p0", "p1", "p2"],
        }
    )


def test_context_qc_excludes_pinned_row_and_preserves_source_rows() -> None:
    module = _module()
    frame = _frame()
    text = frame.loc[1, "text"]
    qc = {
        "maximum_model_tokens": 10,
        "exclusions": [
            {
                "source_shard": 3,
                "source_session_row": 1,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "observed_tokens": 11,
                "response_count": 2,
                "reason": "context_limit",
            }
        ],
    }
    retained, evidence = module._apply_context_qc(frame, 3, qc)
    assert retained["frozen_source_session_row"].tolist() == [0, 2]
    assert evidence[0]["source_session_row"] == 1
    assert evidence[0]["response_count"] == 2


def test_context_qc_rejects_changed_source_text() -> None:
    module = _module()
    qc = {
        "maximum_model_tokens": 10,
        "exclusions": [
            {
                "source_shard": 3,
                "source_session_row": 1,
                "text_sha256": "0" * 64,
                "observed_tokens": 11,
                "response_count": 2,
                "reason": "context_limit",
            }
        ],
    }
    with pytest.raises(RuntimeError, match="source hash changed"):
        module._apply_context_qc(_frame(), 3, qc)


def test_experiment_basis_returns_finite_orthonormal_basis() -> None:
    module = _module()
    rng = np.random.default_rng(7)
    features = rng.normal(size=(64, 6)).astype(np.float32)

    basis, participation_ratio = module._experiment_basis(
        features,
        dimension=3,
        maximum=64,
        seed=11,
    )

    assert basis.shape == (6, 3)
    assert np.isfinite(participation_ratio)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-5)
