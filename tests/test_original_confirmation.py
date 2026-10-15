from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from cognitive_manifold.individuality.model import IndividualReadout, IndividualTaskData

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/original/06_confirm_retest.py"


def _module():
    spec = importlib.util.spec_from_file_location("confirm_original_retest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task(family: str, participants: list[int]) -> IndividualTaskData:
    rows = len(participants)
    return IndividualTaskData(
        states=np.zeros((rows, 2), dtype=np.float32),
        labels=np.zeros(rows, dtype=np.int64),
        participants=np.asarray(participants, dtype=np.int32),
        folds=np.full(rows, -1, dtype=np.int8),
        family=family,
        classes=("A", "B"),
    )


def test_confirmation_gate_requires_two_source_and_one_target_family() -> None:
    module = _module()
    t1 = {
        "source_a": _task("source_a", [0, 1]),
        "source_b": _task("source_b", [0]),
    }
    t2 = {"target": _task("target", [0, 1])}
    eligible = module._eligible_confirmation_participants(
        t1,
        t2,
        {"source_a", "source_b"},
        {"target"},
        2,
        1,
    )
    assert eligible == [0]


def test_confirmation_factor_features_use_pickle_free_unicode_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    scores = tmp_path / "factor_scores.npz"
    np.savez(
        scores,
        confirmation_participant=np.asarray(["p_beta", "p_alpha"], dtype=str),
        confirmation_t1=np.asarray([[2.0, 3.0], [0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(module, "FACTOR_SCORES", scores)

    observed = module._confirmation_factor_features(["p_alpha", "p_beta"], "T1")

    np.testing.assert_array_equal(
        observed,
        np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration test")
def test_response_bias_accepts_staged_cuda_labels() -> None:
    module = _module()
    model = IndividualReadout(
        shared_dim=2,
        rank=1,
        participants=1,
        population_weights={"target": torch.zeros((2, 2))},
        population_biases={"target": torch.zeros(2)},
    ).to("cuda:0")
    source = _task("target_family", [0, 0, 0])
    source.labels = torch.as_tensor([0, 0, 1], dtype=torch.long, device="cuda:0")
    target = _task("target_family", [0, 0])
    target.states = torch.zeros((2, 2), dtype=torch.float32, device="cuda:0")
    target.labels = torch.as_tensor([0, 1], dtype=torch.long, device="cuda:0")

    observed = module._evaluate_response_bias(
        model,
        {"target": source},
        {"target": target},
        [0],
        {"target_family"},
    )

    assert len(observed) == 1
    assert observed[0]["method"] == "response_bias"
