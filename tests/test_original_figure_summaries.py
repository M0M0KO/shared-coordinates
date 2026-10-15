from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/original/12_make_results.py"


def _module():
    spec = importlib.util.spec_from_file_location("original_figure_summaries", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dimension_curve_uses_best_hyperparameters_per_dimension() -> None:
    module = _module()
    frame = pd.DataFrame(
        [
            {
                "dimension": 8,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "fold": 0,
                "validation_nll": 0.4,
            },
            {
                "dimension": 8,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "fold": 1,
                "validation_nll": 0.6,
            },
            {
                "dimension": 8,
                "learning_rate": 3e-3,
                "weight_decay": 0.0,
                "fold": 0,
                "validation_nll": 1.4,
            },
            {
                "dimension": 8,
                "learning_rate": 3e-3,
                "weight_decay": 0.0,
                "fold": 1,
                "validation_nll": 1.6,
            },
            {
                "dimension": 16,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "fold": 0,
                "validation_nll": 0.3,
            },
            {
                "dimension": 16,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "fold": 1,
                "validation_nll": 0.5,
            },
        ]
    )
    summary = module._best_dimension_summary(frame).set_index("dimension")
    assert np.isclose(summary.loc[8, "mean"], 0.5)
    assert np.isclose(summary.loc[16, "mean"], 0.4)


def test_control_summary_uses_folds_and_haar_replicates_as_units() -> None:
    module = _module()
    frame = pd.DataFrame(
        [
            {
                "control": "full_representation",
                "fold": 0,
                "replicate": -1,
                "task": "a",
                "validation_nll": 0.2,
            },
            {
                "control": "full_representation",
                "fold": 0,
                "replicate": -1,
                "task": "b",
                "validation_nll": 0.4,
            },
            {
                "control": "full_representation",
                "fold": 1,
                "replicate": -1,
                "task": "a",
                "validation_nll": 0.4,
            },
            {
                "control": "full_representation",
                "fold": 1,
                "replicate": -1,
                "task": "b",
                "validation_nll": 0.6,
            },
            {
                "control": "haar_random",
                "fold": 0,
                "replicate": 0,
                "task": "a",
                "validation_nll": 1.0,
            },
            {
                "control": "haar_random",
                "fold": 1,
                "replicate": 0,
                "task": "b",
                "validation_nll": 1.0,
            },
            {
                "control": "haar_random",
                "fold": 0,
                "replicate": 1,
                "task": "a",
                "validation_nll": 2.0,
            },
            {
                "control": "haar_random",
                "fold": 1,
                "replicate": 1,
                "task": "b",
                "validation_nll": 2.0,
            },
            {
                "control": "held_out_family_transfer",
                "fold": 0,
                "replicate": -1,
                "task": "a",
                "validation_nll": 9.0,
            },
        ]
    )
    summary = module._control_summary(frame)
    assert set(summary.index) == {"full_representation", "haar_random"}
    assert np.isclose(summary.loc["full_representation", "mean"], 0.4)
    assert np.isclose(summary.loc["haar_random", "mean"], 1.5)


def test_two_way_group_summary_preserves_participant_task_units() -> None:
    module = _module()
    frame = pd.DataFrame(
        [
            {
                "method": method,
                "participant": participant,
                "task": task,
                "delta_nll": value,
            }
            for method, value in (("a", 1.0), ("b", 2.0))
            for participant in ("p1", "p2")
            for task in ("t1", "t2")
        ]
    )
    summary = module._two_way_group_summary(
        frame,
        group="method",
        value="delta_nll",
        replicates=20,
        seed=7,
    ).set_index("method")
    assert np.isclose(summary.loc["a", "mean"], 1.0)
    assert np.isclose(summary.loc["a", "ci_95_low"], 1.0)
    assert np.isclose(summary.loc["b", "mean"], 2.0)
    assert summary.loc["b", "participant_clusters"] == 2
    assert summary.loc["b", "task_clusters"] == 2


def test_final_freeze_rejects_dirty_source_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_git", lambda *arguments: " M reviewed_source.py")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        module.freeze({})
