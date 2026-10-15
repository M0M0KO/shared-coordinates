from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from cognitive_manifold.geometry.shared import (
    SharedSubspace,
    TaskArrays,
    _task_balanced_loss,
    evaluate_task_balanced_nll,
    fit_shared_fold,
    stage_task_arrays,
)
from cognitive_manifold.individuality.model import (
    IndividualReadout,
    IndividualTaskData,
    infer_new_coordinates,
)


def _data() -> dict[str, TaskArrays]:
    rng = np.random.default_rng(17)
    output: dict[str, TaskArrays] = {}
    for task in ("alpha", "beta"):
        features = rng.normal(size=(30, 6)).astype(np.float32)
        labels = (features[:, 0] + 0.25 * features[:, 1] > 0).astype(np.int64)
        output[task] = TaskArrays(
            features=features,
            labels=labels,
            folds=np.arange(30, dtype=np.int8) % 3,
            classes=("0", "1"),
        )
    return output


def test_training_and_validation_reuse_one_qr_basis() -> None:
    data = _data()
    model = SharedSubspace(6, 3, {task: 2 for task in data})
    indices = {task: np.arange(30) for task in data}
    with patch.object(model, "basis", wraps=model.basis) as basis:
        loss = _task_balanced_loss(
            model,
            data,
            indices,
            batch_per_task=8,
            rng=np.random.default_rng(3),
            device=torch.device("cpu"),
        )
        loss.backward()
        assert basis.call_count == 1
    with patch.object(model, "basis", wraps=model.basis) as basis:
        evaluate_task_balanced_nll(
            model,
            data,
            indices,
            torch.device("cpu"),
            chunk_size=7,
        )
        assert basis.call_count == 1


def test_staged_arrays_match_streamed_training(monkeypatch) -> None:
    streamed = _data()
    monkeypatch.setenv("CM_STAGE_TASK_ARRAYS", "0")
    streamed_model, streamed_evidence = fit_shared_fold(
        streamed,
        input_dim=6,
        shared_dim=3,
        validation_fold=0,
        learning_rate=0.01,
        weight_decay=0.001,
        max_epochs=8,
        patience=8,
        minimum_improvement=0.0,
        batch_per_task=8,
        steps_per_epoch=3,
        seed=23,
        device="cpu",
    )

    staged = _data()
    monkeypatch.setenv("CM_STAGE_TASK_ARRAYS", "1")
    staged_model, staged_evidence = fit_shared_fold(
        staged,
        input_dim=6,
        shared_dim=3,
        validation_fold=0,
        learning_rate=0.01,
        weight_decay=0.001,
        max_epochs=8,
        patience=8,
        minimum_improvement=0.0,
        batch_per_task=8,
        steps_per_epoch=3,
        seed=23,
        device="cpu",
    )

    assert staged_evidence == streamed_evidence
    for name, expected in streamed_model.state_dict().items():
        torch.testing.assert_close(staged_model.state_dict()[name], expected, rtol=0, atol=0)
    assert all(isinstance(values.features, torch.Tensor) for values in staged.values())
    assert all(isinstance(values.labels, torch.Tensor) for values in staged.values())


def test_explicit_staging_preserves_values() -> None:
    data = _data()
    expected = {task: values.features.copy() for task, values in data.items()}
    stage_task_arrays(data, "cpu")
    for task, values in data.items():
        torch.testing.assert_close(values.features, torch.from_numpy(expected[task]))


def test_cached_coordinate_inference_matches_reference(monkeypatch) -> None:
    rng = np.random.default_rng(31)
    data: dict[str, IndividualTaskData] = {}
    for task in ("alpha", "beta"):
        data[task] = IndividualTaskData(
            states=rng.normal(size=(24, 5)).astype(np.float32),
            labels=rng.integers(0, 2, size=24, dtype=np.int64),
            participants=np.arange(24, dtype=np.int32) % 4,
            folds=np.arange(24, dtype=np.int8) % 3,
            family="source",
            classes=("0", "1"),
        )
    weights = {task: torch.randn(2, 5) for task in data}
    biases = {task: torch.randn(2) for task in data}
    torch.manual_seed(41)
    model = IndividualReadout(5, 2, 4, weights, biases)
    participants = [0, 1, 2, 3]

    coordinates = nn.Parameter(torch.zeros(4, 2))
    optimizer = torch.optim.Adam([coordinates], lr=0.03)
    remap = {participant: index for index, participant in enumerate(participants)}
    for _ in range(5):
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for task, values in data.items():
            selected = np.flatnonzero(np.isin(values.participants, participants))
            states = torch.as_tensor(values.states[selected], dtype=torch.float32)
            labels = torch.as_tensor(values.labels[selected], dtype=torch.long)
            people = torch.as_tensor(
                [remap[int(value)] for value in values.participants[selected]], dtype=torch.long
            )
            losses.append(
                nn.functional.cross_entropy(
                    model.fixed_logits(task, states, coordinates[people]), labels
                )
            )
        loss = torch.stack(losses).mean() + 0.01 * coordinates.square().mean()
        loss.backward()
        optimizer.step()
    expected = coordinates.detach()

    for parameter in model.parameters():
        parameter.grad = None
    monkeypatch.setenv("CM_STAGE_TASK_ARRAYS", "1")
    actual = infer_new_coordinates(
        model,
        data,
        participant_ids=participants,
        source_families={"source"},
        regularization=0.01,
        steps=5,
        learning_rate=0.03,
        device="cpu",
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
