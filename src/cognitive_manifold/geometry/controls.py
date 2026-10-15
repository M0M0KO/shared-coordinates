"""Fixed-basis task readouts for original shared-manifold controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from cognitive_manifold.geometry.shared import (
    TaskArrays,
    _maybe_stage_task_arrays,
    _task_batch,
)


class FixedBasisReadout(nn.Module):
    def __init__(self, basis: np.ndarray, class_counts: dict[str, int]) -> None:
        super().__init__()
        self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32))
        self.weights = nn.ParameterDict(
            {
                task: nn.Parameter(torch.zeros(classes, basis.shape[1]))
                for task, classes in class_counts.items()
            }
        )
        self.biases = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(classes)) for task, classes in class_counts.items()}
        )

    def forward(self, task: str, features: torch.Tensor) -> torch.Tensor:
        latent = features @ self.basis
        return latent @ self.weights[task].T + self.biases[task]


@torch.inference_mode()
def evaluate_fixed_basis(
    model: FixedBasisReadout,
    data: dict[str, TaskArrays],
    indices: dict[str, np.ndarray],
    *,
    device: str = "cuda:0",
    chunk_size: int = 8192,
) -> dict[str, float]:
    model.eval()
    output: dict[str, float] = {}
    target_device = torch.device(device)
    for task, values in data.items():
        selected = indices[task]
        total = torch.zeros((), dtype=torch.float64, device=target_device)
        for start in range(0, len(selected), chunk_size):
            batch = selected[start : start + chunk_size]
            features, labels = _task_batch(values, batch, target_device)
            total += nn.functional.cross_entropy(model(task, features), labels, reduction="sum")
        output[task] = float((total / len(selected)).item())
    return output


def fit_fixed_basis_fold(
    data: dict[str, TaskArrays],
    basis: np.ndarray,
    *,
    validation_fold: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    minimum_improvement: float,
    batch_per_task: int,
    steps_per_epoch: int,
    seed: int,
    device: str = "cuda:0",
) -> tuple[FixedBasisReadout, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    target_device = torch.device(device)
    _maybe_stage_task_arrays(data, target_device)
    model = FixedBasisReadout(
        basis,
        {task: len(values.classes) for task, values in data.items()},
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train = {task: np.flatnonzero(values.folds != validation_fold) for task, values in data.items()}
    validation = {
        task: np.flatnonzero(values.folds == validation_fold) for task, values in data.items()
    }
    best_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            for task, values in data.items():
                pool = train[task]
                selected = rng.choice(
                    pool,
                    size=batch_per_task,
                    replace=len(pool) < batch_per_task,
                )
                features, labels = _task_batch(values, selected, target_device)
                losses.append(nn.functional.cross_entropy(model(task, features), labels))
            torch.stack(losses).mean().backward()
            optimizer.step()
        task_nll = evaluate_fixed_basis(model, data, validation, device=str(target_device))
        validation_nll = float(np.mean(list(task_nll.values())))
        if validation_nll < best_nll - minimum_improvement:
            best_nll = validation_nll
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Fixed-basis control never produced a validation checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_nll": best_nll}
