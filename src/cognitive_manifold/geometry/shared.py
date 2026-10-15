"""Supervised shared orthonormal subspace with task-specific categorical readouts."""

from __future__ import annotations

import copy
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class TaskArrays:
    features: np.ndarray | torch.Tensor
    labels: np.ndarray | torch.Tensor
    folds: np.ndarray
    classes: tuple[str, ...]


class SharedSubspace(nn.Module):
    def __init__(self, input_dim: int, shared_dim: int, class_counts: dict[str, int]) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        initial = torch.randn(input_dim, shared_dim, generator=generator) / input_dim**0.5
        self.raw_basis = nn.Parameter(initial)
        self.weights = nn.ParameterDict(
            {
                task: nn.Parameter(torch.zeros(classes, shared_dim))
                for task, classes in class_counts.items()
            }
        )
        self.biases = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(classes)) for task, classes in class_counts.items()}
        )

    def basis(self) -> torch.Tensor:
        return torch.linalg.qr(self.raw_basis, mode="reduced").Q

    def forward(
        self,
        task: str,
        features: torch.Tensor,
        *,
        basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = features @ (self.basis() if basis is None else basis)
        return latent @ self.weights[task].T + self.biases[task]


def stage_task_arrays(
    data: dict[str, TaskArrays],
    device: torch.device | str,
) -> dict[str, TaskArrays]:
    """Move immutable features and labels once instead of once per epoch.

    The original arrays remain scientifically unchanged.  This function only
    changes their physical placement and floating-point storage to the same
    float32/long dtypes previously created for every batch.
    """

    target = torch.device(device)
    if target.type == "cuda":
        required = sum(
            (
                values.features.numel()
                if isinstance(values.features, torch.Tensor)
                else values.features.size
            )
            * torch.tensor([], dtype=torch.float32).element_size()
            + (
                values.labels.numel()
                if isinstance(values.labels, torch.Tensor)
                else values.labels.size
            )
            * torch.tensor([], dtype=torch.long).element_size()
            for values in data.values()
            if not (
                isinstance(values.features, torch.Tensor)
                and values.features.device == target
                and isinstance(values.labels, torch.Tensor)
                and values.labels.device == target
            )
        )
        free, _total = torch.cuda.mem_get_info(target)
        reserve_gib = float(os.environ.get("CM_GPU_RESERVE_GIB", "8"))
        reserve = int(reserve_gib * 2**30)
        if required + reserve > free:
            raise RuntimeError(
                "Insufficient free GPU memory to stage task arrays: "
                f"required={required / 2**30:.2f} GiB, free={free / 2**30:.2f} GiB, "
                f"reserve={reserve_gib:.2f} GiB. Set CM_STAGE_TASK_ARRAYS=0 to use "
                "the original streamed placement."
            )
    for values in data.values():
        if not isinstance(values.features, torch.Tensor):
            values.features = torch.tensor(
                np.asarray(values.features), dtype=torch.float32, device=target
            )
        elif values.features.device != target or values.features.dtype != torch.float32:
            values.features = values.features.to(device=target, dtype=torch.float32)
        if not isinstance(values.labels, torch.Tensor):
            values.labels = torch.as_tensor(values.labels, dtype=torch.long, device=target)
        elif values.labels.device != target or values.labels.dtype != torch.long:
            values.labels = values.labels.to(device=target, dtype=torch.long)
    return data


def _maybe_stage_task_arrays(
    data: dict[str, TaskArrays], device: torch.device
) -> dict[str, TaskArrays]:
    mode = os.environ.get("CM_STAGE_TASK_ARRAYS", "auto").lower()
    if mode in {"0", "false", "off", "stream"}:
        return data
    try:
        return stage_task_arrays(data, device)
    except RuntimeError as error:
        if mode in {"1", "true", "on", "force"}:
            raise
        warnings.warn(f"Falling back to streamed task arrays: {error}", stacklevel=2)
        return data


def _task_batch(
    values: TaskArrays,
    selected: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(values.features, torch.Tensor):
        index = torch.as_tensor(selected, dtype=torch.long, device=values.features.device)
        features = torch.index_select(values.features, 0, index)
    else:
        features = torch.as_tensor(
            np.asarray(values.features[selected]), dtype=torch.float32, device=device
        )
    if isinstance(values.labels, torch.Tensor):
        if "index" not in locals() or index.device != values.labels.device:
            index = torch.as_tensor(selected, dtype=torch.long, device=values.labels.device)
        labels = torch.index_select(values.labels, 0, index)
    else:
        labels = torch.as_tensor(values.labels[selected], dtype=torch.long, device=device)
    return features, labels


def task_balanced_cross_entropy(
    logits_by_task: dict[str, torch.Tensor], labels_by_task: dict[str, torch.Tensor]
) -> torch.Tensor:
    if set(logits_by_task) != set(labels_by_task) or not logits_by_task:
        raise ValueError("Task logits and labels must have the same non-empty keys")
    losses = [
        nn.functional.cross_entropy(logits_by_task[task], labels_by_task[task])
        for task in sorted(logits_by_task)
    ]
    return torch.stack(losses).mean()


def load_task_arrays(
    projected_root: Path,
    state_root: Path,
    tasks: list[str],
    input_dim: int,
) -> dict[str, TaskArrays]:
    import pandas as pd

    output: dict[str, TaskArrays] = {}
    for task in tasks:
        features = np.load(projected_root / f"{task}.npy", mmap_mode="r")[:, :input_dim]
        metadata = pd.read_parquet(
            state_root / task / "metadata.parquet",
            columns=["response", "fold"],
        )
        if len(features) != len(metadata):
            raise RuntimeError(f"Feature/metadata mismatch for {task}")
        classes = tuple(sorted(metadata["response"].unique()))
        class_index = {value: index for index, value in enumerate(classes)}
        labels = metadata["response"].map(class_index).to_numpy(np.int64)
        output[task] = TaskArrays(
            features=features,
            labels=labels,
            folds=metadata["fold"].to_numpy(np.int8),
            classes=classes,
        )
    return output


def _task_balanced_loss(
    model: SharedSubspace,
    data: dict[str, TaskArrays],
    indices: dict[str, np.ndarray],
    batch_per_task: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    basis = model.basis()
    for task, values in data.items():
        pool = indices[task]
        selected = rng.choice(pool, size=batch_per_task, replace=len(pool) < batch_per_task)
        features, labels = _task_batch(values, selected, device)
        losses.append(nn.functional.cross_entropy(model(task, features, basis=basis), labels))
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate_task_balanced_nll(
    model: SharedSubspace,
    data: dict[str, TaskArrays],
    indices: dict[str, np.ndarray],
    device: torch.device,
    chunk_size: int = 8192,
) -> tuple[float, dict[str, float]]:
    model.eval()
    task_nll: dict[str, float] = {}
    basis = model.basis()
    for task, values in data.items():
        selected = indices[task]
        total = torch.zeros((), dtype=torch.float64, device=device)
        count = 0
        for start in range(0, len(selected), chunk_size):
            batch_indices = selected[start : start + chunk_size]
            features, labels = _task_batch(values, batch_indices, device)
            total += nn.functional.cross_entropy(
                model(task, features, basis=basis), labels, reduction="sum"
            )
            count += len(batch_indices)
        task_nll[task] = float((total / count).item())
    return float(np.mean(list(task_nll.values()))), task_nll


def fit_shared_fold(
    data: dict[str, TaskArrays],
    *,
    input_dim: int,
    shared_dim: int,
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
) -> tuple[SharedSubspace, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    target_device = torch.device(device)
    _maybe_stage_task_arrays(data, target_device)
    class_counts = {task: len(values.classes) for task, values in data.items()}
    model = SharedSubspace(input_dim, shared_dim, class_counts).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=weight_decay)
    train = {task: np.flatnonzero(values.folds != validation_fold) for task, values in data.items()}
    validation = {
        task: np.flatnonzero(values.folds == validation_fold) for task, values in data.items()
    }
    best_nll = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(max_epochs):
        model.train()
        running = torch.zeros((), device=target_device)
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = _task_balanced_loss(model, data, train, batch_per_task, rng, target_device)
            loss.backward()
            optimizer.step()
            running += loss.detach()
        validation_nll, _ = evaluate_task_balanced_nll(model, data, validation, target_device)
        history.append(
            {
                "epoch": epoch,
                "train_nll": float((running / steps_per_epoch).item()),
                "val_nll": validation_nll,
            }
        )
        if validation_nll < best_nll - minimum_improvement:
            best_nll = validation_nll
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("Shared model never produced a validation checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_nll": best_nll, "best_epoch": best_epoch, "history": history}


def clone_state(model: SharedSubspace) -> dict[str, torch.Tensor]:
    return copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})


def fit_shared_all(
    data: dict[str, TaskArrays],
    *,
    input_dim: int,
    shared_dim: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_per_task: int,
    steps_per_epoch: int,
    seed: int,
    device: str = "cuda:0",
) -> SharedSubspace:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    target_device = torch.device(device)
    _maybe_stage_task_arrays(data, target_device)
    model = SharedSubspace(
        input_dim,
        shared_dim,
        {task: len(values.classes) for task, values in data.items()},
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    indices = {task: np.arange(len(values.labels)) for task, values in data.items()}
    for _ in range(epochs):
        model.train()
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = _task_balanced_loss(
                model,
                data,
                indices,
                batch_per_task,
                rng,
                target_device,
            )
            loss.backward()
            optimizer.step()
    return model
