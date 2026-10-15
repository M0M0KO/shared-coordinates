"""Causal-coordinate readout model from the original proposal."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass
class IndividualTaskData:
    states: np.ndarray | torch.Tensor
    labels: np.ndarray | torch.Tensor
    participants: np.ndarray
    folds: np.ndarray
    family: str
    classes: tuple[str, ...]


class IndividualReadout(nn.Module):
    def __init__(
        self,
        shared_dim: int,
        rank: int,
        participants: int,
        population_weights: dict[str, torch.Tensor],
        population_biases: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.raw_directions = nn.Parameter(torch.randn(shared_dim, rank) / shared_dim**0.5)
        self.raw_coordinates = nn.Parameter(torch.randn(participants, rank) * 0.01)
        self.task_effects = nn.ParameterDict(
            {
                task: nn.Parameter(torch.randn(weight.shape[0], rank) * 0.01)
                for task, weight in population_weights.items()
            }
        )
        self.population_weights = nn.ParameterDict(
            {
                task: nn.Parameter(weight.clone(), requires_grad=False)
                for task, weight in population_weights.items()
            }
        )
        self.population_biases = nn.ParameterDict(
            {
                task: nn.Parameter(bias.clone(), requires_grad=False)
                for task, bias in population_biases.items()
            }
        )

    def directions(self) -> torch.Tensor:
        return torch.linalg.qr(self.raw_directions, mode="reduced").Q

    def coordinates(self) -> torch.Tensor:
        centered = self.raw_coordinates - self.raw_coordinates.mean(dim=0, keepdim=True)
        scale = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
        return centered / scale

    def forward(
        self,
        task: str,
        states: torch.Tensor,
        participant_indices: torch.Tensor,
        *,
        directions: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        population = states @ self.population_weights[task].T + self.population_biases[task]
        state_coordinates = states @ (self.directions() if directions is None else directions)
        all_coordinates = self.coordinates() if coordinates is None else coordinates
        person_coordinates = all_coordinates[participant_indices]
        modulation = (state_coordinates * person_coordinates) @ self.task_effects[task].T
        return population + modulation

    def fixed_logits(
        self,
        task: str,
        states: torch.Tensor,
        coordinates: torch.Tensor,
        *,
        directions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        population = states @ self.population_weights[task].T + self.population_biases[task]
        basis = self.directions() if directions is None else directions
        modulation = (states @ basis * coordinates) @ self.task_effects[task].T
        return population + modulation


def stage_individual_task_data(
    data: dict[str, IndividualTaskData],
    device: torch.device | str,
) -> dict[str, IndividualTaskData]:
    """Place immutable individual-level states and labels on one device once."""

    target = torch.device(device)
    if target.type == "cuda":
        required = sum(
            (
                values.states.numel()
                if isinstance(values.states, torch.Tensor)
                else values.states.size
            )
            * 4
            + (
                values.labels.numel()
                if isinstance(values.labels, torch.Tensor)
                else values.labels.size
            )
            * 8
            for values in data.values()
            if not (
                isinstance(values.states, torch.Tensor)
                and values.states.device == target
                and isinstance(values.labels, torch.Tensor)
                and values.labels.device == target
            )
        )
        free, _total = torch.cuda.mem_get_info(target)
        reserve_gib = float(os.environ.get("CM_GPU_RESERVE_GIB", "8"))
        if required + int(reserve_gib * 2**30) > free:
            raise RuntimeError(
                "Insufficient free GPU memory to stage individual-task data: "
                f"required={required / 2**30:.2f} GiB, free={free / 2**30:.2f} GiB, "
                f"reserve={reserve_gib:.2f} GiB"
            )
    for values in data.values():
        if not isinstance(values.states, torch.Tensor):
            values.states = torch.tensor(
                np.asarray(values.states), dtype=torch.float32, device=target
            )
        elif values.states.device != target or values.states.dtype != torch.float32:
            values.states = values.states.to(device=target, dtype=torch.float32)
        if not isinstance(values.labels, torch.Tensor):
            values.labels = torch.as_tensor(values.labels, dtype=torch.long, device=target)
        elif values.labels.device != target or values.labels.dtype != torch.long:
            values.labels = values.labels.to(device=target, dtype=torch.long)
    return data


def _maybe_stage_individual_task_data(
    data: dict[str, IndividualTaskData], device: torch.device
) -> dict[str, IndividualTaskData]:
    mode = os.environ.get("CM_STAGE_TASK_ARRAYS", "auto").lower()
    if mode in {"0", "false", "off", "stream"}:
        return data
    try:
        return stage_individual_task_data(data, device)
    except RuntimeError as error:
        if mode in {"1", "true", "on", "force"}:
            raise
        warnings.warn(f"Falling back to streamed individual-task data: {error}", stacklevel=2)
        return data


def _individual_batch(
    values: IndividualTaskData,
    selected: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.as_tensor(selected, dtype=torch.long, device=device)
    if isinstance(values.states, torch.Tensor):
        states = torch.index_select(values.states, 0, index)
    else:
        states = torch.as_tensor(values.states[selected], dtype=torch.float32, device=device)
    if isinstance(values.labels, torch.Tensor):
        labels = torch.index_select(values.labels, 0, index)
    else:
        labels = torch.as_tensor(values.labels[selected], dtype=torch.long, device=device)
    return states, labels, index


def load_individual_task_data(
    projected_root: Path,
    state_root: Path,
    shared_basis: np.ndarray,
    task_families: dict[str, str],
    classes: dict[str, tuple[str, ...]],
) -> dict[str, IndividualTaskData]:
    participant_values: set[str] = set()
    metadata_by_task: dict[str, pd.DataFrame] = {}
    for task in task_families:
        metadata = pd.read_parquet(
            state_root / task / "metadata.parquet",
            columns=["participant", "response", "fold"],
        )
        metadata_by_task[task] = metadata
        participant_values.update(metadata["participant"].unique())
    participant_index = {value: index for index, value in enumerate(sorted(participant_values))}
    output: dict[str, IndividualTaskData] = {}
    for task, family in task_families.items():
        projected = np.load(projected_root / f"{task}.npy", mmap_mode="r")
        states = np.asarray(projected[:, : shared_basis.shape[0]], dtype=np.float32) @ shared_basis
        metadata = metadata_by_task[task]
        class_index = {value: index for index, value in enumerate(classes[task])}
        labels = metadata["response"].map(class_index).to_numpy(np.int64)
        if np.any(pd.isna(labels)):
            raise RuntimeError(f"Unknown response class in {task}")
        output[task] = IndividualTaskData(
            states=states.astype(np.float32),
            labels=labels,
            participants=metadata["participant"].map(participant_index).to_numpy(np.int32),
            folds=metadata["fold"].to_numpy(np.int8),
            family=family,
            classes=classes[task],
        )
    return output


def fit_individual_population(
    data: dict[str, IndividualTaskData],
    population_weights: dict[str, torch.Tensor],
    population_biases: dict[str, torch.Tensor],
    *,
    rank: int,
    included_folds: set[int],
    learning_rate: float,
    regularization: float,
    epochs: int,
    batch_per_task: int,
    steps_per_epoch: int,
    seed: int,
    device: str = "cuda:0",
) -> tuple[IndividualReadout, dict[int, int]]:
    target_device = torch.device(device)
    _maybe_stage_individual_task_data(data, target_device)
    included_participants = sorted(
        {
            int(participant)
            for values in data.values()
            for participant, fold in zip(values.participants, values.folds, strict=True)
            if int(fold) in included_folds
        }
    )
    remap = {participant: index for index, participant in enumerate(included_participants)}
    model = IndividualReadout(
        next(iter(data.values())).states.shape[1],
        rank,
        len(included_participants),
        population_weights,
        population_biases,
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    pools = {
        task: np.flatnonzero(np.isin(values.folds, list(included_folds)))
        for task, values in data.items()
    }
    remapped_participants = {
        task: torch.as_tensor(
            [remap.get(int(value), -1) for value in values.participants],
            dtype=torch.long,
            device=target_device,
        )
        for task, values in data.items()
    }
    for _ in range(epochs):
        model.train()
        for _ in range(steps_per_epoch):
            losses: list[torch.Tensor] = []
            optimizer.zero_grad(set_to_none=True)
            directions = model.directions()
            coordinates = model.coordinates()
            for task, values in data.items():
                pool = pools[task]
                selected = rng.choice(pool, size=batch_per_task, replace=len(pool) < batch_per_task)
                states, labels, index = _individual_batch(values, selected, target_device)
                people = torch.index_select(remapped_participants[task], 0, index)
                losses.append(
                    nn.functional.cross_entropy(
                        model(
                            task,
                            states,
                            people,
                            directions=directions,
                            coordinates=coordinates,
                        ),
                        labels,
                    )
                )
            penalty = regularization * (
                model.raw_coordinates.square().mean()
                + torch.stack(
                    [value.square().mean() for value in model.task_effects.values()]
                ).mean()
            )
            loss = torch.stack(losses).mean() + penalty
            loss.backward()
            optimizer.step()
    return model, remap


def infer_new_coordinates(
    model: IndividualReadout,
    data: dict[str, IndividualTaskData],
    *,
    participant_ids: list[int],
    source_families: set[str],
    regularization: float,
    steps: int = 300,
    learning_rate: float = 0.03,
    device: str = "cuda:0",
) -> torch.Tensor:
    target_device = torch.device(device)
    _maybe_stage_individual_task_data(data, target_device)
    remap = {participant: index for index, participant in enumerate(participant_ids)}
    coordinates = nn.Parameter(
        torch.zeros(len(participant_ids), model.raw_coordinates.shape[1], device=target_device)
    )
    optimizer = torch.optim.Adam([coordinates], lr=learning_rate)
    with torch.no_grad():
        directions = model.directions().detach()
    source_batches: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    for task, values in data.items():
        if values.family not in source_families:
            continue
        selected = np.flatnonzero(np.isin(values.participants, participant_ids))
        if not len(selected):
            continue
        states, labels, _index = _individual_batch(values, selected, target_device)
        people = torch.as_tensor(
            [remap[int(value)] for value in values.participants[selected]],
            dtype=torch.long,
            device=target_device,
        )
        with torch.no_grad():
            population = (
                states @ model.population_weights[task].detach().T
                + model.population_biases[task].detach()
            )
            state_coordinates = states @ directions
            effects = model.task_effects[task].detach()
        source_batches.append((population, state_coordinates, effects, people, labels))
    if not source_batches:
        raise RuntimeError("No source-family observations for coordinate inference")
    for _ in range(steps):
        losses: list[torch.Tensor] = []
        optimizer.zero_grad(set_to_none=True)
        for population, state_coordinates, effects, people, labels in source_batches:
            losses.append(
                nn.functional.cross_entropy(
                    population + (state_coordinates * coordinates[people]) @ effects.T,
                    labels,
                )
            )
        loss = torch.stack(losses).mean() + regularization * coordinates.square().mean()
        loss.backward()
        optimizer.step()
    return coordinates.detach()


@torch.inference_mode()
def evaluate_new_coordinates(
    model: IndividualReadout,
    data: dict[str, IndividualTaskData],
    participant_ids: list[int],
    coordinates: torch.Tensor,
    target_families: set[str],
    *,
    device: str = "cuda:0",
) -> list[dict[str, Any]]:
    target_device = torch.device(device)
    _maybe_stage_individual_task_data(data, target_device)
    directions = model.directions()
    remap = {participant: index for index, participant in enumerate(participant_ids)}
    rows: list[dict[str, Any]] = []
    for task, values in data.items():
        if values.family not in target_families:
            continue
        for participant in participant_ids:
            selected = np.flatnonzero(values.participants == participant)
            if not len(selected):
                continue
            states, labels, _index = _individual_batch(values, selected, target_device)
            person_coordinates = coordinates[remap[participant]].expand(len(selected), -1)
            individual = nn.functional.cross_entropy(
                model.fixed_logits(task, states, person_coordinates, directions=directions),
                labels,
            )
            population_logits = (
                states @ model.population_weights[task].T + model.population_biases[task]
            )
            population = nn.functional.cross_entropy(population_logits, labels)
            rows.append(
                {
                    "participant_index": participant,
                    "task": task,
                    "task_family": values.family,
                    "population_nll": float(population.cpu()),
                    "individual_nll": float(individual.cpu()),
                    "delta_nll": float((population - individual).cpu()),
                    "trials": len(selected),
                }
            )
    return rows
