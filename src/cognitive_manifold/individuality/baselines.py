"""Original-profile human-only factor and fixed-feature readout baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import FactorAnalysis
from torch import nn

from cognitive_manifold.individuality.model import (
    IndividualTaskData,
    _individual_batch,
    _maybe_stage_individual_task_data,
)


@dataclass(frozen=True)
class FiveFactorScores:
    development: pd.DataFrame
    confirmation_t1: pd.DataFrame
    confirmation_t2: pd.DataFrame | None
    loadings: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    variables: tuple[str, ...]


def fit_five_factor_scores(
    complete_path: Path,
    retest_path: Path,
    development_ids: set[str],
    confirmation_ids: set[str],
    *,
    components: int,
    random_state: int,
    rotation: str,
) -> FiveFactorScores:
    """Fit five factors on released task matrices without using confirmation rows."""

    complete = pd.read_csv(complete_path, index_col=0)
    retest = pd.read_csv(retest_path, index_col=0)
    complete.index = complete.index.astype(str).str.strip()
    retest.index = retest.index.astype(str).str.strip()
    variables = tuple(complete.columns)
    missing_development = development_ids - set(complete.index)
    if missing_development:
        raise RuntimeError(f"Factor source misses development IDs: {len(missing_development)}")
    missing_t1 = confirmation_ids - set(complete.index)
    missing_t2 = confirmation_ids - set(retest.index)
    if missing_t1 or missing_t2:
        raise RuntimeError(
            f"Factor source misses confirmation IDs: T1={len(missing_t1)}, T2={len(missing_t2)}"
        )
    development_order = sorted(development_ids)
    confirmation_order = sorted(confirmation_ids)
    train = complete.loc[development_order].to_numpy(np.float64)
    if not np.isfinite(train).all():
        raise RuntimeError("Released taskdata_imputed matrix still contains missing values")
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale[scale < 1e-8] = 1.0
    estimator = FactorAnalysis(
        n_components=components,
        rotation=rotation,
        random_state=random_state,
    )
    estimator.fit((train - mean) / scale)

    def transform(frame: pd.DataFrame, order: list[str]) -> np.ndarray:
        values = frame.loc[order].to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError("Released factor source contains non-finite values")
        return estimator.transform((values - mean) / scale)

    dev_scores = transform(complete, development_order)
    t1_scores = transform(complete, confirmation_order)
    loadings = estimator.components_.T.copy()
    for factor in range(loadings.shape[1]):
        anchor = int(np.argmax(np.abs(loadings[:, factor])))
        if loadings[anchor, factor] < 0:
            loadings[:, factor] *= -1
            dev_scores[:, factor] *= -1
            t1_scores[:, factor] *= -1
    columns = [f"factor_{index + 1}" for index in range(components)]
    return FiveFactorScores(
        development=pd.DataFrame(dev_scores, index=development_order, columns=columns),
        confirmation_t1=pd.DataFrame(t1_scores, index=confirmation_order, columns=columns),
        # Retest contains 129 variables, but 24 use different data-transform suffixes than T1.
        # The original T1-to-T2 behavioral baseline requires only frozen T1 factors; inventing a
        # cross-wave feature mapping here would change that baseline and compromise provenance.
        confirmation_t2=None,
        loadings=loadings.astype(np.float32),
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        variables=variables,
    )


class FixedFeatureReadout(nn.Module):
    """State-dependent readout conditioned on fixed participant features."""

    def __init__(
        self,
        shared_dim: int,
        feature_dim: int,
        rank: int,
        population_weights: dict[str, torch.Tensor],
        population_biases: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.raw_directions = nn.Parameter(torch.randn(shared_dim, rank) / shared_dim**0.5)
        self.feature_projection = nn.Parameter(torch.randn(feature_dim, rank) * 0.01)
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

    def forward(
        self,
        task: str,
        states: torch.Tensor,
        features: torch.Tensor,
        *,
        directions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        population = states @ self.population_weights[task].T + self.population_biases[task]
        coordinates = features @ self.feature_projection
        basis = self.directions() if directions is None else directions
        modulation = (states @ basis * coordinates) @ self.task_effects[task].T
        return population + modulation


def fit_fixed_feature_readout(
    data: dict[str, IndividualTaskData],
    participant_features: np.ndarray,
    population_weights: dict[str, torch.Tensor],
    population_biases: dict[str, torch.Tensor],
    *,
    rank: int,
    learning_rate: float,
    regularization: float,
    epochs: int,
    batch_per_task: int,
    steps_per_epoch: int,
    seed: int,
    device: str = "cuda:0",
) -> FixedFeatureReadout:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    target_device = torch.device(device)
    _maybe_stage_individual_task_data(data, target_device)
    staged_features = torch.as_tensor(
        participant_features, dtype=torch.float32, device=target_device
    )
    model = FixedFeatureReadout(
        next(iter(data.values())).states.shape[1],
        participant_features.shape[1],
        rank,
        population_weights,
        population_biases,
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    for _ in range(epochs):
        model.train()
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            directions = model.directions()
            for task, values in data.items():
                pool = np.arange(len(values.labels))
                selected = rng.choice(
                    pool,
                    size=batch_per_task,
                    replace=len(pool) < batch_per_task,
                )
                states, labels, _index = _individual_batch(values, selected, target_device)
                features = staged_features[
                    torch.as_tensor(
                        values.participants[selected], dtype=torch.long, device=target_device
                    )
                ]
                losses.append(
                    nn.functional.cross_entropy(
                        model(task, states, features, directions=directions), labels
                    )
                )
            penalty = regularization * (
                model.feature_projection.square().mean()
                + torch.stack(
                    [value.square().mean() for value in model.task_effects.values()]
                ).mean()
            )
            loss = torch.stack(losses).mean() + penalty
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.inference_mode()
def evaluate_fixed_feature_readout(
    model: FixedFeatureReadout,
    data: dict[str, IndividualTaskData],
    participant_ids: list[int],
    participant_features: np.ndarray,
    target_families: set[str],
    *,
    device: str = "cuda:0",
) -> list[dict[str, float | int | str]]:
    target_device = torch.device(device)
    _maybe_stage_individual_task_data(data, target_device)
    directions = model.directions()
    staged_features = torch.as_tensor(
        participant_features, dtype=torch.float32, device=target_device
    )
    rows: list[dict[str, float | int | str]] = []
    for task, values in data.items():
        if values.family not in target_families:
            continue
        for participant in participant_ids:
            selected = np.flatnonzero(values.participants == participant)
            if not len(selected):
                continue
            states, labels, _index = _individual_batch(values, selected, target_device)
            features = staged_features[participant].expand(len(selected), -1)
            personalized = nn.functional.cross_entropy(
                model(task, states, features, directions=directions), labels
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
                    "personalized_nll": float(personalized.cpu()),
                    "delta_nll": float((population - personalized).cpu()),
                    "trials": len(selected),
                }
            )
    return rows
