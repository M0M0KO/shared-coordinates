"""Independent intrinsic, subspace and trajectory geometry for the original profile."""

from __future__ import annotations

import numpy as np


def participation_ratio(eigenvalues: np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        raise ValueError("Participation ratio requires positive finite eigenvalues")
    return float(values.sum() ** 2 / np.square(values).sum())


def two_nn_dimension(neighbor_distances: np.ndarray) -> float:
    distances = np.asarray(neighbor_distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] < 3:
        raise ValueError("TWO-NN requires self plus at least two neighbor distances")
    first = distances[:, 1]
    second = distances[:, 2]
    valid = np.isfinite(first) & np.isfinite(second) & (first > 0) & (second > first)
    log_ratios = np.log(second[valid] / first[valid])
    if len(log_ratios) < 10 or log_ratios.mean() <= 0:
        raise ValueError("Too few nondegenerate TWO-NN distance ratios")
    return float(1.0 / log_ratios.mean())


def levina_bickel_dimension(neighbor_distances: np.ndarray, neighbors: int) -> float:
    distances = np.asarray(neighbor_distances, dtype=np.float64)
    if neighbors < 3 or distances.ndim != 2 or distances.shape[1] <= neighbors:
        raise ValueError("Levina-Bickel input does not contain the requested neighbors")
    local = distances[:, 1 : neighbors + 1]
    radius = local[:, -1]
    interior = local[:, :-1]
    valid = (
        np.isfinite(radius)
        & (radius > 0)
        & np.isfinite(interior).all(axis=1)
        & (interior > 0).all(axis=1)
    )
    denominator = np.log(radius[valid, None] / interior[valid]).mean(axis=1)
    estimates = 1.0 / denominator[denominator > 0]
    if len(estimates) < 10:
        raise ValueError("Too few nondegenerate Levina-Bickel neighborhoods")
    return float(np.mean(estimates))


def subspace_angles_and_alignment(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, float]:
    singular_values = np.linalg.svd(
        np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64),
        compute_uv=False,
    )
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return np.arccos(singular_values), float(np.square(singular_values).mean())


def trajectory_summary(states: np.ndarray, epsilon: float = 1e-8) -> dict[str, float | int]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("A trajectory requires at least three state vectors")
    steps = np.diff(values, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    total = float(lengths.sum())
    displacement = float(np.linalg.norm(values[-1] - values[0]))
    valid = (lengths[:-1] > epsilon) & (lengths[1:] > epsilon)
    cosines = np.sum(steps[:-1] * steps[1:], axis=1) / np.maximum(
        lengths[:-1] * lengths[1:], epsilon
    )
    curvatures = np.arccos(np.clip(cosines[valid], -1.0, 1.0))
    return {
        "states": len(values),
        "mean_step_length": float(lengths.mean()),
        "total_path_length": total,
        "endpoint_displacement": displacement,
        "tortuosity": total / (displacement + epsilon),
        "mean_discrete_curvature": float(curvatures.mean()) if len(curvatures) else float("nan"),
    }
