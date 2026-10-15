"""Task-balanced sampling, randomized PCA and chunked state projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap
from sklearn.decomposition import PCA


def task_balanced_indices(rows: int, maximum: int, seed: int, task: str) -> np.ndarray:
    if rows <= maximum:
        return np.arange(rows)
    task_seed = seed ^ int.from_bytes(task.encode("utf-8")[:8].ljust(8, b"\0"), "big")
    rng = np.random.default_rng(task_seed)
    return np.sort(rng.choice(rows, size=maximum, replace=False))


def fit_task_balanced_randomized_pca(
    state_files: dict[str, Path],
    *,
    components: int,
    maximum_per_task: int,
    seed: int,
    unit_length_normalization: bool = False,
) -> tuple[PCA, np.ndarray, dict[str, Any]]:
    samples: list[np.ndarray] = []
    counts: dict[str, int] = {}
    task_means: list[np.ndarray] = []
    for task, path in sorted(state_files.items()):
        states = np.load(path, mmap_mode="r")
        indices = task_balanced_indices(len(states), maximum_per_task, seed, task)
        selected = np.asarray(states[indices], dtype=np.float32)
        samples.append(selected)
        counts[task] = len(selected)
        task_means.append(selected.mean(axis=0, dtype=np.float64))
    global_mean = np.mean(np.stack(task_means), axis=0).astype(np.float32)
    centered_samples = [sample - global_mean for sample in samples]
    if unit_length_normalization:
        centered_samples = [
            sample / np.maximum(np.linalg.norm(sample, axis=1, keepdims=True), 1e-12)
            for sample in centered_samples
        ]
    balanced = np.concatenate(centered_samples, axis=0)
    if components > min(balanced.shape):
        raise ValueError(f"Cannot fit {components} components to {balanced.shape}")
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(balanced)
    evidence = {
        "tasks": len(state_files),
        "sample_counts": counts,
        "total_sampled": int(sum(counts.values())),
        "input_dimension": int(balanced.shape[1]),
        "components": components,
        "seed": seed,
        "unit_length_normalization": unit_length_normalization,
        "post_transform_center_norm": float(np.linalg.norm(pca.mean_)),
    }
    return pca, global_mean, evidence


def project_state_file(
    source: Path,
    destination: Path,
    mean: np.ndarray,
    components: np.ndarray,
    *,
    pca_center: np.ndarray | None = None,
    chunk_size: int = 4096,
    device: str = "cuda:0",
    unit_length_normalization: bool = False,
    remove_first_components: int = 0,
) -> None:
    if remove_first_components < 0 or remove_first_components > components.shape[0]:
        raise ValueError("Removed principal-component count is outside the projected dimension")
    states = np.load(source, mmap_mode="r")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npy")
    projected = open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(len(states), components.shape[0]),
    )
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    pca_center_tensor = torch.as_tensor(
        np.zeros(components.shape[1], dtype=np.float32) if pca_center is None else pca_center,
        dtype=torch.float32,
        device=device,
    )
    component_tensor = torch.as_tensor(components.T, dtype=torch.float32, device=device)
    for start in range(0, len(states), chunk_size):
        stop = min(start + chunk_size, len(states))
        batch = torch.as_tensor(
            np.array(states[start:stop], dtype=np.float32, copy=True),
            device=device,
        )
        centered = batch - mean_tensor
        if unit_length_normalization:
            centered = torch.nn.functional.normalize(centered, dim=1)
        values = (centered - pca_center_tensor) @ component_tensor
        if remove_first_components:
            values[:, :remove_first_components] = 0
        projected[start:stop] = values.to(torch.float16).cpu().numpy()
    projected.flush()
    temporary.replace(destination)
