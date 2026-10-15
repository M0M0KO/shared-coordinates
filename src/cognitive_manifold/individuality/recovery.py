"""Independent-coordinate alignment and identity-retrieval statistics."""

from __future__ import annotations

import numpy as np


def orthogonal_procrustes_mapping(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return R minimizing ||A - B R||_F with R.T R = I."""

    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("A and B must be equal two-dimensional coordinate matrices")
    u, _singular, vt = np.linalg.svd(b.T @ a, full_matrices=False)
    return u @ vt


def rowwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(a_centered, axis=1) * np.linalg.norm(b_centered, axis=1)
    return np.divide(
        np.sum(a_centered * b_centered, axis=1),
        denominator,
        out=np.full(len(a), np.nan),
        where=denominator > 0,
    )


def canonical_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Canonical correlations using ridge-stabilized covariance whitening."""

    if a.shape[0] != b.shape[0] or a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must have the same observations")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denominator = max(len(a) - 1, 1)
    ca = a.T @ a / denominator + np.eye(a.shape[1]) * 1e-8
    cb = b.T @ b / denominator + np.eye(b.shape[1]) * 1e-8
    cab = a.T @ b / denominator

    def inverse_sqrt(value: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(value)
        return (eigenvectors / np.sqrt(np.maximum(eigenvalues, 1e-8))) @ eigenvectors.T

    whitened = inverse_sqrt(ca) @ cab @ inverse_sqrt(cb)
    return np.clip(np.linalg.svd(whitened, compute_uv=False), 0.0, 1.0)


def retrieval_predictions(a: np.ndarray, aligned_b: np.ndarray) -> np.ndarray:
    distances = np.square(a[:, None, :] - aligned_b[None, :, :]).sum(axis=2)
    return distances.argmin(axis=1)


def retrieval_accuracy(a: np.ndarray, aligned_b: np.ndarray) -> float:
    predictions = retrieval_predictions(a, aligned_b)
    return float(np.mean(predictions == np.arange(len(a))))


def identity_permutation_test(
    a: np.ndarray,
    aligned_b: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    observed = retrieval_accuracy(a, aligned_b)
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        null[index] = retrieval_accuracy(a, aligned_b[rng.permutation(len(aligned_b))])
    p_value = float((1 + np.count_nonzero(null >= observed)) / (replicates + 1))
    return observed, p_value, null


def global_correlation_permutation_test(
    a: np.ndarray,
    aligned_b: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    if a.shape != aligned_b.shape or a.ndim != 2:
        raise ValueError("Aligned coordinate matrices must have equal two-dimensional shapes")
    observed = float(np.corrcoef(a.ravel(), aligned_b.ravel())[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        shuffled = aligned_b[rng.permutation(len(aligned_b))]
        null[index] = np.corrcoef(a.ravel(), shuffled.ravel())[0, 1]
    p_value = float((1 + np.count_nonzero(null >= observed)) / (replicates + 1))
    return observed, p_value, null
