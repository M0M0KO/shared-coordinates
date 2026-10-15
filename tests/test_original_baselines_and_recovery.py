from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from cognitive_manifold.geometry.characterization import (
    levina_bickel_dimension,
    participation_ratio,
    subspace_angles_and_alignment,
    trajectory_summary,
    two_nn_dimension,
)
from cognitive_manifold.geometry.preprocessing import project_state_file
from cognitive_manifold.individuality.baselines import fit_five_factor_scores
from cognitive_manifold.individuality.recovery import (
    canonical_correlations,
    global_correlation_permutation_test,
    identity_permutation_test,
    orthogonal_procrustes_mapping,
    retrieval_accuracy,
)

ROOT = Path(__file__).resolve().parents[1]


def _individuality_script():
    path = ROOT / "scripts/original/05_fit_individuality.py"
    spec = importlib.util.spec_from_file_location("fit_original_individuality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_released_factor_audit_does_not_confuse_code_with_fitted_results(tmp_path) -> None:
    module = _individuality_script()
    for relative in module.RELEASED_FACTOR_CODE:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    (tmp_path / "Results").mkdir()
    audit = module._released_factor_audit(tmp_path)
    assert audit["status"] == "released_source_code_present_without_saved_fitted_objects"
    assert not audit["exact_released_reconstruction_completed"]
    assert audit["development_only_refit_completed"]
    saved = tmp_path / "Results/dimensional_structure/release/Output/task_results.pkl"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"historical fitted object")
    audit = module._released_factor_audit(tmp_path)
    assert audit["status"] == "released_fitted_objects_present_but_not_silently_substituted"
    assert audit["saved_result_objects"] == [
        "Results/dimensional_structure/release/Output/task_results.pkl"
    ]


def test_five_factor_fit_uses_all_complete_t1_variables_without_t2_name_intersection(
    tmp_path,
) -> None:
    rng = np.random.default_rng(3)
    participant_ids = [f"s{index}" for index in range(18)]
    complete = pd.DataFrame(
        rng.normal(size=(18, 4)),
        index=participant_ids,
        columns=["a", "b", "c.logTr", "d"],
    )
    retest = pd.DataFrame(
        rng.normal(size=(6, 4)),
        index=participant_ids[-6:],
        columns=["a", "b", "c.ReflogTr", "d"],
    )
    complete_path = tmp_path / "complete.csv"
    retest_path = tmp_path / "retest.csv"
    complete.to_csv(complete_path)
    retest.to_csv(retest_path)
    result = fit_five_factor_scores(
        complete_path,
        retest_path,
        set(participant_ids[:-6]),
        set(participant_ids[-6:]),
        components=2,
        random_state=9,
        rotation="varimax",
    )
    assert result.variables == tuple(complete.columns)
    assert result.loadings.shape == (4, 2)
    assert result.confirmation_t1.shape == (6, 2)
    assert result.confirmation_t2 is None


def test_procrustes_recovers_rotated_coordinates() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(size=(80, 5))
    rotation = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    b = a @ rotation.T
    recovered = orthogonal_procrustes_mapping(a, b)
    aligned = b @ recovered
    assert np.allclose(aligned, a, atol=1e-10)
    assert retrieval_accuracy(a, aligned) == 1.0
    assert np.all(canonical_correlations(a, aligned) > 0.999999)


def test_identity_shuffle_collapses_retrieval_toward_chance() -> None:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(40, 6))
    observed, p_value, null = identity_permutation_test(
        a,
        a.copy(),
        replicates=500,
        seed=11,
    )
    assert observed == 1.0
    assert p_value <= 2 / 501
    assert float(null.mean()) < 0.08


def test_global_coordinate_correlation_permutation() -> None:
    rng = np.random.default_rng(17)
    coordinates = rng.normal(size=(50, 4))
    observed, p_value, null = global_correlation_permutation_test(
        coordinates,
        coordinates.copy(),
        replicates=500,
        seed=23,
    )
    assert observed > 0.999999
    assert p_value <= 2 / 501
    assert float(np.nanmean(null)) < 0.1


def test_geometry_characterization_primitives() -> None:
    assert participation_ratio(np.ones(4)) == 4.0
    first = np.eye(4)[:, :2]
    second = first.copy()
    angles, alignment = subspace_angles_and_alignment(first, second)
    assert np.allclose(angles, 0)
    assert alignment == 1.0
    summary = trajectory_summary(np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]))
    assert np.isclose(summary["mean_step_length"], 1.0)
    assert np.isclose(summary["mean_discrete_curvature"], np.pi / 2)


def test_neighbor_dimension_estimators_return_positive_values() -> None:
    rng = np.random.default_rng(19)
    points = rng.normal(size=(200, 3))
    from sklearn.neighbors import NearestNeighbors

    distances, _ = NearestNeighbors(n_neighbors=31).fit(points).kneighbors(points)
    assert two_nn_dimension(distances) > 0
    assert levina_bickel_dimension(distances, 10) > 0


def test_projection_applies_both_task_balanced_and_pca_center(tmp_path) -> None:
    source = tmp_path / "states.npy"
    destination = tmp_path / "projected.npy"
    states = np.asarray([[10.0, 8.0], [5.0, 3.0]], dtype=np.float32)
    np.save(source, states)
    mean = np.asarray([3.0, 2.0], dtype=np.float32)
    pca_center = np.asarray([1.0, 2.0], dtype=np.float32)
    components = np.eye(2, dtype=np.float32)
    project_state_file(
        source,
        destination,
        mean,
        components,
        pca_center=pca_center,
        chunk_size=1,
        device="cpu",
    )
    expected = states - mean - pca_center
    assert np.allclose(np.load(destination), expected)
