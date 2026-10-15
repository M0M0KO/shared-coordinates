from __future__ import annotations

import numpy as np
import pandas as pd

from cognitive_manifold.statistics.inference import (
    holm_adjust,
    participant_sign_permutation,
    two_way_cluster_bootstrap,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"participant": f"p{person}", "task": f"t{task}", "effect": 0.2 + person / 100}
            for person in range(20)
            for task in range(4)
        ]
    )


def test_cluster_inference_is_deterministic_and_positive() -> None:
    frame = _frame()
    estimate, p_value, first = participant_sign_permutation(
        frame, "effect", replicates=1000, seed=8
    )
    _estimate, _p, second = participant_sign_permutation(frame, "effect", replicates=1000, seed=8)
    assert estimate > 0
    assert p_value < 0.01
    assert np.array_equal(first, second)
    bootstrap = two_way_cluster_bootstrap(frame, "effect", replicates=200, seed=9)
    assert np.isfinite(bootstrap).all()


def test_holm_adjustment_is_monotone() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5}, 0.05)
    assert adjusted["a"]["holm_adjusted_p"] <= adjusted["b"]["holm_adjusted_p"]
    assert adjusted["b"]["holm_adjusted_p"] <= adjusted["c"]["holm_adjusted_p"]
