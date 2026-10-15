"""Cluster-aware original-profile inference utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def participant_sign_permutation(
    frame: pd.DataFrame,
    value: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """One-sided participant-cluster sign permutation after equal task averaging."""

    participant_values = frame.groupby("participant")[value].mean().dropna().to_numpy()
    if len(participant_values) < 2:
        raise ValueError("At least two participant clusters are required")
    observed = float(participant_values.mean())
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        signs = rng.choice((-1.0, 1.0), size=(stop - start, len(participant_values)))
        null[start:stop] = (signs * participant_values).mean(axis=1)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (replicates + 1))
    return observed, p_value, null


def two_way_cluster_bootstrap(
    frame: pd.DataFrame,
    value: str,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Resample participant and task clusters independently with replacement."""

    cell = frame.groupby(["participant", "task"], as_index=False)[value].mean()
    participants = np.asarray(sorted(cell["participant"].unique()))
    tasks = np.asarray(sorted(cell["task"].unique()))
    pivot = cell.pivot(index="participant", columns="task", values=value).reindex(
        index=participants, columns=tasks
    )
    values = pivot.to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_people = rng.integers(0, len(participants), len(participants))
        sampled_tasks = rng.integers(0, len(tasks), len(tasks))
        output[replicate] = np.nanmean(values[np.ix_(sampled_people, sampled_tasks)])
    return output


def holm_adjust(p_values: dict[str, float], alpha: float) -> dict[str, dict[str, Any]]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * p_values[name]))
        adjusted[name] = running
    return {
        name: {
            "p_value": float(p_values[name]),
            "holm_adjusted_p": float(adjusted[name]),
            "reject": bool(adjusted[name] < alpha),
        }
        for name in p_values
    }


def crossed_random_intercept_test(frame: pd.DataFrame, value: str) -> dict[str, Any]:
    """Fit an intercept-only model with crossed participant and task random intercepts."""

    from statsmodels.formula.api import mixedlm

    cell = frame.groupby(["participant", "task"], as_index=False)[value].mean()
    if cell["participant"].nunique() < 2 or cell["task"].nunique() < 2:
        raise ValueError("Crossed mixed-effects inference requires multiple participants and tasks")
    model_frame = cell.rename(columns={value: "endpoint_value"}).copy()
    model_frame["all_observations"] = "all"
    fitted = mixedlm(
        "endpoint_value ~ 1",
        model_frame,
        groups=model_frame["all_observations"],
        re_formula="0",
        vc_formula={
            "participant": "0 + C(participant)",
            "task": "0 + C(task)",
        },
    ).fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    estimate = float(fitted.fe_params["Intercept"])
    standard_error = float(fitted.bse_fe["Intercept"])
    z_value = estimate / standard_error
    variance_components = {
        name: float(value)
        for name, value in zip(fitted.model.exog_vc.names, fitted.vcomp, strict=True)
    }
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "z_value": z_value,
        "one_sided_mixed_effects_p": float(norm.sf(z_value)),
        "converged": bool(fitted.converged),
        "participant_random_intercept_variance": variance_components["participant"],
        "task_random_intercept_variance": variance_components["task"],
        "participant_clusters": int(cell["participant"].nunique()),
        "task_clusters": int(cell["task"].nunique()),
        "participant_task_cells": len(cell),
    }


def summarize_cluster_test(
    frame: pd.DataFrame,
    value: str,
    *,
    bootstrap_replicates: int,
    permutation_replicates: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    estimate, p_value, null = participant_sign_permutation(
        frame,
        value,
        replicates=permutation_replicates,
        seed=seed,
    )
    bootstrap = two_way_cluster_bootstrap(
        frame,
        value,
        replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    summary = {
        "estimate": estimate,
        "ci_95_low": float(np.nanpercentile(bootstrap, 2.5)),
        "ci_95_high": float(np.nanpercentile(bootstrap, 97.5)),
        "one_sided_participant_sign_permutation_p": p_value,
        "participant_clusters": int(frame["participant"].nunique()),
        "task_clusters": int(frame["task"].nunique()),
        "participant_task_cells": int(frame[["participant", "task"]].drop_duplicates().shape[0]),
    }
    return summary, bootstrap, null
