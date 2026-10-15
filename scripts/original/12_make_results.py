#!/usr/bin/env python3
"""Freeze original-profile statistics, source data, figures and completion report."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cognitive_manifold.statistics.inference import (
    crossed_random_intercept_test,
    holm_adjust,
    summarize_cluster_test,
    two_way_cluster_bootstrap,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/original/main.yaml"
FALSIFICATION_CONFIG = ROOT / "configs/original/falsification_frozen.yaml"
RESULTS = ROOT / "results/original"
REPORTS = ROOT / "reports/original"
FIGURES = REPORTS / "figures"
SOURCE_DATA = REPORTS / "figure_source_data"
STATISTICS = RESULTS / "primary_statistics.json"
STATISTICS_TABLE = RESULTS / "primary_statistics.parquet"
FALSIFICATION = RESULTS / "falsification_audit.json"
FALSIFICATION_TABLE = RESULTS / "falsification_audit.parquet"
REPORT = REPORTS / "original_results_report.md"
FREEZE = REPORTS / "ORIGINAL_FROZEN.json"
MANIFEST = ROOT / "manifests/private/original/final_results.json"

REQUIRED_TABLES = {
    "shared": RESULTS / "shared_manifold.parquet",
    "shared_controls": RESULTS / "shared_controls.parquet",
    "private_rank": RESULTS / "private_rank.parquet",
    "cross_domain": RESULTS / "cross_domain.parquet",
    "retest": RESULTS / "retest_transfer.parquet",
    "psych201": RESULTS / "psych201_generalization.parquet",
    "jlens": RESULTS / "jlens_access.parquet",
    "causal": RESULTS / "causal_interventions.parquet",
    "intrinsic_dimension": RESULTS / "intrinsic_dimension.parquet",
    "task_subspace_angles": RESULTS / "task_subspace_angles.parquet",
    "shared_trajectories": RESULTS / "shared_trajectories.parquet",
    "representation_sensitivities": RESULTS / "representation_sensitivities.parquet",
    "retained_information": RESULTS / "retained_information.parquet",
}

REQUIRED_FILES = {
    "jlens_statistics": RESULTS / "jlens_statistics.json",
    "causal_statistics": RESULTS / "causal_statistics.json",
    "coordinate_recovery_statistics": RESULTS / "coordinate_recovery_summary.json",
    "psych201_statistics": RESULTS / "psych201_mixed_effects.json",
}
NEURAL_STATUS = RESULTS / "neural_validation_status.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _require_inputs() -> dict[str, Path]:
    missing = [
        str(path)
        for path in (*REQUIRED_TABLES.values(), *REQUIRED_FILES.values())
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"O12 is sealed until required original tables exist: {missing}")
    o11 = RESULTS / "centaur70b_status.json"
    if not o11.exists():
        raise RuntimeError("O12 requires a frozen O11 exact-BF16 result or resource status")
    status = json.loads(o11.read_text())["status"]
    if status not in {"resource_incomplete", "completed"}:
        raise RuntimeError(f"O11 has not reached a terminal original-profile status: {status}")
    return {**REQUIRED_TABLES, **REQUIRED_FILES, "centaur70b_status": o11}


def _primary_frames() -> dict[str, pd.DataFrame]:
    retest = pd.read_parquet(REQUIRED_TABLES["retest"])
    cross_domain = retest.loc[
        (retest["method"] == "manifold")
        & (retest["analysis"] == "primary_cross_domain_T1_leave_one_family_out")
    ].copy()
    cross_time = retest.loc[
        (retest["method"] == "manifold") & (retest["analysis"] == "primary_T1_A_to_T2_B")
    ].copy()
    factor = retest.loc[
        (retest["method"] == "sro_factor") & (retest["analysis"] == "primary_T1_factor_to_T2_B")
    ].copy()
    comparison = cross_time.merge(
        factor[["participant", "task", "delta_nll"]].rename(
            columns={"delta_nll": "factor_delta_nll"}
        ),
        on=["participant", "task"],
        how="inner",
        validate="one_to_one",
    )
    comparison["manifold_minus_factor"] = comparison["delta_nll"] - comparison["factor_delta_nll"]
    if cross_domain.empty or cross_time.empty or comparison.empty:
        raise RuntimeError("One or more frozen primary endpoints have no participant-task cells")
    return {
        "cross_domain": cross_domain,
        "cross_time": cross_time,
        "manifold_minus_factor": comparison,
    }


def run_primary_statistics(config: dict[str, Any]) -> dict[str, Any]:
    frames = _primary_frames()
    bootstrap_replicates = int(config["statistics"]["bootstrap_replicates"])
    permutation_replicates = int(config["statistics"]["permutation_replicates"])
    seed = int(config["execution"]["seed"])
    definitions = {
        "cross_domain": ("delta_nll", seed + 1201),
        "cross_time": ("delta_nll", seed + 1202),
        "manifold_minus_factor": ("manifold_minus_factor", seed + 1203),
    }
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, (column, current_seed) in definitions.items():
        mixed = crossed_random_intercept_test(frames[name], column)
        summary, bootstrap, null = summarize_cluster_test(
            frames[name],
            column,
            bootstrap_replicates=bootstrap_replicates,
            permutation_replicates=permutation_replicates,
            seed=current_seed,
        )
        summary["equal_weighted_estimate"] = summary.pop("estimate")
        summary["mixed_effects"] = mixed
        summary["estimate"] = mixed["estimate"]
        summaries[name] = summary
        rows.append({"endpoint": name, **summary})
        np.savez(
            RESULTS / f"{name}_resampling.npz",
            bootstrap=bootstrap,
            participant_sign_null=null,
        )
    adjusted = holm_adjust(
        {
            name: float(value["mixed_effects"]["one_sided_mixed_effects_p"])
            for name, value in summaries.items()
        },
        float(config["statistics"]["alpha"]),
    )
    for row in rows:
        row.update(adjusted[str(row["endpoint"])])
        summaries[str(row["endpoint"])].update(adjusted[str(row["endpoint"])])
    payload = {
        "schema_version": 1,
        "profile": "original",
        "aggregation": "participant_task_cell_then_equal_task_weighting",
        "bootstrap": "two_way_participant_and_task_cluster",
        "primary_test": "one_sided_crossed_participant_task_random_intercept_mixed_model",
        "complementary_test": "one_sided_participant_cluster_sign_permutation",
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": permutation_replicates,
        "holm_alpha": float(config["statistics"]["alpha"]),
        "endpoints": summaries,
    }
    STATISTICS.parent.mkdir(parents=True, exist_ok=True)
    STATISTICS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(rows).to_parquet(STATISTICS_TABLE, index=False, compression="zstd")
    return payload


def _criterion(status: str, **evidence: Any) -> dict[str, Any]:
    if status not in {"not_falsified", "falsified", "inconclusive"}:
        raise ValueError(status)
    return {"status": status, "evidence": evidence}


def run_falsification_audit(primary: dict[str, Any]) -> dict[str, Any]:
    frozen = yaml.safe_load(FALSIFICATION_CONFIG.read_text())
    alpha = float(frozen["alpha"])
    controls = pd.read_parquet(REQUIRED_TABLES["shared_controls"])
    selection = json.loads((RESULTS / "selected_shared.json").read_text())
    private = json.loads((RESULTS / "selected_private.json").read_text())
    recovery = json.loads((RESULTS / "coordinate_recovery_summary.json").read_text())
    psych = json.loads((RESULTS / "psych201_mixed_effects.json").read_text())
    causal = json.loads((RESULTS / "causal_statistics.json").read_text())

    selected_dimension = int(selection["dimension"])
    selected_controls = controls.loc[
        (controls["dimension"] == selected_dimension)
        | (controls["control"].isin(["full_representation", "null_intercept"]))
    ].copy()
    means: dict[str, float] = {}
    for name in ("shared_behavioral", "task_specific", "full_representation"):
        values = selected_controls.loc[selected_controls["control"] == name, "validation_nll"]
        if values.empty:
            raise RuntimeError(f"Falsification audit is missing shared control: {name}")
        means[name] = float(values.mean())
    random = selected_controls.loc[selected_controls["control"] == "haar_random"].copy()
    if random.empty:
        raise RuntimeError("Falsification audit is missing Haar-random controls")
    random_mean = float(
        random.groupby(["task", "fold"], as_index=False)["validation_nll"]
        .mean()["validation_nll"]
        .mean()
    )
    full_over_random = random_mean - means["full_representation"]
    shared_penalty = means["shared_behavioral"] - means["task_specific"]
    worse_fraction = float(
        frozen["shared_subspace"]["substantially_worse_fraction_of_full_over_random_advantage"]
    )
    worse_threshold = worse_fraction * full_over_random
    shared_status = (
        "inconclusive"
        if full_over_random <= 0
        else "falsified"
        if shared_penalty > worse_threshold
        else "not_falsified"
    )

    selected_rank = int(private["rank"])
    rank_fraction = selected_rank / selected_dimension
    rank_threshold = float(frozen["private_rank"]["approaching_shared_dimension_fraction"])
    recovery_passes = (
        float(recovery["independent_global_procrustes_correlation"]) > 0
        and float(recovery["independent_global_procrustes_permutation_p"]) < alpha
        and float(recovery["independent_top1_retrieval"])
        > float(recovery["independent_top1_chance"])
        and float(recovery["independent_top1_permutation_p"]) < alpha
    )
    psych_passes = float(psych["primary_beta_1"]) > 0 and float(psych["primary_beta_1_p"]) < alpha
    causal_name = f"projection_ablation_strength_{float(frozen['causal_ablation']['strength']):.2f}"
    if causal_name not in causal["tests"]:
        raise RuntimeError(f"Falsification audit is missing causal test: {causal_name}")
    causal_test = causal["tests"][causal_name]
    causal_passes = (
        float(causal_test["estimate"]) > 0
        and float(causal_test["one_sided_participant_permutation_p"]) < alpha
    )

    criteria = {
        "shared_not_substantially_worse_than_task_specific": _criterion(
            shared_status,
            shared_mean_nll=means["shared_behavioral"],
            task_specific_mean_nll=means["task_specific"],
            random_mean_nll=random_mean,
            full_mean_nll=means["full_representation"],
            shared_minus_task_specific=shared_penalty,
            full_over_random_advantage=full_over_random,
            substantially_worse_threshold=worse_threshold,
        ),
        "cross_task_personalization_exceeds_population": _criterion(
            "not_falsified" if primary["endpoints"]["cross_domain"]["reject"] else "falsified",
            **primary["endpoints"]["cross_domain"],
        ),
        "private_rank_does_not_approach_shared_dimension": _criterion(
            "falsified" if rank_fraction >= rank_threshold else "not_falsified",
            selected_private_rank=selected_rank,
            selected_shared_dimension=selected_dimension,
            rank_fraction=rank_fraction,
            approaching_threshold=rank_threshold,
        ),
        "independent_coordinates_correspond": _criterion(
            "not_falsified" if recovery_passes else "falsified",
            global_procrustes_correlation=float(
                recovery["independent_global_procrustes_correlation"]
            ),
            global_procrustes_permutation_p=float(
                recovery["independent_global_procrustes_permutation_p"]
            ),
            top1_retrieval=float(recovery["independent_top1_retrieval"]),
            top1_chance=float(recovery["independent_top1_chance"]),
            top1_permutation_p=float(recovery["independent_top1_permutation_p"]),
        ),
        "t1_coordinates_predict_t2_different_domain_behavior": _criterion(
            "not_falsified" if primary["endpoints"]["cross_time"]["reject"] else "falsified",
            **primary["endpoints"]["cross_time"],
        ),
        "manifold_outperforms_sro_factors_cross_time": _criterion(
            "not_falsified"
            if primary["endpoints"]["manifold_minus_factor"]["reject"]
            else "falsified",
            **primary["endpoints"]["manifold_minus_factor"],
        ),
        "psych201_alignment_predicts_finetuning_gain": _criterion(
            "not_falsified" if psych_passes else "falsified",
            standardized_alignment_beta=float(psych["primary_beta_1"]),
            mixed_model_p=float(psych["primary_beta_1_p"]),
        ),
        "behavioral_ablation_exceeds_matched_random": _criterion(
            "not_falsified" if causal_passes else "falsified",
            test=causal_name,
            estimate=float(causal_test["estimate"]),
            participant_permutation_p=float(causal_test["one_sided_participant_permutation_p"]),
        ),
    }
    payload = {
        "schema_version": 1,
        "profile": "original",
        "falsification_config_sha256": _sha256(FALSIFICATION_CONFIG),
        "alpha": alpha,
        "criteria": criteria,
        "counts": {
            status: sum(item["status"] == status for item in criteria.values())
            for status in ("not_falsified", "falsified", "inconclusive")
        },
        "strong_hypothesis_not_falsified": all(
            item["status"] == "not_falsified" for item in criteria.values()
        ),
    }
    FALSIFICATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(
        [
            {
                "criterion": name,
                "status": item["status"],
                "evidence_json": json.dumps(item["evidence"], sort_keys=True),
            }
            for name, item in criteria.items()
        ]
    ).to_parquet(FALSIFICATION_TABLE, index=False, compression="zstd")
    return payload


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _save_figure(figure: Any, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.tiff", dpi=600, bbox_inches="tight")


def _panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.18,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def _best_dimension_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize the best frozen hyperparameter pair within each dimension."""
    grouped = (
        frame.groupby(["dimension", "learning_rate", "weight_decay"])["validation_nll"]
        .mean()
        .reset_index()
    )
    best = grouped.loc[grouped.groupby("dimension")["validation_nll"].idxmin()]
    selected = frame.merge(
        best[["dimension", "learning_rate", "weight_decay"]],
        on=["dimension", "learning_rate", "weight_decay"],
        how="inner",
        validate="many_to_one",
    )
    return (
        selected.groupby("dimension")["validation_nll"].agg(["mean", "sem", "count"]).reset_index()
    )


def _control_summary(controls: pd.DataFrame) -> pd.DataFrame:
    """Avoid treating task rows or Haar folds as independent replicates."""
    names = {
        "shared_behavioral",
        "task_specific",
        "full_representation",
        "haar_random",
        "shared_reproduced_intact",
        "shared_response_shuffled",
    }
    selected = controls.loc[controls["control"].isin(names)].copy()
    ordinary = selected.loc[selected["control"] != "haar_random"]
    ordinary = ordinary.groupby(["control", "fold"], as_index=False)["validation_nll"].mean()
    haar = selected.loc[selected["control"] == "haar_random"]
    haar = haar.groupby(["control", "replicate"], as_index=False)["validation_nll"].mean()
    return (
        pd.concat(
            [
                ordinary[["control", "validation_nll"]],
                haar[["control", "validation_nll"]],
            ],
            ignore_index=True,
        )
        .groupby("control")["validation_nll"]
        .agg(["mean", "sem", "count"])
        .sort_values("mean")
    )


def _two_way_group_summary(
    frame: pd.DataFrame,
    *,
    group: str,
    value: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Return descriptive two-way participant/task cluster intervals by group."""
    rows: list[dict[str, Any]] = []
    for index, (label, current) in enumerate(frame.groupby(group, sort=True)):
        cells = current.groupby(["participant", "task"], as_index=False)[value].mean()
        if cells["participant"].nunique() < 2 or cells["task"].nunique() < 2:
            raise RuntimeError(
                f"Two-way figure interval for {group}={label} requires multiple "
                "participants and tasks"
            )
        bootstrap = two_way_cluster_bootstrap(
            cells,
            value,
            replicates=replicates,
            seed=seed + index,
        )
        rows.append(
            {
                group: label,
                "mean": float(cells[value].mean()),
                "ci_95_low": float(np.nanpercentile(bootstrap, 2.5)),
                "ci_95_high": float(np.nanpercentile(bootstrap, 97.5)),
                "participant_clusters": int(cells["participant"].nunique()),
                "task_clusters": int(cells["task"].nunique()),
                "participant_task_cells": len(cells),
            }
        )
    return pd.DataFrame(rows)


def figure_1() -> None:
    import matplotlib.pyplot as plt

    shared = pd.read_parquet(REQUIRED_TABLES["shared"])
    controls = pd.read_parquet(REQUIRED_TABLES["shared_controls"])
    base_path = RESULTS / "shared_manifold_llama_base_8b_layer_24.parquet"
    if not base_path.exists():
        raise RuntimeError("Figure 1 requires the frozen base-Llama shared curve")
    base = pd.read_parquet(base_path)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    shared.to_csv(SOURCE_DATA / "Figure1_shared_curve.csv", index=False)
    controls.to_csv(SOURCE_DATA / "Figure1_controls.csv", index=False)
    base.to_csv(SOURCE_DATA / "Figure1_base_curve.csv", index=False)
    selection = json.loads((RESULTS / "selected_shared.json").read_text())
    pd.DataFrame([selection]).to_csv(SOURCE_DATA / "Figure1_shared_selection.csv", index=False)
    summary = _best_dimension_summary(shared)
    base_summary = _best_dimension_summary(base)
    summary.assign(model="minitaur_8b").to_csv(
        SOURCE_DATA / "Figure1_minitaur_dimension_summary.csv", index=False
    )
    base_summary.assign(model="llama_base_8b").to_csv(
        SOURCE_DATA / "Figure1_base_dimension_summary.csv", index=False
    )
    palette = {"minitaur": "#3F7CAC", "base": "#9AA0A6", "signal": "#C45A3C"}
    figure, axes = plt.subplots(2, 2, figsize=(7.09, 5.2), constrained_layout=True)
    axis = axes[0, 0]
    axis.errorbar(
        summary["dimension"],
        summary["mean"],
        yerr=summary["sem"],
        marker="o",
        color=palette["minitaur"],
        label="Minitaur",
    )
    axis.errorbar(
        base_summary["dimension"],
        base_summary["mean"],
        yerr=base_summary["sem"],
        marker="o",
        color=palette["base"],
        label="Base Llama",
    )
    axis.axvline(selection["dimension"], color=palette["signal"], linestyle="--", linewidth=1)
    axis.axhline(
        selection["one_standard_error_threshold"],
        color="#B6B0A8",
        linestyle=":",
        linewidth=0.9,
    )
    axis.set(xscale="log", xlabel="Shared dimension", ylabel="Validation NLL")
    axis.legend()
    _panel_label(axis, "a")
    axis = axes[0, 1]
    selected = controls.loc[
        (controls["dimension"] == int(selection["dimension"]))
        | (controls["control"] == "full_representation")
    ].copy()
    required_controls = {
        "shared_behavioral",
        "task_specific",
        "full_representation",
        "haar_random",
        "shared_reproduced_intact",
        "shared_response_shuffled",
    }
    missing_controls = sorted(required_controls - set(selected["control"]))
    if missing_controls:
        raise RuntimeError(f"Figure 1 is missing frozen controls: {missing_controls}")
    control_summary = _control_summary(selected)
    control_summary.reset_index().to_csv(SOURCE_DATA / "Figure1_control_summary.csv", index=False)
    axis.barh(
        control_summary.index, control_summary["mean"], xerr=control_summary["sem"], color="#A9C5D8"
    )
    axis.set_xlabel("Validation NLL")
    _panel_label(axis, "b")
    axis = axes[1, 0]
    transfer = controls.loc[controls["control"] == "held_out_family_transfer"]
    task_specific = controls.loc[
        controls["control"] == "task_specific", ["task", "fold", "validation_nll"]
    ].rename(columns={"validation_nll": "task_specific_nll"})
    transfer = transfer.merge(
        task_specific,
        on=["task", "fold"],
        how="inner",
        validate="one_to_one",
    )
    transfer["held_out_penalty"] = transfer["validation_nll"] - transfer["task_specific_nll"]
    transfer_folds = transfer.groupby(["held_out_family", "fold"], as_index=False)[
        "held_out_penalty"
    ].mean()
    transfer_summary = transfer_folds.groupby("held_out_family")["held_out_penalty"].agg(
        ["mean", "sem", "count"]
    )
    transfer.to_csv(SOURCE_DATA / "Figure1_held_out_penalty_cells.csv", index=False)
    transfer_folds.to_csv(SOURCE_DATA / "Figure1_held_out_penalty_folds.csv", index=False)
    transfer_summary.reset_index().to_csv(
        SOURCE_DATA / "Figure1_held_out_penalty_summary.csv", index=False
    )
    axis.errorbar(
        np.arange(len(transfer_summary)),
        transfer_summary["mean"],
        yerr=transfer_summary["sem"],
        fmt="o",
        color=palette["minitaur"],
    )
    axis.set_xticks(
        np.arange(len(transfer_summary)), transfer_summary.index, rotation=35, ha="right"
    )
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_ylabel("Held-out minus task-specific NLL")
    _panel_label(axis, "c")
    axis = axes[1, 1]
    random = selected.loc[selected["control"] == "haar_random"]
    random_replicates = random.groupby("replicate", as_index=False)["validation_nll"].mean()
    random_replicates.to_csv(SOURCE_DATA / "Figure1_haar_replicate_summary.csv", index=False)
    observed = selected.loc[selected["control"] == "shared_behavioral", "validation_nll"].mean()
    axis.hist(random_replicates["validation_nll"], bins=25, color="#D6D9DC", edgecolor="white")
    axis.axvline(observed, color=palette["signal"], linewidth=1.5, label="Behavioral")
    axis.set(xlabel="Validation NLL", ylabel="Haar replicates")
    axis.legend()
    _panel_label(axis, "d")
    _save_figure(figure, "Figure1")
    plt.close(figure)


def figure_2() -> None:
    import matplotlib.pyplot as plt

    rank = pd.read_parquet(REQUIRED_TABLES["private_rank"])
    cross = pd.read_parquet(REQUIRED_TABLES["cross_domain"])
    recovery = pd.read_parquet(RESULTS / "coordinate_recovery_confirmation.parquet")
    recovery_summary = json.loads((RESULTS / "coordinate_recovery_summary.json").read_text())
    private_selection = json.loads((RESULTS / "selected_private.json").read_text())
    rank.to_csv(SOURCE_DATA / "Figure2_rank.csv", index=False)
    cross.to_csv(SOURCE_DATA / "Figure2_cross_domain.csv", index=False)
    recovery.to_csv(SOURCE_DATA / "Figure2_recovery.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(7.09, 2.55), constrained_layout=True)
    summary = (
        rank.groupby("rank")["mean_individual_nll"].agg(["mean", "sem", "count"]).reset_index()
    )
    summary.to_csv(SOURCE_DATA / "Figure2_rank_summary.csv", index=False)
    pd.DataFrame([private_selection]).to_csv(
        SOURCE_DATA / "Figure2_private_selection.csv", index=False
    )
    pd.DataFrame(
        [{key: value for key, value in recovery_summary.items() if not isinstance(value, list)}]
    ).to_csv(SOURCE_DATA / "Figure2_recovery_summary.csv", index=False)
    axes[0].errorbar(
        summary["rank"], summary["mean"], yerr=summary["sem"], marker="o", color="#3F7CAC"
    )
    axes[0].axvline(private_selection["rank"], color="#C45A3C", linestyle="--", linewidth=1)
    axes[0].axhline(
        private_selection["one_standard_error_threshold"],
        color="#B6B0A8",
        linestyle=":",
        linewidth=0.9,
    )
    axes[0].set(xscale="log", xlabel="Private rank", ylabel="Cross-domain individual NLL")
    _panel_label(axes[0], "a")
    config = yaml.safe_load(CONFIG.read_text())
    method = _two_way_group_summary(
        cross,
        group="method",
        value="delta_nll",
        replicates=int(config["statistics"]["bootstrap_replicates"]),
        seed=int(config["execution"]["seed"]) + 2201,
    ).set_index("method")
    method.reset_index().to_csv(
        SOURCE_DATA / "Figure2_cross_domain_cluster_summary.csv", index=False
    )
    method_error = np.vstack(
        [
            method["mean"] - method["ci_95_low"],
            method["ci_95_high"] - method["mean"],
        ]
    )
    axes[1].bar(
        method.index,
        method["mean"],
        yerr=method_error,
        color=["#3F7CAC" if value == "manifold" else "#B6B0A8" for value in method.index],
    )
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(ylabel="Development ΔNLL")
    axes[1].tick_params(axis="x", rotation=25)
    _panel_label(axes[1], "b")
    axes[2].hist(recovery["independent_A_B_aligned_distance"], bins=20, color="#A9C5D8")
    axes[2].set(xlabel="A-B aligned distance", ylabel="Participants")
    axes[2].text(
        0.98,
        0.96,
        f"Top-1={recovery_summary['independent_top1_retrieval']:.3f}\nchance={recovery_summary['independent_top1_chance']:.3f}",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
    )
    _panel_label(axes[2], "c")
    _save_figure(figure, "Figure2")
    plt.close(figure)


def figure_3(primary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    retest = pd.read_parquet(REQUIRED_TABLES["retest"])
    primary_rows = retest.loc[
        retest["analysis"].isin(
            [
                "primary_T1_A_to_T2_B",
                "primary_T1_factor_to_T2_B",
                "primary_T1_questionnaire_persona_to_T2_B",
            ]
        )
    ]
    primary_rows.to_csv(SOURCE_DATA / "Figure3_retest_cells.csv", index=False)
    stats = pd.DataFrame(
        [{"endpoint": name, **values} for name, values in primary["endpoints"].items()]
    )
    stats.to_csv(SOURCE_DATA / "Figure3_primary_statistics.csv", index=False)
    config = yaml.safe_load(CONFIG.read_text())
    summary = _two_way_group_summary(
        primary_rows,
        group="method",
        value="delta_nll",
        replicates=int(config["statistics"]["bootstrap_replicates"]),
        seed=int(config["execution"]["seed"]) + 2301,
    )
    summary.to_csv(SOURCE_DATA / "Figure3_method_cluster_summary.csv", index=False)
    method_error = np.vstack(
        [
            summary["mean"] - summary["ci_95_low"],
            summary["ci_95_high"] - summary["mean"],
        ]
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.09, 2.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.5, 1]},
    )
    colors = {"manifold": "#3F7CAC", "sro_factor": "#B6B0A8", "persona": "#D39A5B"}
    axes[0].bar(
        summary["method"],
        summary["mean"],
        yerr=method_error,
        color=[colors[value] for value in summary["method"]],
    )
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set(ylabel="T1→T2 ΔNLL")
    axes[0].tick_params(axis="x", rotation=25)
    _panel_label(axes[0], "a")
    manifold = primary_rows.loc[primary_rows["method"] == "manifold"]
    task = manifold.groupby("task")["delta_nll"].agg(["mean", "sem", "count"]).sort_values("mean")
    task.reset_index().to_csv(SOURCE_DATA / "Figure3_task_summary.csv", index=False)
    axes[1].errorbar(task["mean"], np.arange(len(task)), xerr=task["sem"], fmt="o", color="#3F7CAC")
    axes[1].axvline(0, color="#777777", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(task)), task.index)
    axes[1].set_xlabel("Task ΔNLL")
    _panel_label(axes[1], "b")
    recovery = json.loads((RESULTS / "coordinate_recovery_summary.json").read_text())
    correlations = np.asarray(recovery["time_coordinate_correlations"])
    iccs = np.asarray(recovery["time_coordinate_icc_3_1"])
    pd.DataFrame(
        {
            "coordinate": np.arange(len(correlations)),
            "T1_T2_correlation": correlations,
            "ICC_3_1": iccs,
        }
    ).to_csv(SOURCE_DATA / "Figure3_coordinate_reliability.csv", index=False)
    axes[2].scatter(correlations, iccs, color="#3F7CAC", s=20)
    axes[2].axhline(0, color="#CCCCCC", linewidth=0.7)
    axes[2].axvline(0, color="#CCCCCC", linewidth=0.7)
    axes[2].set(xlabel="T1-T2 coordinate r", ylabel="ICC(3,1)")
    _panel_label(axes[2], "c")
    _save_figure(figure, "Figure3")
    plt.close(figure)


def figure_4() -> None:
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    data = pd.read_parquet(REQUIRED_TABLES["psych201"])
    data.to_csv(SOURCE_DATA / "Figure4_psych201.csv", index=False)
    statistics = json.loads((RESULTS / "psych201_mixed_effects.json").read_text())
    statistics_summary = {
        key: value
        for key, value in statistics.items()
        if isinstance(value, (str, int, float, bool))
    }
    statistics_summary.update({"experiments": len(data), "studies": int(data["study"].nunique())})
    pd.DataFrame([statistics_summary]).to_csv(SOURCE_DATA / "Figure4_statistics.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(7.09, 2.8), constrained_layout=True)
    nuisance_columns = [
        "intrinsic_dimension",
        "mean_log_context_length",
        "response_set_complexity",
        "study",
    ]
    nuisance = ColumnTransformer(
        [
            (
                "continuous",
                StandardScaler(),
                nuisance_columns[:-1],
            ),
            (
                "study",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                ["study"],
            ),
        ]
    ).fit_transform(data[nuisance_columns])
    adjusted_alignment = data["behavioral_alignment_increment"].to_numpy() - (
        LinearRegression().fit(nuisance, data["behavioral_alignment_increment"]).predict(nuisance)
    )
    adjusted_gain = data["fine_tuning_gain"].to_numpy() - (
        LinearRegression().fit(nuisance, data["fine_tuning_gain"]).predict(nuisance)
    )
    partial = pd.DataFrame(
        {
            "experiment": data["experiment"],
            "study": data["study"],
            "adjusted_alignment_increment": adjusted_alignment,
            "adjusted_fine_tuning_gain": adjusted_gain,
        }
    )
    partial.to_csv(SOURCE_DATA / "Figure4_partial_regression.csv", index=False)
    axes[0].scatter(
        adjusted_alignment,
        adjusted_gain,
        s=12,
        alpha=0.65,
        color="#3F7CAC",
    )
    fit = np.polyfit(adjusted_alignment, adjusted_gain, 1)
    x = np.linspace(
        adjusted_alignment.min(),
        adjusted_alignment.max(),
        100,
    )
    axes[0].plot(x, np.polyval(fit, x), color="#C45A3C")
    axes[0].set(
        xlabel="Adjusted alignment increment",
        ylabel="Adjusted fine-tuning gain",
    )
    _panel_label(axes[0], "a")
    axes[1].scatter(
        data["loso_prediction"], data["fine_tuning_gain"], s=12, alpha=0.65, color="#3F7CAC"
    )
    limits = [
        min(data["loso_prediction"].min(), data["fine_tuning_gain"].min()),
        max(data["loso_prediction"].max(), data["fine_tuning_gain"].max()),
    ]
    axes[1].plot(limits, limits, color="#999999", linestyle="--")
    axes[1].set(xlabel="LOSO predicted gain", ylabel="Observed gain")
    axes[1].text(
        0.04,
        0.96,
        f"R2={statistics['loso_r2']:.2f}\nrho={statistics['loso_spearman']:.2f}",
        transform=axes[1].transAxes,
        va="top",
    )
    _panel_label(axes[1], "b")
    _save_figure(figure, "Figure4")
    plt.close(figure)


def figure_5() -> None:
    import matplotlib.pyplot as plt

    jlens = pd.read_parquet(REQUIRED_TABLES["jlens"])
    causal = pd.read_parquet(REQUIRED_TABLES["causal"])
    jlens.to_csv(SOURCE_DATA / "Figure5_jlens.csv", index=False)
    causal.to_csv(SOURCE_DATA / "Figure5_interventions.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(7.09, 2.6), constrained_layout=True)
    observed = jlens.loc[jlens["subspace"] == "behavioral"]
    random = (
        jlens.loc[jlens["subspace"] == "haar_random"]
        .groupby(["model", "layer"])["causal_access"]
        .agg(random_mean="mean", random_sd="std", random_subspaces="count")
        .reset_index()
    )
    access = observed.merge(random, on=["model", "layer"])
    access["causal_access_enrichment"] = access["causal_access"] - access["random_mean"]
    access.to_csv(SOURCE_DATA / "Figure5_access_summary.csv", index=False)
    for model, frame in access.groupby("model"):
        axes[0].plot(frame["layer"], frame["causal_access_enrichment"], marker="o", label=model)
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set(xlabel="Source layer", ylabel="Causal-access enrichment")
    axes[0].legend(fontsize=5)
    _panel_label(axes[0], "a")
    ablation_trials = (
        causal.loc[causal["analysis"] == "projection_ablation"]
        .groupby(
            ["participant", "task", "trial", "strength", "subspace"],
            as_index=False,
        )["delta_nll"]
        .mean()
    )
    ablation_participant_tasks = ablation_trials.groupby(
        ["participant", "task", "strength", "subspace"],
        as_index=False,
    )["delta_nll"].mean()
    ablation_participants = ablation_participant_tasks.groupby(
        ["participant", "strength", "subspace"],
        as_index=False,
    )["delta_nll"].mean()
    ablation = (
        ablation_participants.groupby(["strength", "subspace"])["delta_nll"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    ablation_trials.to_csv(SOURCE_DATA / "Figure5_ablation_trials.csv", index=False)
    ablation_participant_tasks.to_csv(
        SOURCE_DATA / "Figure5_ablation_participant_tasks.csv", index=False
    )
    ablation_participants.to_csv(SOURCE_DATA / "Figure5_ablation_participants.csv", index=False)
    ablation.to_csv(SOURCE_DATA / "Figure5_ablation_summary.csv", index=False)
    for subspace, frame in ablation.groupby("subspace"):
        axes[1].errorbar(
            frame["strength"], frame["mean"], yerr=frame["sem"], marker="o", label=subspace
        )
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set(xlabel="Ablation strength", ylabel="ΔNLL")
    axes[1].legend(fontsize=5)
    _panel_label(axes[1], "b")
    swap_pairs = (
        causal.loc[causal["analysis"] == "coordinate_swap"]
        .groupby(
            ["participant", "task", "trial", "replicate", "subspace"],
            as_index=False,
        )["donor_logprob_shift"]
        .mean()
    )
    swap_participant_tasks = swap_pairs.groupby(
        ["participant", "task", "subspace"], as_index=False
    )["donor_logprob_shift"].mean()
    swap_participants = swap_participant_tasks.groupby(["participant", "subspace"], as_index=False)[
        "donor_logprob_shift"
    ].mean()
    swap = swap_participants.groupby("subspace")["donor_logprob_shift"].agg(
        ["mean", "sem", "count"]
    )
    swap_pairs.to_csv(SOURCE_DATA / "Figure5_swap_pairs.csv", index=False)
    swap_participant_tasks.to_csv(SOURCE_DATA / "Figure5_swap_participant_tasks.csv", index=False)
    swap_participants.to_csv(SOURCE_DATA / "Figure5_swap_participants.csv", index=False)
    swap.reset_index().to_csv(SOURCE_DATA / "Figure5_swap_summary.csv", index=False)
    axes[2].bar(
        swap.index,
        swap["mean"],
        yerr=swap["sem"],
        color=["#3F7CAC" if value == "behavioral" else "#B6B0A8" for value in swap.index],
    )
    axes[2].axhline(0, color="#777777", linewidth=0.8)
    axes[2].set(ylabel="Donor-response log-probability shift")
    axes[2].tick_params(axis="x", rotation=25)
    _panel_label(axes[2], "c")
    _save_figure(figure, "Figure5")
    plt.close(figure)


def make_figures(primary: dict[str, Any]) -> None:
    if not NEURAL_STATUS.exists():
        raise RuntimeError("Figure generation is sealed until the optional neural gate is frozen")
    _style()
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    for name in (
        "intrinsic_dimension",
        "task_subspace_angles",
        "shared_trajectories",
        "representation_sensitivities",
        "retained_information",
    ):
        pd.read_parquet(REQUIRED_TABLES[name]).to_csv(
            SOURCE_DATA / f"Supplementary_{name}.csv", index=False
        )
    figure_1()
    figure_2()
    figure_3(primary)
    figure_4()
    figure_5()


def _endpoint_sentence(name: str, values: dict[str, Any]) -> str:
    direction = "positive" if values["estimate"] > 0 else "non-positive"
    decision = "passes" if values["reject"] else "does not pass"
    return (
        f"- `{name}` is {direction}: mixed-model intercept {values['estimate']:.6f}; "
        f"complementary 95% two-way cluster CI "
        f"[{values['ci_95_low']:.6f}, {values['ci_95_high']:.6f}]; "
        f"Holm-adjusted p={values['holm_adjusted_p']:.6g}; {decision} the frozen alpha gate."
    )


def write_report(primary: dict[str, Any], inputs: dict[str, Path]) -> None:
    o11 = json.loads(inputs["centaur70b_status"].read_text())
    psych = json.loads((RESULTS / "psych201_mixed_effects.json").read_text())
    jlens_statistics = json.loads((RESULTS / "jlens_statistics.json").read_text())
    causal_statistics = json.loads((RESULTS / "causal_statistics.json").read_text())
    neural = json.loads(NEURAL_STATUS.read_text())
    baselines = json.loads((ROOT / "manifests/private/original/baselines.json").read_text())
    released_factor = baselines["released_factor_reconstruction"]
    eligibility = json.loads((RESULTS / "confirmation_eligibility.json").read_text())
    if not FALSIFICATION.exists():
        raise RuntimeError(
            "Original report requires the frozen eight-criterion falsification audit"
        )
    falsification = json.loads(FALSIFICATION.read_text())
    intrinsic = pd.read_parquet(RESULTS / "intrinsic_dimension.parquet")
    sensitivities = pd.read_parquet(RESULTS / "representation_sensitivities.parquet")
    retained = pd.read_parquet(RESULTS / "retained_information.parquet")
    endpoints = "\n".join(
        _endpoint_sentence(name, values) for name, values in primary["endpoints"].items()
    )
    primary_supported = all(value["reject"] for value in primary["endpoints"].values())
    strong_not_falsified = bool(falsification["strong_hypothesis_not_falsified"])
    if not primary_supported:
        conclusion = (
            "At least one frozen confirmatory endpoint does not pass; the strong "
            "original-profile shared-manifold hypothesis is not supported."
        )
    elif not strong_not_falsified:
        conclusion = (
            "All three confirmatory endpoints pass, but at least one of the eight frozen "
            "falsification criteria fails or is inconclusive; the strong original-profile "
            "shared-manifold hypothesis is therefore not fully supported."
        )
    elif o11["status"] != "completed":
        conclusion = (
            "All three confirmatory endpoints and all eight falsification criteria pass, but "
            "the exact 70B confirmation is resource-incomplete. The 8B/model-internal account "
            "is not falsified, while the full original-profile claim remains incomplete."
        )
    else:
        conclusion = (
            "All three confirmatory endpoints and all eight frozen falsification criteria pass, "
            "and exact 70B confirmation completed. This supports the original model-internal "
            "shared-manifold account, subject to the stated control limitations."
        )
    falsification_rows = "\n".join(
        f"- `{name}`: **{item['status']}**" for name, item in falsification["criteria"].items()
    )
    text = f"""# Original-profile results report

## Completion boundary

O1-O10 and O12 produced frozen original-profile artifacts. O11 terminal status is
`{o11["status"]}`. No strict-profile result is included here and no patient or clinical data were
used.

## Three confirmatory endpoints

{endpoints}

{conclusion}

The frozen confirmation-family gate retained
{eligibility["eligible_confirmation_participants"]} of
{eligibility["total_confirmation_participants"]} retest participants; eligibility required at
least {eligibility["minimum_source_families"]} source families and
{eligibility["minimum_distinct_target_families"]} distinct target family.

## Eight-criterion falsification audit

{falsification_rows}

Counts: {falsification["counts"]}. These secondary decisions do not enter the Holm family for the
three confirmatory endpoints. Full numerical evidence is frozen in
`results/original/falsification_audit.json` and `results/original/falsification_audit.parquet`.

## External generalization

The frozen Psych-201 mixed model gives standardized alignment coefficient
{psych["primary_beta_1"]:.6f} (p={psych["primary_beta_1_p"]:.6g}). Leave-one-study-out prediction
has R²={psych["loso_r2"]:.4f}, Spearman rho={psych["loso_spearman"]:.4f}, and calibration slope
{psych["loso_calibration_slope"]:.4f}. These are experiment-level external-generalization results;
they do not test stable identity across studies.

## Classical SRO-factor baseline boundary

The frozen SRO repository audit status is `{released_factor["status"]}`. The repository contains
the released factor-analysis source code but no saved fitted task-factor object. Consequently the
reported `sro_factor` comparison is the preregistered development-only five-factor refit, not an
exact recovered released factor solution. This boundary is recorded in the private baseline
manifest and is not treated as evidence that the historical solution was reproduced.

## Causal-language boundary

J-Lens is reported only as downstream causal accessibility. Projection ablation and coordinate
swap establish effects inside the frozen model under the original Haar-random controls. They do
not establish a human neural mechanism, and the original random-subspace design is not equivalent
to the stricter prediction-matched, rescue and leave-task-out controls planned for the separate
strict profile.

The J-Lens table contains {len(jlens_statistics["tests"])} frozen empirical/permutation tests;
the finite-intervention table contains {len(causal_statistics["tests"])} participant-cluster
permutation tests, each using {causal_statistics["permutation_replicates"]:,} replicates.

## Geometry and sensitivity analyses

Independent geometry reports {len(intrinsic)} intrinsic-dimension estimates. The frozen
representation-sensitivity table contains {len(sensitivities)} preprocessing conditions, and the
retained-information table contains {len(retained)} task-fold cells. Full task-specific principal
angles and participant trajectory summaries are included in the frozen result tables and exported
as supplementary source data; they are descriptive or sensitivity analyses rather than additional
confirmatory endpoints.

## 70B boundary

{o11["reason"]} Quantized output was not substituted for the original exact-BF16 requirement.
Therefore a resource-incomplete O11 is not interpreted as a negative 70B scientific result.

## Optional neural-validation boundary

The frozen neural-validation status is `{neural["status"]}`: {neural["reason"]}

## Reproducibility

- Configuration SHA-256: `{_sha256(CONFIG)}`
- Git commit used for this report: `{_git("rev-parse", "HEAD")}`
- Primary statistics: `results/original/primary_statistics.json`
- Figure source tables: `reports/original/figure_source_data/`
- Figures: editable SVG, PDF and 600-dpi TIFF in `reports/original/figures/`
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)


def freeze(inputs: dict[str, Path]) -> None:
    dirty = _git("status", "--porcelain")
    if dirty:
        raise RuntimeError(
            "Final original-profile freeze requires a clean Git worktree; commit the "
            "reviewed code and figure-QA record first"
        )
    files = sorted(
        path
        for root in (RESULTS, REPORTS, ROOT / "manifests/private/original")
        for path in root.rglob("*")
        if path.is_file() and path != FREEZE and path != MANIFEST
    )
    payload = {
        "schema_version": 1,
        "profile": "original",
        "status": "frozen",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": False,
        "config_sha256": _sha256(CONFIG),
        "input_tables": {name: _sha256(path) for name, path in inputs.items()},
        "files": {str(path.relative_to(ROOT)): _sha256(path) for path in files},
    }
    FREEZE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("statistics", "figures", "report", "all"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = _require_inputs()
    config = yaml.safe_load(CONFIG.read_text())
    primary = (
        run_primary_statistics(config)
        if args.action in {"statistics", "all"}
        else json.loads(STATISTICS.read_text())
    )
    if args.action in {"statistics", "all"}:
        run_falsification_audit(primary)
    elif not FALSIFICATION.exists():
        raise RuntimeError("Run the O12 statistics action to freeze falsification decisions")
    if args.action in {"figures", "all"}:
        make_figures(primary)
    if args.action in {"report", "all"}:
        write_report(primary, inputs)
        freeze(inputs)
    print(json.dumps({"action": args.action, "status": "completed"}))


if __name__ == "__main__":
    main()
