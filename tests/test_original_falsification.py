from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def _module() -> object:
    path = Path(__file__).parents[1] / "scripts/original/12_make_results.py"
    spec = importlib.util.spec_from_file_location("original_results_falsification_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_eight_criterion_falsification_audit(tmp_path: Path) -> None:
    module = _module()
    results = tmp_path / "results"
    results.mkdir()
    controls = pd.DataFrame(
        [
            {
                "control": "shared_behavioral",
                "dimension": 16,
                "task": "t",
                "fold": 0,
                "validation_nll": 0.60,
            },
            {
                "control": "task_specific",
                "dimension": 16,
                "task": "t",
                "fold": 0,
                "validation_nll": 0.58,
            },
            {
                "control": "full_representation",
                "dimension": 512,
                "task": "t",
                "fold": 0,
                "validation_nll": 0.50,
            },
            {
                "control": "haar_random",
                "dimension": 16,
                "task": "t",
                "fold": 0,
                "validation_nll": 1.00,
            },
        ]
    )
    controls_path = results / "shared_controls.parquet"
    controls.to_parquet(controls_path, index=False)
    (results / "selected_shared.json").write_text(json.dumps({"dimension": 16}))
    (results / "selected_private.json").write_text(json.dumps({"rank": 4}))
    (results / "coordinate_recovery_summary.json").write_text(
        json.dumps(
            {
                "independent_global_procrustes_correlation": 0.4,
                "independent_global_procrustes_permutation_p": 0.01,
                "independent_top1_retrieval": 0.2,
                "independent_top1_chance": 0.01,
                "independent_top1_permutation_p": 0.01,
            }
        )
    )
    (results / "psych201_mixed_effects.json").write_text(
        json.dumps({"primary_beta_1": 0.3, "primary_beta_1_p": 0.01})
    )
    (results / "causal_statistics.json").write_text(
        json.dumps(
            {
                "tests": {
                    "projection_ablation_strength_1.00": {
                        "estimate": 0.2,
                        "one_sided_participant_permutation_p": 0.01,
                    }
                }
            }
        )
    )
    frozen = tmp_path / "falsification.yaml"
    frozen.write_text(
        """alpha: 0.05
shared_subspace:
  substantially_worse_fraction_of_full_over_random_advantage: 0.20
private_rank:
  approaching_shared_dimension_fraction: 0.80
causal_ablation:
  strength: 1.00
"""
    )
    module.RESULTS = results
    module.REQUIRED_TABLES = {"shared_controls": controls_path}
    module.FALSIFICATION_CONFIG = frozen
    module.FALSIFICATION = results / "falsification_audit.json"
    module.FALSIFICATION_TABLE = results / "falsification_audit.parquet"
    primary = {
        "endpoints": {
            "cross_domain": {"reject": True, "estimate": 0.2},
            "cross_time": {"reject": True, "estimate": 0.2},
            "manifold_minus_factor": {"reject": True, "estimate": 0.1},
        }
    }
    audit = module.run_falsification_audit(primary)
    assert audit["strong_hypothesis_not_falsified"]
    assert audit["counts"] == {"not_falsified": 8, "falsified": 0, "inconclusive": 0}

    primary["endpoints"]["cross_time"]["reject"] = False
    audit = module.run_falsification_audit(primary)
    assert not audit["strong_hypothesis_not_falsified"]
    assert (
        audit["criteria"]["t1_coordinates_predict_t2_different_domain_behavior"]["status"]
        == "falsified"
    )
