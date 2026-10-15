from __future__ import annotations

from pathlib import Path
import importlib.util

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "original" / "main.yaml"


def load_jlens_script():
    path = ROOT / "scripts" / "original" / "09_fit_jlens.py"
    spec = importlib.util.spec_from_file_location("original_jlens", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_original() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_original_profile_is_explicitly_not_strict() -> None:
    config = load_original()
    assert config["analysis_profile"] == "original"
    assert config["scientific_source"]["rule"] == "reproduce_source_before_strict_revision"
    assert config["paths"]["results"] == "results/original"
    assert config["paths"]["reports"] == "reports/original"
    assert "strict" not in str(config["paths"])


def test_original_execution_is_runnable_after_source_ambiguities_are_frozen() -> None:
    config = load_original()
    assert config["execution"]["runnable"] is True
    assert config["execution"]["blocked_by_decisions"] == []


def test_original_order_matches_source_proposal() -> None:
    config = load_original()
    steps = config["execution"]["ordered_steps"]
    assert steps[0:3] == ["00_download_data", "01_audit_tasks", "02_build_sro_sessions"]
    assert steps[-4:] == [
        "09_fit_jlens",
        "10_run_interventions",
        "11_run_centaur70b",
        "12_make_results",
    ]
    assert steps.index("freeze_shared_layer_and_dimension") < steps.index(
        "05_fit_individuality_on_development_participants_only"
    )
    assert steps.index("freeze_private_rank_and_regularization") < steps.index(
        "06_confirm_retest_on_untouched_participants"
    )


def test_original_numeric_defaults_match_source() -> None:
    config = load_original()
    assert config["execution"]["seed"] == 20260821
    assert config["likelihood_and_representation"]["pca"]["primary_dim"] == 512
    assert config["shared_manifold"]["dimensions"] == [8, 16, 32, 64, 128, 256]
    assert config["individuality"]["private_ranks"] == [1, 2, 4, 8, 16, 32]
    assert config["jlens"]["psych101_prompts"] == 1000
    assert config["jlens"]["random_subspaces_for_null"] == 1000
    assert config["intervention"]["strengths"] == [0.25, 0.5, 1.0]
    assert config["statistics"]["bootstrap_replicates"] == 10000
    assert config["statistics"]["permutation_replicates"] == 10000


def test_jlens_prompt_quota_redistributes_capacity_shortfalls() -> None:
    jlens = load_jlens_script()
    capacities = {"small": 7, **{f"task_{index:02d}": 100 for index in range(75)}}
    allocation = jlens._capacity_balanced_quotas(capacities, 1000)
    assert sum(allocation.values()) == 1000
    assert allocation["small"] == 7
    unrestricted = [value for key, value in allocation.items() if key != "small"]
    assert min(unrestricted) == 13
    assert max(unrestricted) == 14


def test_original_prompt_keeps_source_history_rule() -> None:
    config = load_original()
    prompt = config["sro_prompt"]
    assert prompt["include_all_previous_events_available_to_participant"] is True
    assert prompt["current_response_in_prefix"] is False
    assert prompt["acceptance"]["manual_complete_sessions_per_task"] == 50


def test_neural_payload_remains_conditional() -> None:
    config = load_original()
    assert config["data"]["ds004636"]["role"] == "optional_neural_validation"
    assert config["data"]["ds004636"]["payload_download_gate"] == "behavioral_hypotheses_confirmed"
