from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_stage0_sources_are_immutable_and_public_only() -> None:
    config = yaml.safe_load((ROOT / "configs" / "stage0.yaml").read_text(encoding="utf-8"))
    downloads = config["downloads"]
    assert downloads["public_data_only"] is True
    for name in ("sro", "llama_base_8b", "minitaur_8b"):
        spec = downloads[name]
        revisions = [value for key, value in spec.items() if key.endswith("revision")]
        assert revisions
        assert all(len(revision) == 40 for revision in revisions)
        assert all(revision not in {"main", "master", "latest"} for revision in revisions)


def test_strict_cold_start_is_primary() -> None:
    config = yaml.safe_load((ROOT / "configs" / "stage0.yaml").read_text(encoding="utf-8"))
    assert config["stage0"]["primary_history_mode"] == "strict_cold_start"
    assert config["stage0"]["strict_target_history"] == "response_masked"
    assert "warm_start" in config["stage0"]["secondary_history_modes"]


def test_model_hash_contract_has_four_shards_per_model() -> None:
    manifest = json.loads(
        (ROOT / "manifests" / "public" / "source_revisions.json").read_text(encoding="utf-8")
    )
    for name in ("llama_base_8b", "minitaur_8b"):
        files = manifest["expected_model_files"][name]
        assert len(files) == 4
        assert all(len(contract["sha256"]) == 64 for contract in files.values())
        assert sum(contract["bytes"] for contract in files.values()) == 16060556376


def test_modelscope_is_used_without_substituting_minitaur() -> None:
    config = yaml.safe_load((ROOT / "configs" / "stage0.yaml").read_text(encoding="utf-8"))
    assert config["downloads"]["llama_base_8b"]["provider"] == "modelscope"
    minitaur = config["downloads"]["minitaur_8b"]
    assert minitaur["official_id"] == "marcelbinz/Llama-3.1-Minitaur-8B"
    assert minitaur["provider_reason"].startswith("no_verified_modelscope_repository")
