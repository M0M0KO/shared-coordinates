from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/original/00_validate_remaining_pipeline.py"
RUNNER = ROOT / "scripts/original/00_run_original_pipeline.py"


def _module():
    spec = importlib.util.spec_from_file_location("original_remaining_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_o5_contract_rejects_missing_outputs(tmp_path: Path) -> None:
    module = _module()
    result = module.validate_step("o5_individuality", tmp_path)
    assert result["status"] == "failed"
    assert result["failures"]


def test_contract_inspects_json_parquet_numpy_and_model(tmp_path: Path) -> None:
    module = _module()
    files = {
        "results/original/selected_private.json": json.dumps({"rank": 2}),
        "results/original/coordinate_recovery_development.parquet": None,
        "results/original/private_rank.parquet": None,
        "results/original/cross_domain.parquet": None,
        "cache/original/individuality_final/model.pt": "model",
        "manifests/private/original/baselines.json": json.dumps({"status": "complete"}),
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
        else:
            path.write_text(str(content))
    result = module.validate_step("o5_individuality", tmp_path)
    assert result == {
        "step": "o5_individuality",
        "contracts": 6,
        "status": "passed",
        "failures": [],
    }


def test_dynamic_o8_extraction_contract_requires_all_tasks(tmp_path: Path) -> None:
    module = _module()
    contracts = module.contracts_for_step("o8_extract_response_shuffled_psych101_8b")
    assert contracts[0]["minimum"] == 16
    assert "response_shuffled_psych101_8b" in contracts[0]["glob"]


def test_runner_places_preflight_before_o5_and_validates_successful_steps() -> None:
    source = RUNNER.read_text()
    assert source.index('"preflight_o5_o12"') < source.index('"o5_individuality"')
    assert "_validate_step_contract(name, log)" in source
