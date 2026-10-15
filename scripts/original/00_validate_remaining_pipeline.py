#!/usr/bin/env python3
"""Preflight O5-O12 and validate critical outputs before downstream use."""

from __future__ import annotations

import argparse
import json
import py_compile
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "logs/original_pipeline/o5_o12_preflight.json"
MAIN_PYTHON = ROOT / ".venv/bin/python"
O8_PYTHON = ROOT / ".venv-o8/bin/python"

PIPELINE_SCRIPTS = tuple(
    ROOT / f"scripts/original/{name}"
    for name in (
        "05_fit_individuality.py",
        "06_confirm_retest.py",
        "07_run_psych201.py",
        "08_train_shuffled_control.py",
        "09_fit_jlens.py",
        "10_run_interventions.py",
        "11_run_centaur70b.py",
        "12_make_results.py",
    )
)


def _file(path: str) -> dict[str, Any]:
    return {"path": path}


def _glob(pattern: str, minimum: int) -> dict[str, Any]:
    return {"glob": pattern, "minimum": minimum}


STATIC_CONTRACTS: dict[str, list[dict[str, Any]]] = {
    "o5_individuality": [
        _file("results/original/selected_private.json"),
        _file("results/original/private_rank.parquet"),
        _file("results/original/cross_domain.parquet"),
        _file("cache/original/individuality_final/model.pt"),
        _file("results/original/coordinate_recovery_development.parquet"),
        _file("manifests/private/original/baselines.json"),
    ],
    "o6_build": [
        _glob("data/derived/original/sro_sessions/confirmation_t1/*.parquet", 16),
        _glob("data/derived/original/sro_sessions/confirmation_t2/*.parquet", 16),
    ],
    "o6_extract": [
        _glob("cache/original/confirmation_states/T1/*/metadata.parquet", 16),
        _glob("cache/original/confirmation_states/T2/*/metadata.parquet", 16),
        _glob("cache/original/confirmation_states/T1/*/layer_*.npy", 16),
        _glob("cache/original/confirmation_states/T2/*/layer_*.npy", 16),
    ],
    "o6_analyze": [
        _file("results/original/retest_transfer.parquet"),
        _file("manifests/private/original/retest_coordinates.npz"),
        _file("results/original/coordinate_recovery_confirmation.parquet"),
        _file("results/original/coordinate_recovery_summary.json"),
        _file("results/original/confirmation_eligibility.json"),
    ],
    "o6_persona": [_file("results/original/persona_comparison.parquet")],
    "o7_extract": [
        _glob("cache/original/psych201/*/shard_*/metadata.parquet", 8),
        _glob("cache/original/psych201/*/shard_*/layer_*.npy", 8),
        _file("manifests/private/original/psych201.json"),
    ],
    "o7_analyze": [
        _file("results/original/psych201_generalization.parquet"),
        _file("results/original/psych201_mixed_effects.json"),
    ],
    "o8_training_preflight": [_file("manifests/private/original/o8_training_environment.json")],
    "o8_train_intact": [_glob("models/Psych101-Reproduced-8B/*.safetensors", 1)],
    "o8_train_shuffled": [_glob("models/Psych101-Response-Shuffled-8B/*.safetensors", 1)],
    "o4_controls_finalize": [_file("results/original/shared_controls.parquet")],
    "o4_geometry": [
        _file("results/original/intrinsic_dimension.parquet"),
        _file("results/original/task_subspace_angles.parquet"),
        _file("results/original/shared_trajectories.parquet"),
        _file("results/original/representation_sensitivities.parquet"),
        _file("results/original/retained_information.parquet"),
    ],
    "o9_freeze_prompts": [_file("manifests/private/original/jlens_prompts.jsonl")],
    "o9_fit": [
        _glob("cache/original/jlens/*/lens.pt", 3),
        _glob("cache/original/jlens/*/checkpoint.pt", 3),
    ],
    "o9_analyze": [
        _file("results/original/jlens_access.parquet"),
        _file("results/original/jlens_statistics.json"),
        _file("results/original/jlens_permutation_nulls.npz"),
    ],
    "o10_run": [_glob("cache/original/interventions/*.parquet", 16)],
    "o10_finalize": [
        _file("results/original/causal_interventions.parquet"),
        _file("results/original/causal_contrasts.parquet"),
        _file("results/original/causal_statistics.json"),
        _file("results/original/causal_permutation_nulls.npz"),
    ],
    "o11_freeze-prompts": [_file("manifests/private/original/centaur70b_benchmark_prompts.jsonl")],
    "o11_benchmark": [
        _file("reports/original/centaur70b_exact_bf16_benchmark.json"),
        _file("results/original/centaur70b_status.json"),
        _file("manifests/private/original/centaur70b.json"),
    ],
    "o12_statistics": [
        _file("results/original/primary_statistics.json"),
        _file("results/original/primary_statistics.parquet"),
        _file("results/original/falsification_audit.json"),
        _file("results/original/falsification_audit.parquet"),
    ],
    "o12_figures": [
        _glob("reports/original/figures/Figure*.svg", 5),
        _glob("reports/original/figures/Figure*.pdf", 5),
        _glob("reports/original/figures/Figure*.tiff", 5),
        _glob("reports/original/figure_source_data/Figure*.csv", 5),
    ],
}


def contracts_for_step(name: str) -> list[dict[str, Any]]:
    if name.startswith("o8_extract_") or name.startswith("o8_reconcile_"):
        model = name.split("_", 2)[2]
        return [
            _glob(f"cache/original/8b_states/{model}/*/metadata.parquet", 16),
            _file("manifests/private/original/8b_state_extraction.json"),
        ]
    if name.startswith("o8_shared_"):
        model = name.removeprefix("o8_shared_")
        suffix = f"{model}_layer_24"
        return [
            _file(f"results/original/selected_shared_{suffix}.json"),
            _file(f"results/original/shared_manifold_{suffix}.parquet"),
        ]
    return STATIC_CONTRACTS.get(name, [])


def _inspect_file(path: Path, root: Path = ROOT) -> str | None:
    if not path.is_file() or path.stat().st_size == 0:
        return f"missing_or_empty:{path.relative_to(root)}"
    try:
        if path.suffix == ".json":
            json.loads(path.read_text())
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq

            metadata = pq.ParquetFile(path).metadata
            if metadata is None or metadata.num_rows == 0:
                return f"empty_parquet:{path.relative_to(root)}"
        elif path.suffix == ".npy":
            import numpy as np

            value = np.load(path, mmap_mode="r", allow_pickle=False)
            if value.size == 0:
                return f"empty_npy:{path.relative_to(root)}"
        elif path.suffix == ".npz":
            import numpy as np

            with np.load(path, allow_pickle=False) as value:
                if not value.files:
                    return f"empty_npz:{path.relative_to(root)}"
    except Exception as error:
        return f"unreadable:{path.relative_to(root)}:{type(error).__name__}:{error}"
    return None


def validate_step(name: str, root: Path = ROOT) -> dict[str, Any]:
    failures: list[str] = []
    contracts = contracts_for_step(name)
    for contract in contracts:
        if "path" in contract:
            paths = [root / contract["path"]]
        else:
            paths = sorted(root.glob(contract["glob"]))
            minimum = int(contract["minimum"])
            if len(paths) < minimum:
                failures.append(
                    f"glob_count:{contract['glob']}:expected>={minimum}:observed={len(paths)}"
                )
        for path in paths:
            failure = _inspect_file(path, root)
            if failure:
                failures.append(failure)
    return {
        "step": name,
        "contracts": len(contracts),
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }


def _run_import_gate(interpreter: Path, imports: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(interpreter), "-c", imports],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return {
        "interpreter": str(interpreter.relative_to(ROOT)),
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-1000:],
    }


def run_preflight() -> dict[str, Any]:
    import torch
    import yaml

    errors: list[str] = []
    warnings: list[str] = []
    for script in PIPELINE_SCRIPTS:
        try:
            py_compile.compile(str(script), doraise=True)
        except Exception as error:
            errors.append(f"compile:{script.relative_to(ROOT)}:{error}")
    config = yaml.safe_load((ROOT / "configs/original/main.yaml").read_text())
    if config["analysis_profile"] != "original" or not config["execution"]["runnable"]:
        errors.append("original_config_is_not_runnable")

    required = [
        "results/original/selected_shared.json",
        "data/raw/Psych-101/prompts_training.jsonl",
        "data/raw/Psych-201-discrete/data",
        "data/raw/jacobian-lens-code",
        "data/derived/original/psych101_shuffled/prompts_training.jsonl",
        "models/Llama-3.1-8B/config.json",
        "models/Llama-3.1-Minitaur-8B/config.json",
        "models/Llama-3.1-70B/model.safetensors.index.json",
        "models/Llama-3.1-Centaur-70B-adapter/adapter_model.safetensors",
    ]
    for relative in required:
        if not (ROOT / relative).exists():
            errors.append(f"missing_static_input:{relative}")

    exit_gates = (
        "original_o8_env.exit",
        "original_o8_prepare.exit",
        "original_70b_promote.exit",
    )
    for exit_name in exit_gates:
        path = ROOT / "logs" / exit_name
        if not path.is_file() or path.read_text().strip() != "0":
            errors.append(f"failed_or_missing_exit_gate:logs/{exit_name}")

    shards = sorted((ROOT / "models/Llama-3.1-70B").glob("model-*.safetensors"))
    if len(shards) != 30:
        errors.append(f"70b_shard_count:expected=30:observed={len(shards)}")
    partials = sorted((ROOT / "models").rglob("*.part")) + sorted(
        (ROOT / "models").rglob("*.incomplete")
    )
    if partials:
        errors.extend(f"partial_model_artifact:{path.relative_to(ROOT)}" for path in partials)

    disk = shutil.disk_usage(ROOT)
    if disk.free < 500 * 1024**3:
        errors.append(f"free_disk_below_500GiB:{disk.free}")
    if not torch.cuda.is_available():
        errors.append("cuda_unavailable")
        gpu: dict[str, Any] = {"available": False}
    else:
        properties = torch.cuda.get_device_properties(0)
        gpu = {"available": True, "name": properties.name, "bytes": properties.total_memory}
        if properties.total_memory < 70 * 1024**3:
            errors.append(f"gpu_memory_below_70GiB:{properties.total_memory}")

    imports = [
        _run_import_gate(
            MAIN_PYTHON,
            "import numpy,pandas,pyarrow,sklearn,torch,transformers,peft,yaml",
        ),
        _run_import_gate(
            O8_PYTHON,
            "import unsloth; import bitsandbytes,datasets,torch,transformers,trl",
        ),
    ]
    for item in imports:
        if item["status"] != "passed":
            errors.append(f"environment_import_gate:{item['interpreter']}")

    pending = {
        "o6": not (ROOT / "results/original/selected_private.json").exists(),
        "o7": not (ROOT / "results/original/retest_transfer.parquet").exists(),
        "o9": not (ROOT / "models/Psych101-Response-Shuffled-8B").exists(),
        "o10": not (ROOT / "results/original/jlens_access.parquet").exists(),
        "o12": not (ROOT / "results/original/causal_interventions.parquet").exists(),
    }
    if any(pending.values()):
        warnings.append("Some downstream stages are correctly waiting for upstream outputs.")
    payload = {
        "schema_version": 1,
        "profile": "original",
        "scope": "O5-O12",
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "pending_upstream": pending,
        "gpu": gpu,
        "free_disk_bytes": disk.free,
        "environment_imports": imports,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(REPORT)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_step(args.validate_step) if args.validate_step else run_preflight()
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
