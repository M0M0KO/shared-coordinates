#!/usr/bin/env python3
"""Resume the frozen original O3-O12 pipeline and stop before final figure QA/freeze."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
O8_PYTHON = ROOT / ".venv-o8/bin/python"
LOG_ROOT = ROOT / "logs/original_pipeline"
STATE = LOG_ROOT / "state.json"
NEURAL_STATUS = ROOT / "results/original/neural_validation_status.json"
REMAINING_VALIDATOR = ROOT / "scripts/original/00_validate_remaining_pipeline.py"
STATE_LOCK = threading.RLock()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state() -> dict[str, Any]:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"schema_version": 1, "profile": "original", "steps": {}}


def _write_state(payload: dict[str, Any]) -> None:
    with STATE_LOCK:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = STATE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(STATE)


def _step_complete(state: dict[str, Any], name: str) -> bool:
    return state.get("steps", {}).get(name, {}).get("status") == "completed"


def _validate_step_contract(name: str, log: Any) -> int:
    if name == "preflight_o5_o12":
        return 0
    command = [str(PYTHON), str(REMAINING_VALIDATOR), "--validate-step", name]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode


def run_step(
    state: dict[str, Any],
    name: str,
    arguments: list[str],
    *,
    interpreter: Path = PYTHON,
) -> None:
    log_path = LOG_ROOT / f"{name}.log"
    if _step_complete(state, name):
        with log_path.open("a", encoding="utf-8") as log:
            if _validate_step_contract(name, log) != 0:
                raise RuntimeError(
                    f"Completed step has an invalid artifact contract: {name}; "
                    f"inspect {log_path}"
                )
        return
    command = [str(interpreter), *arguments]
    with STATE_LOCK:
        state.setdefault("steps", {})[name] = {
            "status": "running",
            "started_at": _timestamp(),
            "command": command,
            "log": str(log_path.relative_to(ROOT)),
        }
        _write_state(state)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    threads = environment.get("CM_CPU_THREADS_PER_JOB", "128")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[variable] = threads
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_timestamp()}] START {json.dumps(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        returncode = completed.returncode
        if returncode == 0:
            returncode = _validate_step_contract(name, log)
        log.write(f"[{_timestamp()}] EXIT {returncode}\n")
    status = "completed" if returncode == 0 else "failed"
    with STATE_LOCK:
        state["steps"][name].update(
            {"status": status, "finished_at": _timestamp(), "returncode": returncode}
        )
        _write_state(state)
    if returncode:
        raise RuntimeError(f"Original pipeline step failed: {name}; inspect {log_path}")


def run_parallel_steps(
    state: dict[str, Any],
    steps: list[tuple[str, list[str]]],
    *,
    workers: int,
) -> None:
    pending = [(name, arguments) for name, arguments in steps if not _step_complete(state, name)]
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
        futures = {
            executor.submit(run_step, state, name, arguments): name for name, arguments in pending
        }
        for future in as_completed(futures):
            future.result()


def wait_for_exit(state: dict[str, Any], name: str, exit_path: Path) -> None:
    if _step_complete(state, name):
        return
    state.setdefault("steps", {})[name] = {
        "status": "waiting",
        "started_at": _timestamp(),
        "exit_file": str(exit_path.relative_to(ROOT)),
    }
    _write_state(state)
    while not exit_path.exists():
        time.sleep(60)
    code = int(exit_path.read_text().strip())
    state["steps"][name].update(
        {
            "status": "completed" if code == 0 else "failed",
            "finished_at": _timestamp(),
            "returncode": code,
        }
    )
    _write_state(state)
    if code:
        raise RuntimeError(f"Background prerequisite failed: {name}; exit code {code}")


def neural_gate(state: dict[str, Any]) -> bool:
    statistics = json.loads((ROOT / "results/original/primary_statistics.json").read_text())
    supported = all(value["reject"] for value in statistics["endpoints"].values())
    if not supported:
        payload = {
            "schema_version": 1,
            "profile": "original",
            "status": "not_triggered",
            "reason": "At least one frozen behavioral confirmatory endpoint did not pass.",
        }
        NEURAL_STATUS.parent.mkdir(parents=True, exist_ok=True)
        NEURAL_STATUS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        state["steps"]["optional_neural_gate"] = {
            "status": "completed",
            "finished_at": _timestamp(),
            **payload,
        }
        _write_state(state)
        return True
    if NEURAL_STATUS.exists():
        payload = json.loads(NEURAL_STATUS.read_text())
        if payload.get("status") == "completed":
            state["steps"]["optional_neural_gate"] = {
                "status": "completed",
                "finished_at": _timestamp(),
                "neural_status": "completed",
            }
            _write_state(state)
            return True
    state["status"] = "awaiting_triggered_optional_neural_validation"
    state["steps"]["optional_neural_gate"] = {
        "status": "waiting",
        "started_at": _timestamp(),
        "reason": "All behavioral endpoints passed; ds004636 validation is now triggered.",
    }
    _write_state(state)
    return False


def pipeline() -> None:
    state = _load_state()
    wait_for_exit(state, "wait_o3_minitaur", ROOT / "logs/original_o3_minitaur.exit")
    wait_for_exit(state, "wait_o3_base", ROOT / "logs/original_o3_base.exit")
    run_step(
        state,
        "o3_reconcile_discovery",
        [
            "scripts/original/03_reconcile_8b_manifest.py",
            "--models",
            "minitaur_8b",
            "llama_base_8b",
        ],
    )

    shared_runs = [
        ("o4_shared_minitaur_layer24", ["--model", "minitaur_8b", "--layer", "24"]),
        ("o4_shared_minitaur_layer08", ["--model", "minitaur_8b", "--layer", "8"]),
        ("o4_shared_minitaur_layer16", ["--model", "minitaur_8b", "--layer", "16"]),
        ("o4_shared_minitaur_layer32", ["--model", "minitaur_8b", "--layer", "32"]),
        ("o4_shared_base_layer24", ["--model", "llama_base_8b", "--layer", "24"]),
        ("o4_sensitivity_q256", ["--pca-dim", "256"]),
        ("o4_sensitivity_q1024", ["--pca-dim", "1024"]),
        ("o4_sensitivity_remove_pc1", ["--remove-pcs", "1"]),
        ("o4_sensitivity_remove_pc3", ["--remove-pcs", "3"]),
        ("o4_sensitivity_remove_pc5", ["--remove-pcs", "5"]),
        ("o4_sensitivity_unit_length", ["--unit-normalize"]),
    ]
    o4_workers = max(1, int(os.environ.get("CM_O4_WORKERS", "2")))
    run_parallel_steps(
        state,
        [
            (name, ["scripts/original/04_fit_shared_manifold.py", *extra])
            for name, extra in shared_runs
        ],
        workers=o4_workers,
    )
    run_parallel_steps(
        state,
        [
            (
                f"o4_control_{action}",
                ["scripts/original/04_fit_shared_controls.py", action],
            )
            for action in ("full", "task-specific", "haar", "held-out")
        ],
        workers=o4_workers,
    )

    run_step(
        state,
        "preflight_o5_o12",
        ["scripts/original/00_validate_remaining_pipeline.py"],
    )
    run_step(state, "o5_individuality", ["scripts/original/05_fit_individuality.py"])
    for action in ("build", "extract", "analyze", "persona"):
        run_step(state, f"o6_{action}", ["scripts/original/06_confirm_retest.py", action])
    run_step(
        state,
        "o7_extract",
        ["scripts/original/07_run_psych201.py", "extract", "--model", "all", "--shard", "all"],
    )
    run_step(state, "o7_analyze", ["scripts/original/07_run_psych201.py", "analyze"])

    wait_for_exit(state, "wait_o8_environment", ROOT / "logs/original_o8_env.exit")
    wait_for_exit(state, "wait_o8_prepare", ROOT / "logs/original_o8_prepare.exit")
    run_step(
        state,
        "o8_training_preflight",
        ["scripts/original/08_train_shuffled_control.py", "preflight"],
        interpreter=O8_PYTHON,
    )
    run_step(
        state,
        "o8_train_intact",
        ["scripts/original/08_train_shuffled_control.py", "train", "--kind", "intact"],
        interpreter=O8_PYTHON,
    )
    run_step(
        state,
        "o8_train_shuffled",
        ["scripts/original/08_train_shuffled_control.py", "train", "--kind", "shuffled"],
        interpreter=O8_PYTHON,
    )
    for model in ("reproduced_intact_psych101_8b", "response_shuffled_psych101_8b"):
        run_step(
            state,
            f"o8_extract_{model}",
            ["scripts/original/03_extract_8b_states.py", "--model", model, "--task", "all"],
        )
        run_step(
            state,
            f"o8_reconcile_{model}",
            ["scripts/original/03_reconcile_8b_manifest.py", "--models", model],
        )
        run_step(
            state,
            f"o8_shared_{model}",
            ["scripts/original/04_fit_shared_manifold.py", "--model", model, "--layer", "24"],
        )
    run_step(
        state,
        "o4_controls_finalize",
        ["scripts/original/04_fit_shared_controls.py", "finalize"],
    )
    run_step(state, "o4_geometry", ["scripts/original/04_characterize_geometry.py", "all"])

    run_step(state, "o9_freeze_prompts", ["scripts/original/09_fit_jlens.py", "freeze-prompts"])
    run_step(
        state,
        "o9_fit",
        ["scripts/original/09_fit_jlens.py", "fit", "--model", "all"],
    )
    run_step(state, "o9_analyze", ["scripts/original/09_fit_jlens.py", "analyze"])
    run_step(
        state,
        "o10_run",
        ["scripts/original/10_run_interventions.py", "run", "--task", "all"],
    )
    run_step(state, "o10_finalize", ["scripts/original/10_run_interventions.py", "finalize"])

    wait_for_exit(state, "wait_70b_promotion", ROOT / "logs/original_70b_promote.exit")
    for action in ("freeze-prompts", "preflight", "benchmark"):
        run_step(state, f"o11_{action}", ["scripts/original/11_run_centaur70b.py", action])
    run_step(state, "o12_statistics", ["scripts/original/12_make_results.py", "statistics"])
    if not neural_gate(state):
        return
    run_step(state, "o12_figures", ["scripts/original/12_make_results.py", "figures"])
    state["status"] = "awaiting_figure_qa_and_final_freeze"
    state["updated_at"] = _timestamp()
    _write_state(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "status"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "status":
        print(json.dumps(_load_state(), indent=2, sort_keys=True))
    else:
        pipeline()


if __name__ == "__main__":
    main()
