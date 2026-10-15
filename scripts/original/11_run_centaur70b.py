#!/usr/bin/env python3
"""Assess and benchmark the exact-BF16 original Centaur-70B confirmation route."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/original/centaur70b_frozen.yaml"
MAIN_CONFIG = ROOT / "configs/original/main.yaml"
SELECTION = ROOT / "results/original/selected_private.json"
BASE_MODEL = ROOT / "models/Llama-3.1-70B"
ADAPTER = ROOT / "models/Llama-3.1-Centaur-70B-adapter"
SESSION_ROOT = ROOT / "data/derived/original/sro_sessions/development"
PROMPTS = ROOT / "manifests/private/original/centaur70b_benchmark_prompts.jsonl"
BENCHMARK = ROOT / "reports/original/centaur70b_exact_bf16_benchmark.json"
STATUS = ROOT / "results/original/centaur70b_status.json"
MANIFEST = ROOT / "manifests/private/original/centaur70b.json"


def _effective_gpu_max_memory(frozen: dict[str, Any]) -> str:
    """Honor the runtime reserve without changing weights or precision."""
    configured = str(frozen["maximum_memory"]["gpu_0"])
    reserve_gib = float(os.environ.get("CM_GPU_RESERVE_GIB", "0"))
    if reserve_gib <= 0:
        return configured
    match = re.fullmatch(r"(\d+(?:\.\d+)?)GiB", configured)
    if match is None:
        raise RuntimeError(f"Unsupported GPU memory limit: {configured}")
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    usable_gib = max(1, int(total_gib - reserve_gib))
    configured_gib = float(match.group(1))
    return f"{min(int(configured_gib), usable_gib)}GiB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    """Convert pandas/NumPy scalar values without changing their logical type."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _weight_inventory(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No safetensor weights found in {root}")
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "names": [path.name for path in files],
    }


def freeze_prompts() -> dict[str, Any]:
    if PROMPTS.exists():
        rows = PROMPTS.read_text().splitlines()
        return {"status": "reused", "prompts": len(rows), "sha256": _sha256(PROMPTS)}
    rows: list[dict[str, Any]] = []
    for task in SPECS:
        frame = pd.read_parquet(SESSION_ROOT / f"{task}.parquet")
        primary = frame.loc[frame["validity"] == "primary"].copy()
        primary["_tokens"] = primary["prompt_prefix"].str.len()
        row = primary.sort_values(["_tokens", "participant", "trial"], kind="stable").iloc[0]
        rows.append(
            {
                "task": task,
                "participant": _json_scalar(row["participant"]),
                "trial": _json_scalar(row["trial"]),
                "prompt_prefix": row["prompt_prefix"],
                "candidate_responses": [
                    _json_scalar(value) for value in row["candidate_responses"]
                ],
                "response": _json_scalar(row["response"]),
            }
        )
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return {"status": "frozen", "prompts": len(rows), "sha256": _sha256(PROMPTS)}


def preflight() -> dict[str, Any]:
    if not SELECTION.exists():
        raise RuntimeError("O11 is sealed until the complete 8B analysis selections are frozen")
    frozen = yaml.safe_load(CONFIG.read_text())
    if not BASE_MODEL.exists() or not ADAPTER.exists():
        raise RuntimeError("The exact 70B base model and Centaur adapter must both be promoted")
    base = _weight_inventory(BASE_MODEL)
    adapter = _weight_inventory(ADAPTER)
    gpu = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": 1,
        "profile": "original",
        "precision": frozen["precision"],
        "base": base,
        "adapter": adapter,
        "gpu_name": gpu.name,
        "gpu_total_bytes": gpu.total_memory,
        "system_memory_bytes": psutil.virtual_memory().total,
        "direct_gpu_load_possible_by_weight_bytes": base["bytes"] < gpu.total_memory,
        "cpu_offload_configured": frozen["device_map"] == "auto",
        "offload_to_disk": frozen["offload_to_disk"],
        "config_sha256": _sha256(CONFIG),
        "main_config_sha256": _sha256(MAIN_CONFIG),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if result["direct_gpu_load_possible_by_weight_bytes"]:
        raise RuntimeError("Unexpected 70B inventory: direct-load feasibility assumption changed")
    return result


def _candidate_nll(
    logits: torch.Tensor,
    tokenizer: Any,
    candidates: list[str],
    response: str,
) -> float:
    ids = [tokenizer(value, add_special_tokens=False).input_ids for value in candidates]
    if any(len(value) != 1 for value in ids):
        raise RuntimeError("Frozen SRO benchmark response labels must be one token")
    selected = torch.stack(
        [torch.log_softmax(logits.float(), dim=-1)[0, value[0]] for value in ids]
    )
    normalized = torch.log_softmax(selected, dim=0)
    return float(-normalized[candidates.index(response)].cpu())


def _load_exact(adapter: bool) -> tuple[Any, Any, float]:
    frozen = yaml.safe_load(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: _effective_gpu_max_memory(frozen), "cpu": frozen["maximum_memory"]["cpu"]},
        low_cpu_mem_usage=True,
        offload_state_dict=False,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ADAPTER, local_files_only=True)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer, time.perf_counter() - started


@torch.inference_mode()
def benchmark_model(name: str, adapter: bool) -> dict[str, Any]:
    model, tokenizer, load_seconds = _load_exact(adapter)
    frozen = yaml.safe_load(CONFIG.read_text())
    layers = [int(value) for value in frozen["benchmark"]["layers"]]
    prompts = [json.loads(line) for line in PROMPTS.read_text().splitlines()]
    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    for row in prompts:
        encoded = tokenizer(row["prompt_prefix"], add_special_tokens=True, return_tensors="pt")
        embedding_device = model.get_input_embeddings().weight.device
        input_ids = encoded.input_ids.to(embedding_device)
        started = time.perf_counter()
        output = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            # Only the final-token distribution enters the frozen candidate NLL.
            # Avoid materializing prompt_length x vocabulary logits for long
            # prefixes; the retained logits are numerically identical.
            logits_to_keep=1,
        )
        elapsed = time.perf_counter() - started
        nll = _candidate_nll(
            output.logits[:, -1, :],
            tokenizer,
            list(row["candidate_responses"]),
            str(row["response"]),
        )
        rows.append(
            {
                "task": row["task"],
                "tokens": int(input_ids.shape[1]),
                "seconds": elapsed,
                "tokens_per_second": float(input_ids.shape[1] / elapsed),
                "nll": nll,
                "hidden_norms": {
                    str(layer): float(output.hidden_states[layer][0, -1].float().norm().cpu())
                    for layer in layers
                },
            }
        )
        del output
        del input_ids, encoded
    total_trials = sum(frozen["resource_gate"]["extrapolation_trials"].values())
    fastest = min(value["seconds"] for value in rows)
    lower_bound_days = fastest * total_trials / 86400
    result = {
        "model": name,
        "precision": "bfloat16_exact",
        "adapter_loaded": adapter,
        "load_seconds": load_seconds,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "rows": rows,
        "fastest_single_prefix_seconds": fastest,
        "projected_lower_bound_wall_days_without_cache": lower_bound_days,
        "hf_device_map": {
            str(key): str(value) for key, value in getattr(model, "hf_device_map", {}).items()
        },
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def benchmark() -> dict[str, Any]:
    gate = preflight()
    prompt_evidence = freeze_prompts()
    models = [
        benchmark_model("llama_base_70b", adapter=False),
        benchmark_model("centaur_70b", adapter=True),
    ]
    frozen = yaml.safe_load(CONFIG.read_text())
    projected = max(value["projected_lower_bound_wall_days_without_cache"] for value in models)
    limit = float(frozen["resource_gate"]["maximum_projected_wall_days_on_current_host"])
    status = "resource_incomplete" if projected > limit else "benchmark_passed_for_full_launch"
    payload = {
        "schema_version": 1,
        "profile": "original",
        "status": status,
        "reason": (
            "Exact BF16 weights require CPU offload and the measured lower-bound projection "
            "exceeds the frozen current-host wall-time gate."
            if status == "resource_incomplete"
            else "Exact BF16 benchmark is inside the frozen current-host wall-time gate."
        ),
        "preflight": gate,
        "prompts": prompt_evidence,
        "models": models,
        "wall_time_gate_days": limit,
        "quantized_substitution_used": False,
    }
    BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "original",
                "status": status,
                "benchmark_sha256": _sha256(BENCHMARK),
                "base_inventory": _weight_inventory(BASE_MODEL),
                "adapter_inventory": _weight_inventory(ADAPTER),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(payload, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze-prompts", "preflight", "benchmark"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "freeze-prompts":
        print(json.dumps(freeze_prompts(), sort_keys=True))
    elif args.action == "preflight":
        print(json.dumps(preflight(), sort_keys=True))
    else:
        benchmark()


if __name__ == "__main__":
    main()
