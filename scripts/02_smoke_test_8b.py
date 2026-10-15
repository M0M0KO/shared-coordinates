#!/usr/bin/env python3
"""Run a local-only BF16 inference smoke test for the frozen 8B checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The server also has an FSL-bundled compiler that cannot resolve Debian's
# multiarch Python headers. Keep any optional CUDA helper compilation on the
# system toolchain and avoid experimental native-JIT kernels in this smoke test.
os.environ.setdefault("CC", "/usr/bin/gcc")
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "stage0.yaml"
OUTPUT_PATH = ROOT / "results" / "m0" / "model_smoke.json"
TARGETS = ("llama_base_8b", "minitaur_8b")
PROMPT = "A participant chooses between option A and option B. The next choice is"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def run_target(target: str, model_path: Path) -> dict[str, Any]:
    if not model_path.is_dir():
        raise RuntimeError(f"Missing local checkpoint directory: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    set_seed(20260821)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded.input_ids.to("cuda:0")
    attention_mask = encoded.attention_mask.to("cuda:0")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    generated_ids = output_ids[0, input_ids.shape[1] :]
    result = {
        "target": target,
        "model_path_relative_to_project": str(model_path.relative_to(ROOT)),
        "device_name": torch.cuda.get_device_name(0),
        "dtype": str(next(model.parameters()).dtype),
        "input_tokens": int(input_ids.shape[1]),
        "generated_tokens": int(generated_ids.shape[0]),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }

    del output_ids, input_ids, attention_mask, encoded, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=(*TARGETS, "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    selected = TARGETS if args.target == "all" else (args.target,)
    results = [
        run_target(target, ROOT / config["downloads"][target]["local_dir"]) for target in selected
    ]
    payload = {
        "schema_version": 1,
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_files_only": True,
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT_PATH)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
