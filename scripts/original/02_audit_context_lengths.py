#!/usr/bin/env python3
"""Prove full-history SRO prefixes fit the frozen 8B context without truncation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from transformers import AutoConfig, AutoTokenizer

from cognitive_manifold.prompts.sro.specs import SPECS

ROOT = Path(__file__).resolve().parents[2]
SESSION_ROOT = ROOT / "data/derived/original/sro_sessions/development"
MODEL = ROOT / "models/Llama-3.1-8B"
OUTPUT = ROOT / "reports/original/context_length_audit.json"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    maximum = int(AutoConfig.from_pretrained(MODEL, local_files_only=True).max_position_embeddings)
    evidence: dict[str, object] = {
        "model_context_limit": maximum,
        "truncate_sro_prefix": False,
        "tasks": {},
    }
    global_maximum = 0
    for task in SPECS:
        frame = pd.read_parquet(
            SESSION_ROOT / f"{task}.parquet",
            columns=["participant", "trial", "prompt_prefix"],
        )
        final_rows = frame.sort_values(["participant", "trial"]).groupby("participant").tail(1)
        lengths: list[int] = []
        prefixes = final_rows["prompt_prefix"].tolist()
        for start in range(0, len(prefixes), 32):
            encoded = tokenizer(
                prefixes[start : start + 32],
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            lengths.extend(map(len, encoded.input_ids))
        task_maximum = max(lengths)
        global_maximum = max(global_maximum, task_maximum)
        evidence["tasks"][task] = {
            "sessions": len(lengths),
            "minimum_final_prefix_tokens": min(lengths),
            "median_final_prefix_tokens": int(pd.Series(lengths).median()),
            "maximum_final_prefix_tokens": task_maximum,
        }
    labels = [chr(ord("A") + index) for index in range(26)] + ["AA"]
    evidence["response_label_token_counts"] = {
        label: len(tokenizer(label, add_special_tokens=False).input_ids) for label in labels
    }
    evidence["maximum_observed_tokens"] = global_maximum
    evidence["all_full_history_prefixes_fit"] = global_maximum <= maximum
    if not evidence["all_full_history_prefixes_fit"]:
        raise RuntimeError(f"Observed {global_maximum} tokens exceeds model context {maximum}")
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
