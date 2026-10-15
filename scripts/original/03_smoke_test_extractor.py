#!/usr/bin/env python3
"""Compare exact full-prefix and cache-reused extraction before original O3."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cognitive_manifold.models.choice_extractor import FrozenChoiceExtractor

ROOT = Path(__file__).resolve().parents[2]
SESSION = ROOT / "data/derived/original/sro_sessions/development/simon.parquet"
MODEL = ROOT / "models/Llama-3.1-Minitaur-8B"
OUTPUT = ROOT / "reports/original/8b_extractor_smoke.json"
LAYERS = [8, 16, 24, 32]


def main() -> None:
    frame = pd.read_parquet(SESSION)
    frame = frame.loc[frame["validity"] == "primary"].sort_values(
        ["participant", "trial"], kind="stable"
    )
    participant = frame.iloc[0]["participant"]
    sample = frame.loc[frame["participant"] == participant].head(8)
    extractor = FrozenChoiceExtractor(MODEL, LAYERS)
    cached, cache_evidence = extractor.extract_session(
        sample["prompt_prefix"].tolist(),
        sample["candidate_responses"].tolist(),
        sample["response"].tolist(),
    )
    independent = [
        extractor.extract(row.prompt_prefix, row.candidate_responses, row.response)
        for row in sample.itertuples(index=False)
    ]
    comparisons: list[dict[str, object]] = []
    for index, (left, right) in enumerate(zip(cached, independent, strict=True)):
        layer_cosines: dict[str, float] = {}
        layer_maximum_absolute_difference: dict[str, float] = {}
        for layer in LAYERS:
            cached_state = left.hidden_states[layer].numpy()
            full_state = right.hidden_states[layer].numpy()
            denominator = np.linalg.norm(cached_state) * np.linalg.norm(full_state)
            layer_cosines[str(layer)] = float(np.dot(cached_state, full_state) / denominator)
            layer_maximum_absolute_difference[str(layer)] = float(
                np.max(np.abs(cached_state - full_state))
            )
        comparisons.append(
            {
                "trial_index": index,
                "token_count": left.token_count,
                "response_logprob_absolute_difference": abs(
                    left.response_logprob - right.response_logprob
                ),
                "minimum_layer_cosine": min(layer_cosines.values()),
                "maximum_layer_absolute_difference": max(
                    layer_maximum_absolute_difference.values()
                ),
                "layer_cosines": layer_cosines,
                "layer_maximum_absolute_difference": layer_maximum_absolute_difference,
            }
        )
    evidence = {
        "profile": "original",
        "model": "minitaur_8b",
        "model_identity": extractor.identity(),
        "task": "simon",
        "trials": len(sample),
        "cache_evidence": cache_evidence,
        "maximum_response_logprob_absolute_difference": max(
            row["response_logprob_absolute_difference"] for row in comparisons
        ),
        "minimum_layer_cosine": min(row["minimum_layer_cosine"] for row in comparisons),
        "maximum_layer_absolute_difference": max(
            row["maximum_layer_absolute_difference"] for row in comparisons
        ),
        "comparisons": comparisons,
    }
    evidence["acceptance_thresholds"] = {
        "maximum_response_logprob_absolute_difference": 0.1,
        "minimum_layer_cosine": 0.9995,
    }
    evidence["passed"] = bool(
        evidence["cache_evidence"]["cache_reused"] > 0
        and evidence["maximum_response_logprob_absolute_difference"] <= 0.1
        and evidence["minimum_layer_cosine"] >= 0.9995
    )
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, sort_keys=True))
    if not evidence["passed"]:
        raise RuntimeError("Cache-reused extraction did not match full-prefix extraction")


if __name__ == "__main__":
    main()
