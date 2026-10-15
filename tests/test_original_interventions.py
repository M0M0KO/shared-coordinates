from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from cognitive_manifold.models import residual_intervention

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/original/10_run_interventions.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_original_interventions", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_swap_translates_donor_response_through_unhashed_semantics() -> None:
    module = _module()
    recipient = module._current_option_semantics("CURRENT stimulus OPTIONS[A=key:37,B=key:39]\n<<")
    donor = module._current_option_semantics("CURRENT stimulus OPTIONS[B=key:37,A=key:39]\n<<")
    assert module._recipient_label_for_donor_semantic(recipient, donor, "B") == "A"
    assert module._recipient_label_for_donor_semantic(recipient, donor, "A") == "B"


def test_swap_rejects_semantic_action_absent_from_recipient_options() -> None:
    module = _module()
    recipient = {"A": "choose:red", "B": "choose:green"}
    donor = {"Q": "box:17", "R": "choose:red"}
    assert module._recipient_label_for_donor_semantic(recipient, donor, "Q") is None


def test_exact_cache_reuses_full_forward_then_crops_final_token() -> None:
    class FakeCache:
        def __init__(self, length: int) -> None:
            self.length = length
            self.crops: list[int] = []

        def get_seq_length(self, _layer: int) -> int:
            return self.length

        def crop(self, length: int) -> None:
            self.crops.append(length)

    runner = object.__new__(residual_intervention.LlamaResidualSuffix)
    runner.decoder = SimpleNamespace(config=object())
    runner.source_index = 23
    source = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    full_forward_cache = FakeCache(length=3)
    cache = runner.build_cache(source, full_forward_cache)

    assert cache is full_forward_cache
    assert cache.crops == [2]


def test_o10_uses_fp32_exact_downstream_rerun() -> None:
    module = _module()
    assert module.INTERVENTION_MODEL_DTYPE is torch.float32
    assert (
        module.DOWNSTREAM_CACHE_STRATEGY
        == "capture_downstream_kv_during_full_forward_then_crop_final_token"
    )


def test_rotate_half_matches_llama_pairing() -> None:
    tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    rotated = residual_intervention.LlamaResidualSuffix._rotate_half(tensor)
    assert torch.equal(rotated, torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))
