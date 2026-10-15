from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from cognitive_manifold.data.splits import participant_fold, participant_pseudonym
from cognitive_manifold.geometry.shared import task_balanced_cross_entropy
from cognitive_manifold.models.choice_extractor import (
    FrozenChoiceExtractor,
    _configure_cuda_sdpa_backend,
    load_local_tokenizer,
)
from cognitive_manifold.prompts.sro import build_session


def test_participant_hash_and_fold_are_stable_and_namespaced() -> None:
    assert participant_pseudonym(" example ") == participant_pseudonym("example")
    assert participant_fold("example") == participant_fold("example")
    assert participant_pseudonym("example").startswith("p_")
    assert 0 <= participant_fold("example") < 5


def test_sro_prefix_excludes_current_response_span() -> None:
    frame = pd.DataFrame(
        [
            {
                "worker_id": "participant",
                "exp_stage": "test",
                "time_elapsed": 1,
                "trial_id": 1,
                "trial_num": 1,
                "condition": "go",
                "stimulus": "go",
                "key_press": 32,
                "correct": 1,
                "rt": 200,
            },
            {
                "worker_id": "participant",
                "exp_stage": "test",
                "time_elapsed": 2,
                "trial_id": 2,
                "trial_num": 2,
                "condition": "nogo",
                "stimulus": "nogo",
                "key_press": -1,
                "correct": 1,
                "rt": -1,
            },
        ]
    )
    session = build_session("go_nogo", "participant", frame, "T1")
    assert len(session) == 2
    assert 'CURRENT {"stimulus_kind":"go"}' in session[0]["prompt_prefix"]
    assert session[0]["prompt_prefix"].endswith("\n<<")
    assert ">>" not in session[0]["prompt_prefix"]
    assert session[1]["prompt_prefix"].count(">>") == 1
    assert session[1]["response"] in session[1]["candidate_responses"]


def test_hidden_state_capture_uses_final_prefix_token_before_response() -> None:
    block = torch.nn.Identity()
    extractor = object.__new__(FrozenChoiceExtractor)
    extractor.layers = (1,)
    extractor.model = SimpleNamespace(model=SimpleNamespace(layers=[block]))
    sequence = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    with extractor._capture() as captured:
        block(sequence)
    assert torch.equal(captured[1], torch.tensor([[5.0, 6.0]]))


def test_cuda_extractor_disables_only_cudnn_sdpa(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        torch.backends.cuda, "enable_cudnn_sdp", lambda enabled: calls.append(enabled)
    )
    monkeypatch.setattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: False)
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "math_sdp_enabled", lambda: True)
    evidence = _configure_cuda_sdpa_backend(torch.device("cuda:0"))
    assert calls == [False]
    assert evidence == {
        "policy": "cudnn_disabled_flash_efficient_math_available",
        "cudnn_enabled": False,
        "flash_enabled": True,
        "memory_efficient_enabled": True,
        "math_enabled": True,
    }


def test_local_tokenizer_falls_back_for_transformers5_backend_metadata(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "TokenizersBackend"}'
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    sentinel = object()

    def reject_auto(*args, **kwargs):
        raise ValueError(
            "Tokenizer class TokenizersBackend does not exist or is not currently imported."
        )

    monkeypatch.setattr(
        "cognitive_manifold.models.choice_extractor.AutoTokenizer.from_pretrained",
        reject_auto,
    )
    monkeypatch.setattr(
        "cognitive_manifold.models.choice_extractor.Tokenizer.from_file",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "cognitive_manifold.models.choice_extractor.PreTrainedTokenizerFast",
        lambda *args, **kwargs: sentinel,
    )
    assert load_local_tokenizer(tmp_path) is sentinel


def test_task_balanced_loss_is_invariant_to_duplicate_rows_in_one_task() -> None:
    logits = {
        "a": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        "b": torch.tensor([[1.0, 0.0]]),
    }
    labels = {"a": torch.tensor([0, 1]), "b": torch.tensor([0])}
    baseline = task_balanced_cross_entropy(logits, labels)
    duplicated_logits = {"a": logits["a"].repeat((2, 1)), "b": logits["b"]}
    duplicated_labels = {"a": labels["a"].repeat(2), "b": labels["b"]}
    duplicated = task_balanced_cross_entropy(duplicated_logits, duplicated_labels)
    assert torch.allclose(baseline, duplicated)
