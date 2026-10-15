"""Extract final-prefix residual states and normalized candidate-choice likelihoods."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


def load_local_tokenizer(model_path: Path):
    """Load a frozen local tokenizer across Transformers 4/5 metadata names."""
    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except ValueError as error:
        if "Tokenizer class TokenizersBackend does not exist" not in str(error):
            raise
        config_path = model_path / "tokenizer_config.json"
        tokenizer_path = model_path / "tokenizer.json"
        if not config_path.exists() or not tokenizer_path.exists():
            raise
        config = json.loads(config_path.read_text())
        if config.get("tokenizer_class") != "TokenizersBackend":
            raise
        init_kwargs = {
            key: value
            for key, value in config.items()
            if key
            not in {
                "tokenizer_class",
                "auto_map",
                "added_tokens_decoder",
                "tokenizer_file",
                "from_slow",
                "__slow_tokenizer",
            }
        }
        backend = Tokenizer.from_file(str(tokenizer_path))
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            **init_kwargs,
        )


def _configure_cuda_sdpa_backend(device: torch.device) -> dict[str, Any]:
    """Avoid the unstable cuDNN SDPA graph while retaining exact BF16 execution."""
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    return {
        "policy": (
            "cudnn_disabled_flash_efficient_math_available"
            if device.type == "cuda"
            else "default_non_cuda"
        ),
        "cudnn_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "memory_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }


@dataclass(frozen=True)
class ChoiceExtraction:
    hidden_states: dict[int, torch.Tensor]
    candidate_logprobs: dict[str, float]
    nll: float
    response_logprob: float
    token_count: int


@dataclass(frozen=True)
class MarkedLikelihoodExtraction:
    response_index: int
    response: str
    hidden_states: dict[int, torch.Tensor]
    nll: float
    response_tokens: int
    state_token_count: int


class FrozenChoiceExtractor:
    """One-model extractor with explicit one-based post-block layer indexing."""

    def __init__(
        self,
        model_path: Path,
        layers: Iterable[int],
        *,
        device: str = "cuda:0",
    ) -> None:
        self.model_path = model_path
        self.layers = tuple(sorted(set(layers)))
        self.device = torch.device(device)
        self.sdpa_backend = _configure_cuda_sdpa_backend(self.device)
        self.tokenizer = load_local_tokenizer(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        blocks = self.model.model.layers
        if min(self.layers) < 1 or max(self.layers) > len(blocks):
            raise ValueError(f"Requested layers {self.layers} for {len(blocks)} model blocks")

    def _tokenize_prefix(self, prefix: str) -> torch.Tensor:
        encoded = self.tokenizer(prefix, add_special_tokens=True, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.device)
        maximum = int(self.model.config.max_position_embeddings)
        if input_ids.shape[1] > maximum:
            raise RuntimeError(
                f"Original prefix has {input_ids.shape[1]} tokens, exceeding model limit "
                f"{maximum}; "
                "truncation is forbidden by the original contract"
            )
        return input_ids

    def _single_token_choice(
        self,
        logits: torch.Tensor,
        candidates: list[str],
        response: str,
    ) -> tuple[dict[str, float], float]:
        """Normalize candidate likelihoods without changing an active KV cache."""
        continuation_ids = [self._continuation_ids(value) for value in candidates]
        if any(len(value) != 1 for value in continuation_ids):
            raise RuntimeError(
                "Session-cache extraction requires single-token response labels; "
                "run the response-label tokenizer audit before extraction"
            )
        first_logprobs = torch.log_softmax(logits.float(), dim=-1)[0]
        raw = torch.stack([first_logprobs[value[0]] for value in continuation_ids])
        normalized = torch.log_softmax(raw.to(torch.float64), dim=0)
        response_index = candidates.index(response)
        scores = {
            candidate: float(value)
            for candidate, value in zip(candidates, normalized.cpu(), strict=True)
        }
        return scores, float(normalized[response_index].cpu())

    @contextmanager
    def _capture_positions(self, local_positions: list[int]) -> Iterable[dict[int, torch.Tensor]]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        if not local_positions:
            yield captured
            return
        for one_based in self.layers:
            block = self.model.model.layers[one_based - 1]

            def hook(_module: Any, _inputs: Any, output: Any, layer: int = one_based) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                captured[layer] = tensor[0, local_positions, :].detach().to(torch.float32).cpu()

            handles.append(block.register_forward_hook(hook))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def _capture(self) -> Iterable[dict[int, torch.Tensor]]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for one_based in self.layers:
            block = self.model.model.layers[one_based - 1]

            def hook(_module: Any, _inputs: Any, output: Any, layer: int = one_based) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                captured[layer] = tensor[:, -1, :].detach().to(torch.float32).cpu()

            handles.append(block.register_forward_hook(hook))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _continuation_ids(self, value: str) -> list[int]:
        ids = self.tokenizer(value, add_special_tokens=False).input_ids
        if not ids:
            raise RuntimeError(f"Candidate tokenized to an empty sequence: {value!r}")
        return ids

    @torch.inference_mode()
    def _multitoken_logprob(self, prefix_ids: torch.Tensor, continuation: list[int]) -> float:
        output = self.model.model(input_ids=prefix_ids, use_cache=True, return_dict=True)
        next_logits = self.model.lm_head(output.last_hidden_state[:, -1, :]).float()
        score = torch.log_softmax(next_logits, dim=-1)[0, continuation[0]]
        cache = output.past_key_values
        for current, following in pairwise(continuation):
            token = torch.tensor([[current]], device=self.device)
            step = self.model.model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = step.past_key_values
            logits = self.model.lm_head(step.last_hidden_state[:, -1, :]).float()
            score = score + torch.log_softmax(logits, dim=-1)[0, following]
        return float(score.cpu())

    @torch.inference_mode()
    def extract(self, prefix: str, candidates: Iterable[str], response: str) -> ChoiceExtraction:
        candidate_list = list(candidates)
        if response not in candidate_list:
            raise ValueError("Observed response is absent from candidate set")
        prefix_ids = self._tokenize_prefix(prefix)
        with self._capture() as captured:
            output = self.model.model(input_ids=prefix_ids, use_cache=True, return_dict=True)
        final_logits = self.model.lm_head(output.last_hidden_state[:, -1, :]).float()
        first_logprobs = torch.log_softmax(final_logits, dim=-1)[0]
        scores: dict[str, float] = {}
        for candidate in candidate_list:
            continuation = self._continuation_ids(candidate)
            if len(continuation) == 1:
                scores[candidate] = float(first_logprobs[continuation[0]].cpu())
            else:
                scores[candidate] = self._multitoken_logprob(prefix_ids, continuation)
        vector = torch.tensor([scores[item] for item in candidate_list], dtype=torch.float64)
        normalized = torch.log_softmax(vector, dim=0)
        response_index = candidate_list.index(response)
        response_logprob = float(normalized[response_index])
        return ChoiceExtraction(
            hidden_states={layer: tensor.squeeze(0) for layer, tensor in captured.items()},
            candidate_logprobs={
                candidate: float(value)
                for candidate, value in zip(candidate_list, normalized, strict=True)
            },
            nll=-response_logprob,
            response_logprob=response_logprob,
            token_count=int(prefix_ids.shape[1]),
        )

    @torch.inference_mode()
    def extract_session(
        self,
        prefixes: Iterable[str],
        candidates: Iterable[Iterable[str]],
        responses: Iterable[str],
    ) -> tuple[list[ChoiceExtraction], dict[str, int]]:
        """Extract one chronological session while exactly reusing prefix KV states.

        Every current prefix is tokenized independently, as required by the frozen
        prompt contract. A cached forward is used only when the preceding token ID
        sequence is an exact prefix of the current sequence. Otherwise the cache is
        discarded and that trial is recomputed from its complete prefix.
        """
        results: list[ChoiceExtraction] = []
        previous_ids: torch.Tensor | None = None
        cache: Any | None = None
        reused = 0
        recomputed = 0
        cropped = 0
        reused_prefix_tokens = 0
        for prefix, candidate_values, response in zip(prefixes, candidates, responses, strict=True):
            candidate_list = list(candidate_values)
            if response not in candidate_list:
                raise ValueError("Observed response is absent from candidate set")
            current_ids = self._tokenize_prefix(prefix)
            common = 0
            if cache is not None and previous_ids is not None:
                shared_length = min(current_ids.shape[1], previous_ids.shape[1])
                differences = torch.nonzero(
                    current_ids[:, :shared_length] != previous_ids[:, :shared_length],
                    as_tuple=False,
                )
                common = shared_length if not len(differences) else int(differences[0, 1])
            can_reuse = cache is not None and common > 0 and common < current_ids.shape[1]
            if can_reuse:
                if common < previous_ids.shape[1]:
                    if not hasattr(cache, "crop"):
                        raise RuntimeError("The active Transformers cache cannot be safely cropped")
                    cache.crop(common)
                    cropped += 1
                input_ids = current_ids[:, common:]
                past_key_values = cache
                reused += 1
                reused_prefix_tokens += common
            else:
                input_ids = current_ids
                past_key_values = None
                recomputed += 1
            with self._capture() as captured:
                output = self.model.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            final_logits = self.model.lm_head(output.last_hidden_state[:, -1, :])
            scores, response_logprob = self._single_token_choice(
                final_logits, candidate_list, response
            )
            results.append(
                ChoiceExtraction(
                    hidden_states={layer: tensor.squeeze(0) for layer, tensor in captured.items()},
                    candidate_logprobs=scores,
                    nll=-response_logprob,
                    response_logprob=response_logprob,
                    token_count=int(current_ids.shape[1]),
                )
            )
            previous_ids = current_ids
            cache = output.past_key_values
        return results, {
            "cache_reused": reused,
            "cache_crop_events": cropped,
            "cache_prefix_tokens_reused": reused_prefix_tokens,
            "full_recomputations": recomputed,
        }

    @torch.inference_mode()
    def extract_recorded_session(
        self,
        prefixes: list[str],
        candidates: list[Iterable[str]],
        responses: list[str],
        validities: list[str],
        *,
        chunk_size: int = 4096,
    ) -> tuple[list[ChoiceExtraction], dict[str, int]]:
        """Run one canonical marked session and collect every primary choice state."""
        if not prefixes or not (
            len(prefixes) == len(candidates) == len(responses) == len(validities)
        ):
            raise ValueError("Recorded session fields must have one common nonzero length")
        final_response = responses[-1] if validities[-1] != "invalid" else ""
        text = prefixes[-1] + final_response + ">>"
        markers = list(re.finditer(r"<<(.*?)>>", text, flags=re.DOTALL))
        if len(markers) != len(prefixes):
            raise RuntimeError(f"Expected {len(prefixes)} recorded markers, found {len(markers)}")
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self.device)
        offsets = encoded.offset_mapping[0].tolist()
        maximum = int(self.model.config.max_position_embeddings)
        if input_ids.shape[1] > maximum:
            raise RuntimeError(
                f"Original session has {input_ids.shape[1]} tokens, exceeding {maximum}"
            )
        requests: list[tuple[int, int, list[str], str]] = []
        for event_index, (marker, candidate_values, response, validity) in enumerate(
            zip(markers, candidates, responses, validities, strict=True)
        ):
            if validity != "primary":
                continue
            if marker.group(1) != response:
                raise RuntimeError(
                    f"Rendered response mismatch at event {event_index}: "
                    f"{marker.group(1)!r} != {response!r}"
                )
            response_tokens = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > marker.start(1) and start < marker.end(1)
            ]
            if len(response_tokens) != 1:
                raise RuntimeError(
                    f"Response label at event {event_index} occupies {len(response_tokens)} tokens"
                )
            response_position = response_tokens[0]
            state_position = response_position - 1
            candidate_list = list(candidate_values)
            observed_ids = self._continuation_ids(response)
            if len(observed_ids) != 1 or input_ids[0, response_position].item() != observed_ids[0]:
                raise RuntimeError("Contextual response token differs from frozen label token")
            requests.append((event_index, state_position, candidate_list, response))
        by_position = {
            state_position: request_index
            for request_index, (_, state_position, *_rest) in enumerate(requests)
        }
        results: dict[int, ChoiceExtraction] = {}
        cache: Any | None = None
        chunks = 0
        for start in range(0, input_ids.shape[1], chunk_size):
            end = min(start + chunk_size, input_ids.shape[1])
            positions = sorted(position for position in by_position if start <= position < end)
            local_positions = [position - start for position in positions]
            with self._capture_positions(local_positions) as captured:
                output = self.model.model(
                    input_ids=input_ids[:, start:end],
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
            cache = output.past_key_values
            chunks += 1
            if not positions:
                continue
            logits = self.model.lm_head(output.last_hidden_state[:, local_positions, :]).float()
            for local_index, state_position in enumerate(positions):
                event_index = by_position[state_position]
                _, _, candidate_list, response = requests[event_index]
                scores, response_logprob = self._single_token_choice(
                    logits[:, local_index, :], candidate_list, response
                )
                results[event_index] = ChoiceExtraction(
                    hidden_states={
                        layer: tensor[local_index] for layer, tensor in captured.items()
                    },
                    candidate_logprobs=scores,
                    nll=-response_logprob,
                    response_logprob=response_logprob,
                    token_count=state_position + 1,
                )
        if len(results) != len(requests):
            raise RuntimeError("Not all primary states were captured from the recorded session")
        return [results[index] for index in range(len(requests))], {
            "chunks": chunks,
            "session_tokens": int(input_ids.shape[1]),
            "primary_choices": len(requests),
        }

    @torch.inference_mode()
    def extract_marked_likelihood_session(
        self,
        text: str,
        *,
        chunk_size: int = 4096,
    ) -> tuple[list[MarkedLikelihoodExtraction], dict[str, int]]:
        """Reproduce completion-only full response-span likelihoods for one text session."""
        markers = list(re.finditer(r"<<(.*?)>>", text, flags=re.DOTALL))
        if not markers:
            raise RuntimeError("Marked session contains no response spans")
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self.device)
        offsets = encoded.offset_mapping[0].tolist()
        maximum = int(self.model.config.max_position_embeddings)
        if input_ids.shape[1] > maximum:
            raise RuntimeError(
                f"Marked session has {input_ids.shape[1]} tokens, exceeding {maximum}"
            )
        response_positions: list[list[int]] = []
        state_positions: list[int] = []
        for response_index, marker in enumerate(markers):
            positions = [
                index
                for index, (start, end) in enumerate(offsets)
                if end > marker.start(1) and start < marker.end(1)
            ]
            if not positions or positions[0] == 0:
                raise RuntimeError(f"Cannot locate response tokens for span {response_index}")
            for position in positions:
                start, end = offsets[position]
                if start < marker.start(1) or end > marker.end(1):
                    raise RuntimeError(
                        f"Tokenizer crosses a response-marker boundary at span {response_index}"
                    )
            response_positions.append(positions)
            state_positions.append(positions[0] - 1)
        predictor_owner = {
            position - 1: (response_index, position)
            for response_index, positions in enumerate(response_positions)
            for position in positions
        }
        state_owner = {
            position: response_index for response_index, position in enumerate(state_positions)
        }
        losses = [0.0 for _ in markers]
        states: list[dict[int, torch.Tensor] | None] = [None for _ in markers]
        cache: Any | None = None
        chunks = 0
        for start in range(0, input_ids.shape[1], chunk_size):
            end = min(start + chunk_size, input_ids.shape[1])
            prediction_positions = sorted(
                position for position in predictor_owner if start <= position < end
            )
            selected_states = sorted(
                position for position in state_owner if start <= position < end
            )
            state_locals = [position - start for position in selected_states]
            with self._capture_positions(state_locals) as captured:
                output = self.model.model(
                    input_ids=input_ids[:, start:end],
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
            cache = output.past_key_values
            chunks += 1
            for local_index, position in enumerate(selected_states):
                response_index = state_owner[position]
                states[response_index] = {
                    layer: tensor[local_index] for layer, tensor in captured.items()
                }
            if not prediction_positions:
                continue
            prediction_locals = [position - start for position in prediction_positions]
            logits = self.model.lm_head(output.last_hidden_state[:, prediction_locals, :]).float()
            logprobabilities = torch.log_softmax(logits, dim=-1)
            for local_index, position in enumerate(prediction_positions):
                response_index, target_position = predictor_owner[position]
                target = input_ids[0, target_position]
                losses[response_index] -= float(logprobabilities[0, local_index, target].cpu())
        if any(value is None for value in states):
            raise RuntimeError("Not all marked response states were captured")
        results = [
            MarkedLikelihoodExtraction(
                response_index=index,
                response=marker.group(1),
                hidden_states=states[index] or {},
                nll=losses[index],
                response_tokens=len(response_positions[index]),
                state_token_count=state_positions[index] + 1,
            )
            for index, marker in enumerate(markers)
        ]
        return results, {
            "chunks": chunks,
            "session_tokens": int(input_ids.shape[1]),
            "responses": len(results),
        }

    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.model_path),
            "model_type": self.model.config.model_type,
            "hidden_size": int(self.model.config.hidden_size),
            "layers": list(self.layers),
            "dtype": str(next(self.model.parameters()).dtype),
            "tokenizer": self.tokenizer.__class__.__name__,
            "sdpa_backend": self.sdpa_backend,
        }

    @staticmethod
    def encode_scores(scores: dict[str, float]) -> str:
        return json.dumps(scores, sort_keys=True, separators=(",", ":"))
