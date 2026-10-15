"""Exact post-block Llama residual interventions with reusable downstream KV state."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask


class LlamaResidualSuffix:
    """Capture one source layer and rerun only downstream blocks after perturbation."""

    def __init__(self, model: nn.Module, source_layer_one_based: int, chunk_size: int = 4096):
        self.model = model
        self.decoder = model.model
        self.source_layer_one_based = source_layer_one_based
        self.source_index = source_layer_one_based - 1
        self.downstream = list(self.decoder.layers[source_layer_one_based:])
        self.chunk_size = chunk_size
        if not self.downstream:
            raise ValueError("Residual source must precede at least one downstream block")

    @contextmanager
    def _capture_source(self) -> Any:
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            captured["state"] = tensor.detach()

        handle = self.decoder.layers[self.source_index].register_forward_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

    @staticmethod
    def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
        midpoint = tensor.shape[-1] // 2
        first = tensor[..., :midpoint]
        second = tensor[..., midpoint:]
        return torch.cat((-second, first), dim=-1)

    @contextmanager
    def _capture_downstream_cache(self) -> Any:
        cache = DynamicCache(config=self.decoder.config)
        handles: list[Any] = []

        def hook(module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                if not inputs:
                    raise RuntimeError("Attention pre-hook received no hidden states")
                hidden_states = inputs[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError("Attention pre-hook received no rotary embeddings")
            cos, sin = position_embeddings
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            key_states = key_states * cos + self._rotate_half(key_states) * sin
            cache.update(
                key_states.detach(),
                value_states.detach(),
                module.layer_idx,
                {"cache_position": kwargs.get("cache_position")},
            )

        for layer in self.downstream:
            handles.append(
                layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
            )
        try:
            yield cache
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def capture(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, DynamicCache]:
        with self._capture_source() as captured, self._capture_downstream_cache() as cache:
            output = self.decoder(input_ids=input_ids, use_cache=False, return_dict=True)
        source = captured.get("state")
        if source is None:
            raise RuntimeError("Source-layer hook did not fire")
        logits = self.model.lm_head(output.last_hidden_state[:, -1, :]).float()
        return source, logits, cache

    def _advance(
        self,
        hidden_states: torch.Tensor,
        cache: DynamicCache,
        start_position: int,
    ) -> torch.Tensor:
        length = hidden_states.shape[1]
        cache_position = torch.arange(
            start_position,
            start_position + length,
            device=hidden_states.device,
        )
        position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.decoder.config,
            input_embeds=hidden_states,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=cache,
            position_ids=position_ids,
        )
        position_embeddings = self.decoder.rotary_emb(hidden_states, position_ids)
        for layer in self.downstream:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        return hidden_states

    @torch.inference_mode()
    def build_cache(
        self,
        source_sequence: torch.Tensor,
        cache: DynamicCache,
    ) -> DynamicCache:
        if source_sequence.ndim != 3 or source_sequence.shape[0] != 1:
            raise ValueError("Source sequence must have shape [1, tokens, width]")
        if source_sequence.shape[1] < 1:
            raise ValueError("Source sequence must contain at least one token")

        # Reuse downstream K/V captured alongside the exact full forward that
        # produced the baseline logits. Recomputing the sequence changes
        # long-context numerics, while explicitly constructing its dense causal
        # mask can exceed accelerator memory.
        observed = cache.get_seq_length(self.source_index + 1)
        if observed != source_sequence.shape[1]:
            raise RuntimeError(
                f"Full-forward cache length {observed} differs from source sequence "
                f"{source_sequence.shape[1]}"
            )
        cache.crop(source_sequence.shape[1] - 1)
        return cache

    @torch.inference_mode()
    def logits_from_source_token(
        self,
        source_token: torch.Tensor,
        cache: DynamicCache,
        prefix_length: int,
    ) -> torch.Tensor:
        if source_token.ndim == 1:
            source_token = source_token[None, None, :]
        elif source_token.ndim == 2:
            source_token = source_token[:, None, :]
        if source_token.shape[:2] != (1, 1):
            raise ValueError("Downstream intervention currently requires one source token")
        if cache.get_seq_length(self.source_index + 1) != prefix_length:
            observed = cache.get_seq_length(self.source_index + 1)
            raise RuntimeError(
                f"Downstream cache length {observed} differs from prefix {prefix_length}"
            )
        hidden = self._advance(source_token, cache, prefix_length)
        final = self.decoder.norm(hidden)
        logits = self.model.lm_head(final[:, -1, :]).float()
        cache.crop(prefix_length)
        return logits
