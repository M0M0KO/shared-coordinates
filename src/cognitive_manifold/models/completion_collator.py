"""Completion-only masking compatible with the publisher's TRL training recipe."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from transformers import DataCollatorForLanguageModeling


def mask_marked_response_labels(
    labels: Any,
    response_token_ids: list[int],
    instruction_token_ids: list[int],
    *,
    ignore_index: int = -100,
) -> bool:
    """Mask one tokenized sequence and return whether both delimiters were found."""

    def matches(start: int, token_ids: list[int]) -> bool:
        return token_ids == labels[start : start + len(token_ids)].tolist()

    response_starts = [
        int(index) + len(response_token_ids)
        for index in np.where(labels.cpu().numpy() == response_token_ids[0])[0]
        if matches(int(index), response_token_ids)
    ]
    instruction_starts = [
        int(index)
        for index in np.where(labels.cpu().numpy() == instruction_token_ids[0])[0]
        if matches(int(index), instruction_token_ids)
    ]
    if not response_starts or not instruction_starts:
        labels[:] = ignore_index
        return False
    if instruction_starts[0] > response_starts[0]:
        instruction_starts = [0, *instruction_starts]
    for pair, (start, end) in enumerate(zip(instruction_starts, response_starts, strict=False)):
        if pair == 0:
            labels[:end] = ignore_index
        else:
            labels[start:end] = ignore_index
    if len(response_starts) < len(instruction_starts):
        labels[instruction_starts[-1] :] = ignore_index
    return True


class MarkedResponseCollator(DataCollatorForLanguageModeling):
    """Train only tokens between repeated response and instruction delimiters.

    This preserves the behavior of TRL 0.15.2's removed
    ``DataCollatorForCompletionOnlyLM`` for the publisher's repeated
    ``<<response>>`` format.
    """

    def __init__(
        self,
        *,
        response_template: list[int],
        instruction_template: list[int],
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        super().__init__(mlm=False, **kwargs)
        if not response_template or not instruction_template:
            raise ValueError("Response and instruction templates must be non-empty")
        self.response_token_ids = response_template
        self.instruction_token_ids = instruction_template
        self.ignore_index = ignore_index

    def torch_call(self, examples: list[Any]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for row in range(len(examples)):
            labels = batch["labels"][row]
            found = mask_marked_response_labels(
                labels,
                self.response_token_ids,
                self.instruction_token_ids,
                ignore_index=self.ignore_index,
            )
            if not found:
                warnings.warn(
                    "Could not find all marked-response delimiters; ignoring this example",
                    stacklevel=2,
                )
        return batch
