from __future__ import annotations

import torch

from cognitive_manifold.models.completion_collator import mask_marked_response_labels


def test_repeated_marked_responses_are_the_only_unmasked_tokens() -> None:
    labels = torch.tensor([9, 10, 11, 20, 21, 30, 12, 13, 20, 22, 23, 30, 14])
    found = mask_marked_response_labels(labels, [20], [30])
    assert found
    assert labels.tolist() == [
        -100,
        -100,
        -100,
        -100,
        21,
        -100,
        -100,
        -100,
        -100,
        22,
        23,
        -100,
        -100,
    ]


def test_missing_delimiter_masks_the_complete_example() -> None:
    labels = torch.tensor([1, 2, 3, 4])
    found = mask_marked_response_labels(labels, [20], [30])
    assert not found
    assert labels.tolist() == [-100, -100, -100, -100]
