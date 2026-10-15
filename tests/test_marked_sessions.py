from __future__ import annotations

import pytest

from cognitive_manifold.data.marked_sessions import (
    marked_responses,
    parse_bool_flag,
    parse_marked_choices,
    replace_marked_responses,
)


def test_marked_choice_prefix_stops_after_opening_delimiter() -> None:
    text = "Instruction. Trial one <<A>> feedback. Trial two <<B>> feedback."
    choices = parse_marked_choices(text)
    assert choices[0].prefix == "Instruction. Trial one <<"
    assert choices[1].prefix.endswith("Trial two <<")
    assert choices[1].response == "B"
    assert marked_responses(text) == ("A", "B")


def test_marked_replacement_preserves_every_nonresponse_character() -> None:
    text = "x<<A>>y<<B>>z"
    assert replace_marked_responses(text, ["B", "A"]) == "x<<B>>y<<A>>z"


@pytest.mark.parametrize("value", [True, "True", "true", "1", "yes"])
def test_bool_flags_accept_true_variants(value: object) -> None:
    assert parse_bool_flag(value) is True


@pytest.mark.parametrize("value", [False, "False", "false", "0", "no"])
def test_bool_flags_accept_false_variants(value: object) -> None:
    assert parse_bool_flag(value) is False
