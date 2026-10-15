"""Parse Psych-101/Psych-201 response-delimited natural-language sessions."""

from __future__ import annotations

import re
from dataclasses import dataclass

RESPONSE = re.compile(r"<<(.*?)>>", re.DOTALL)


@dataclass(frozen=True)
class MarkedChoice:
    response_index: int
    prefix: str
    response: str
    suffix_start: int


def parse_marked_choices(text: str) -> list[MarkedChoice]:
    output: list[MarkedChoice] = []
    for index, match in enumerate(RESPONSE.finditer(text)):
        response = match.group(1)
        if not response:
            raise ValueError(f"Empty marked response at index {index}")
        output.append(
            MarkedChoice(
                response_index=index,
                prefix=text[: match.start(1)],
                response=response,
                suffix_start=match.end(1),
            )
        )
    return output


def marked_responses(text: str) -> tuple[str, ...]:
    return tuple(item.response for item in parse_marked_choices(text))


def replace_marked_responses(text: str, replacements: list[str]) -> str:
    matches = list(RESPONSE.finditer(text))
    if len(matches) != len(replacements):
        raise ValueError("Replacement count does not match marked-response count")
    pieces: list[str] = []
    cursor = 0
    for match, replacement in zip(matches, replacements, strict=True):
        pieces.extend([text[cursor : match.start(1)], replacement])
        cursor = match.end(1)
    pieces.append(text[cursor:])
    return "".join(pieces)


def parse_bool_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unrecognized boolean flag: {value!r}")
