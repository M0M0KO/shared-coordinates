"""Stable, public-namespace participant pseudonyms and folds."""

from __future__ import annotations

import hashlib
import unicodedata

NAMESPACE = "cognitive-manifold|original|v1"


def canonical_participant_id(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value).strip())
    if not text:
        raise ValueError("Participant identifier is empty")
    return text


def participant_digest(value: object) -> bytes:
    canonical = canonical_participant_id(value)
    return hashlib.sha256(f"{NAMESPACE}|{canonical}".encode()).digest()


def participant_fold(value: object, folds: int = 5) -> int:
    if folds < 2:
        raise ValueError("At least two folds are required")
    return int.from_bytes(participant_digest(value)[:8], "big") % folds


def participant_pseudonym(value: object) -> str:
    return "p_" + participant_digest(value).hex()[:20]
