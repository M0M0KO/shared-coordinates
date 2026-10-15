from __future__ import annotations

import json

from cognitive_manifold.utils.redaction import contains_secret, redact_mapping, redact_text


def test_redact_text_removes_supported_secret_forms() -> None:
    raw = "Authorization=Bearer abcdefghijklmnop hf_token=hf_1234567890abcdefgh"
    redacted = redact_text(raw)
    assert "abcdefghijklmnop" not in redacted
    assert "hf_1234567890abcdefgh" not in redacted
    assert not contains_secret(redacted)


def test_redact_mapping_never_serializes_secret_values() -> None:
    payload = {
        "model": "Minitaur",
        "api_key": "not-for-logs",
        "nested": {"access-token": "also-not-for-logs"},
    }
    serialized = json.dumps(redact_mapping(payload), sort_keys=True)
    assert "not-for-logs" not in serialized
    assert "also-not-for-logs" not in serialized
    assert "Minitaur" in serialized
