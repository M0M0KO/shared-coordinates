"""Small, conservative helpers for keeping credentials out of logs and manifests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(hf_token|modelscope_token|api[_-]?key|access[_-]?token|"
    r"secret|password|passwd|authorization)\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{12,}\b")


def redact_text(value: str) -> str:
    """Replace common credential forms while leaving ordinary scientific text intact."""

    value = _BEARER.sub("Bearer <redacted>", value)
    value = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", value)
    return _HF_TOKEN.sub("hf_<redacted>", value)


def redact_mapping(value: Any) -> Any:
    """Recursively redact string values and secret-looking mapping keys."""

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(term in normalized for term in ("token", "secret", "password", "api_key")):
                output[str(key)] = "<redacted>"
            else:
                output[str(key)] = redact_mapping(item)
        return output
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_mapping(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


def contains_secret(value: str) -> bool:
    """Return true when text still contains a supported credential signature."""

    assignments = (
        match for match in _SECRET_ASSIGNMENT.finditer(value) if match.group(2) != "<redacted>"
    )
    return bool(next(assignments, None) or _BEARER.search(value) or _HF_TOKEN.search(value))
