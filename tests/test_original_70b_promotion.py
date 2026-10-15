from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/original/00_promote_70b.py"


def _module():
    spec = importlib.util.spec_from_file_location("promote_original_70b", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_modelscope_inventory_rejects_size_mismatch(tmp_path: Path) -> None:
    module = _module()
    shard = "model-00001-of-00001.safetensors"
    (tmp_path / shard).write_bytes(b"weight")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": shard}})
    )
    for name in module.ESSENTIAL - {"model.safetensors.index.json"}:
        (tmp_path / name).write_text("metadata")
    files = [
        {"path": path.name, "bytes": path.stat().st_size}
        for path in tmp_path.iterdir()
        if path.is_file()
    ]
    payload = {
        "model_id": "LLM-Research/Meta-Llama-3.1-70B",
        "files": files,
    }
    module._verify_against_frozen_source(tmp_path, payload)
    next(item for item in payload["files"] if item["path"] == shard)["bytes"] += 1
    with pytest.raises(RuntimeError, match="differs from the frozen ModelScope"):
        module._verify_against_frozen_source(tmp_path, payload)
