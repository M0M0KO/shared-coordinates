from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/original/11_run_centaur70b.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_original_centaur70b", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_scalar_normalizes_numpy_values() -> None:
    module = _module()
    assert module._json_scalar(np.int64(17)) == 17
    assert type(module._json_scalar(np.int64(17))) is int
    assert module._json_scalar(np.float64(2.5)) == 2.5
    assert type(module._json_scalar(np.bool_(True))) is bool
    assert module._json_scalar("participant-1") == "participant-1"


def test_benchmark_only_materializes_final_token_logits() -> None:
    tree = ast.parse(SCRIPT.read_text())
    model_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "model"
    ]
    assert len(model_calls) == 1
    keywords = {item.arg: item.value for item in model_calls[0].keywords}
    value = keywords["logits_to_keep"]
    assert isinstance(value, ast.Constant)
    assert value.value == 1


def test_runtime_gpu_reserve_limits_weight_placement(monkeypatch) -> None:
    module = _module()
    frozen = {"maximum_memory": {"gpu_0": "74GiB"}}
    monkeypatch.delenv("CM_GPU_RESERVE_GIB", raising=False)
    assert module._effective_gpu_max_memory(frozen) == "74GiB"

    monkeypatch.setenv("CM_GPU_RESERVE_GIB", "8")
    monkeypatch.setattr(
        module.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=80 * 1024**3),
    )
    assert module._effective_gpu_max_memory(frozen) == "72GiB"


def test_benchmark_collects_cycles_before_emptying_cuda_cache() -> None:
    source = SCRIPT.read_text()
    assert "del model\n    gc.collect()\n    torch.cuda.empty_cache()" in source
