from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/original/04_fit_shared_controls.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("original_shared_controls", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_basis_creates_missing_parent_directory(tmp_path: Path) -> None:
    module = _load_script_module()
    basis = np.eye(3, dtype=np.float32)
    target = tmp_path / "new-cache" / "nested" / "basis.npy"

    module._write_basis(target, basis)

    assert target.is_file()
    np.testing.assert_array_equal(np.load(target), basis)
