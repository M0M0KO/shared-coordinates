#!/usr/bin/env python3
"""Capture a whitelisted, credential-free M0 environment manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch

from cognitive_manifold.utils.redaction import contains_secret, redact_mapping

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manifests" / "public" / "environment.json"
CONFIG = ROOT / "configs" / "stage0.yaml"


def run_checked(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def cuda_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        torch.manual_seed(20260821)
        left = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
        product = left @ right
        torch.cuda.synchronize()
        properties = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "device_name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": properties.total_memory,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "smoke_shape": list(product.shape),
                "smoke_all_finite": bool(torch.isfinite(product).all().item()),
            }
        )
    return payload


def main() -> None:
    config_sha256 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    disk = shutil.disk_usage(ROOT)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_relative_to_project": os.path.relpath(sys.executable, ROOT),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "hardware": {
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total_bytes": psutil.virtual_memory().total,
            "project_filesystem_total_bytes": disk.total,
            "project_filesystem_free_bytes": disk.free,
            "nvidia_driver": run_checked(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
        },
        "cuda": cuda_payload(),
        "packages": package_versions(
            [
                "torch",
                "transformers",
                "datasets",
                "modelscope",
                "huggingface-hub",
                "accelerate",
                "numpy",
                "pandas",
                "pyarrow",
                "scikit-learn",
                "scipy",
                "pytest",
                "ruff",
            ]
        ),
        "git": {
            "branch": run_checked(["git", "branch", "--show-current"]),
            "commit": run_checked(["git", "rev-parse", "HEAD"]),
            "dirty": bool(run_checked(["git", "status", "--porcelain"])),
            "remote_count": len(run_checked(["git", "remote"]).splitlines()),
        },
        "config": {"path": "configs/stage0.yaml", "sha256": config_sha256},
        "capture_policy": {
            "environment_variables_captured": False,
            "command_lines_captured": False,
            "absolute_home_path_captured": False,
        },
    }
    payload = redact_mapping(payload)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if contains_secret(serialized):
        raise RuntimeError("Environment manifest still contains a credential signature")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
