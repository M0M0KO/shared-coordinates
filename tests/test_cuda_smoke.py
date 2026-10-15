from __future__ import annotations

import pytest
import torch


@pytest.mark.cuda
def test_h100_cuda_bfloat16_matmul() -> None:
    assert torch.cuda.is_available(), "M0 requires a working CUDA runtime"
    assert torch.cuda.device_count() >= 1
    major, _minor = torch.cuda.get_device_capability(0)
    assert major >= 8, "Stage 0 requires native bfloat16-capable NVIDIA hardware"
    assert torch.cuda.is_bf16_supported()

    torch.manual_seed(20260821)
    left = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    product = left @ right
    torch.cuda.synchronize()

    assert product.shape == (256, 256)
    assert torch.isfinite(product).all()
