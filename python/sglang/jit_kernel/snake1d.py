from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
try:
    from sglang.kernel_api_logging import debug_kernel_api
except ImportError:  # older sglang (e.g. 0.5.8) has no kernel_api_logging

    def debug_kernel_api(fn):
        return fn

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_snake1d_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT Snake1d module for a given dtype."""
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "snake1d",
        *args,
        cuda_files=["elementwise/snake1d.cuh"],
        cuda_wrappers=[("snake1d", f"snake1d<{args}>")],
    )


@debug_kernel_api
def snake1d(
    x: torch.Tensor,
    alpha: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused Snake1d activation: ``out = x + 1/(alpha+1e-9) * sin(alpha*x)^2``.

    Replaces the ~7 element-wise ops of the eager Snake1d with a single kernel.

    Parameters
    ----------
    x     : CUDA tensor ``[B, C, T]`` (FP16 / BF16 / FP32), contiguous
    alpha : per-channel parameter, ``[C]`` or ``[1, C, 1]`` (same dtype as ``x``)
    out   : optional pre-allocated output (same shape/dtype as ``x``); ``out=x``
            is allowed for in-place.

    Returns
    -------
    Activated tensor, same shape/dtype as ``x``.
    """
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"Unsupported dtype {x.dtype}. Supported: float16, bfloat16, float32"
        )
    if x.dim() != 3:
        raise RuntimeError(f"snake1d expects a 3D [B, C, T] tensor, got {tuple(x.shape)}")
    if out is None:
        out = torch.empty_like(x)

    module = _jit_snake1d_module(x.dtype)
    module.snake1d(out, x, alpha.reshape(-1).contiguous())
    return out
