import pytest
import torch

from sglang.jit_kernel.snake1d import snake1d
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")


def _eager_snake1d(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a = alpha.reshape(1, -1, 1)
    return x + (a + 1e-9).reciprocal() * torch.sin(a * x).pow(2)


# DAC decoder stages span large-C/small-T (early) to small-C/large-T (late),
# plus tail (T=1) and B>1; alpha is per-channel.
_SHAPES = [
    (1, 1024, 64),
    (1, 32, 960),
    (1, 512, 127),
    (2, 256, 63),
    (1, 64, 1),
    (1, 128, 4096),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", _SHAPES)
def test_snake1d_correctness(dtype, shape):
    b, c, t = shape
    x = torch.randn(b, c, t, dtype=dtype, device="cuda")
    alpha = torch.randn(c, dtype=dtype, device="cuda") * 0.5
    out = snake1d(x, alpha)
    expected = _eager_snake1d(x, alpha)
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_snake1d_inplace(dtype):
    x = torch.randn(1, 256, 128, dtype=dtype, device="cuda")
    alpha = torch.randn(256, dtype=dtype, device="cuda")
    expected = _eager_snake1d(x.clone(), alpha)
    result = snake1d(x, alpha, out=x)
    assert result is x
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(x, expected, rtol=rtol, atol=atol)


def test_snake1d_alpha_3d():
    # alpha in the model's native [1, C, 1] shape must work too.
    x = torch.randn(1, 64, 100, dtype=torch.bfloat16, device="cuda")
    alpha = torch.randn(1, 64, 1, dtype=torch.bfloat16, device="cuda")
    out = snake1d(x, alpha)
    expected = _eager_snake1d(x, alpha.reshape(-1))
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_snake1d_cpu_error():
    x = torch.randn(1, 8, 16, dtype=torch.float16)
    alpha = torch.randn(8, dtype=torch.float16)
    with pytest.raises(RuntimeError):
        snake1d(x, alpha)


def test_snake1d_unsupported_dtype():
    x = torch.randint(0, 10, (1, 8, 16), dtype=torch.int32, device="cuda")
    alpha = torch.randint(0, 10, (8,), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="dtype"):
        snake1d(x, alpha)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
