import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.benchmark.utils import create_random, get_benchmark_range
from sglang.jit_kernel.snake1d import snake1d as jit_snake1d
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="base-b-kernel-benchmark-1-gpu-large")


@torch.compile()
def torch_snake1d(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a = alpha.reshape(1, -1, 1)
    return x + (a + 1e-9).reciprocal() * torch.sin(a * x).pow(2)


# (C, T) pairs spanning the DAC decoder stages: early = large C / small T,
# late = small C / large T (after upsampling).
SHAPE_LIST = get_benchmark_range(
    full_range=[
        (1024, 64),
        (512, 128),
        (256, 256),
        (128, 512),
        (64, 1024),
        (32, 2048),
    ],
    ci_range=[(1024, 64), (64, 1024)],
)
FN_MAP = {"jit": jit_snake1d, "torch": torch_snake1d}


@marker.mark_args("shape", SHAPE_LIST)
@marker.mark_benchmark("impl", ["jit", "torch"])
def benchmark(shape, impl: str):
    c, t = shape
    x = create_random(1, c, t)
    alpha = create_random(c)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x, alpha),
        # x is read -> clone per iter to defeat L2 reuse; alpha is tiny.
        graph_clone_args=(0,),
        # Snake is memory-bound -> report GB/s based on the activation tensor.
        memory_args=(x,),
    )


if __name__ == "__main__":
    benchmark.run()
