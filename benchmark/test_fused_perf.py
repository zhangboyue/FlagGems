import torch

import flag_gems

from .performance_utils import (
    FLOAT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    XPU_REDUCTION_BATCH,
    Benchmark,
    binary_args,
    skip_layernorm_args,
    skip_rmsnorm_args,
    torch_op_gelu_and_mul,
    torch_op_silu_and_mul,
    torch_op_skip_layernorm,
    torch_op_skip_rmsnorm,
)


def test_perf_gelu_and_mul():
    gems_op = flag_gems.gelu_and_mul

    bench = Benchmark(
        op_name="gelu_and_mul",
        torch_op=torch_op_gelu_and_mul,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.set_gems(gems_op)
    bench.run()


def test_perf_silu_and_mul():
    gems_op = flag_gems.silu_and_mul

    bench = Benchmark(
        op_name="silu_and_mul",
        torch_op=torch_op_silu_and_mul,
        arg_func=binary_args,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.set_gems(gems_op)
    bench.run()


def test_perf_skip_layernorm():
    gems_op = flag_gems.skip_layer_norm

    bench = Benchmark(
        op_name="skip_layernorm",
        torch_op=torch_op_skip_layernorm,
        arg_func=skip_layernorm_args,
        dtypes=[torch.float32, torch.bfloat16],
        batch=XPU_REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.set_gems(gems_op)
    bench.run()


def test_perf_skip_rmsnorm():
    gems_op = flag_gems.skip_rms_norm

    bench = Benchmark(
        op_name="skip_rmsnorm",
        torch_op=torch_op_skip_rmsnorm,
        arg_func=skip_rmsnorm_args,
        dtypes=FLOAT_DTYPES,
        batch=XPU_REDUCTION_BATCH,
        sizes=SIZES,
    )
    bench.set_gems(gems_op)
    bench.run()
