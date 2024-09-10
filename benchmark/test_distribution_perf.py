import torch

from .performance_utils import (
    FLOAT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    Benchmark,
    normal_arg,
    unary_arg,
)


def test_perf_normal():
    bench = Benchmark(
        op_name="normal",
        torch_op=torch.distributions.normal.Normal,
        arg_func=normal_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_uniform():
    bench = Benchmark(
        op_name="uniform_",
        torch_op=torch.Tensor.uniform_,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_exponential_():
    bench = Benchmark(
        op_name="exponential_",
        torch_op=torch.Tensor.exponential_,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()
