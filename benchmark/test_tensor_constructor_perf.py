import torch

from .performance_utils import (
    FLOAT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    Benchmark,
    full_kwargs,
    full_like_kwargs,
    ones_kwargs,
    rand_kwargs,
    randn_kwargs,
    unary_arg,
    zeros_kwargs,
)


def test_perf_rand():
    bench = Benchmark(
        op_name="rand",
        torch_op=torch.rand,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=rand_kwargs,
    )
    bench.run()


def test_perf_randn():
    # SIZES>384 will exceed memory
    bench = Benchmark(
        op_name="randn",
        torch_op=torch.randn,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=[i * 64 for i in range(1, 7, 5)],
        kwargs_func=randn_kwargs,
    )
    bench.run()


def test_perf_rand_like():
    bench = Benchmark(
        op_name="rand_like",
        torch_op=torch.rand_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_randn_like():
    # SIZES>384 will exceed memory
    bench = Benchmark(
        op_name="randn_like",
        torch_op=torch.randn_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=[i * 64 for i in range(1, 7, 5)],
    )
    bench.run()


def test_perf_ones():
    bench = Benchmark(
        op_name="ones",
        torch_op=torch.ones,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=ones_kwargs,
    )
    bench.run()


def test_perf_zeros():
    bench = Benchmark(
        op_name="zeros",
        torch_op=torch.zeros,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=zeros_kwargs,
    )
    bench.run()


def test_perf_full():
    bench = Benchmark(
        op_name="full",
        torch_op=torch.full,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=full_kwargs,
    )
    bench.run()


def test_perf_ones_like():
    bench = Benchmark(
        op_name="ones_like",
        torch_op=torch.ones_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_zeros_like():
    bench = Benchmark(
        op_name="zeros_like",
        torch_op=torch.zeros_like,
        arg_func=unary_arg,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_full_like():
    bench = Benchmark(
        op_name="full_like",
        torch_op=torch.full_like,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=full_like_kwargs,
    )
    bench.run()
