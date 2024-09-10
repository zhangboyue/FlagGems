import torch

from .performance_utils import (
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_BATCH,
    SIZES,
    XPU_POINTWISE_BATCH,
    Benchmark,
    arange_kwargs,
    embedding_kwargs,
    resolve_conj_arg,
    resolve_neg_arg,
    unary_int_arg,
)


def test_perf_embedding():
    bench = Benchmark(
        op_name="embedding",
        torch_op=torch.nn.functional.embedding,
        arg_func=None,
        dtypes=[
            torch.float32,
        ],  # Note(Zhengzekang): triton do not support bfloat16 atomic add which is used in embedding grad.
        batch=XPU_POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=embedding_kwargs,
    )
    bench.run()


def test_perf_topk():
    def topk_kwargs(dtype, batch, size):
        x = torch.randn((batch, size), device="cuda", dtype=dtype)
        return {"x": x, "k": 5, "dim": -1}

    bench = Benchmark(
        op_name="topk",
        torch_op=torch.topk,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=topk_kwargs,
    )
    bench.run()


def test_perf_resolve_neg():
    bench = Benchmark(
        op_name="resolve_neg",
        torch_op=torch.resolve_neg,
        arg_func=resolve_neg_arg,
        dtypes=[torch.cfloat],
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_resolve_conj():
    bench = Benchmark(
        op_name="resolve_conj",
        torch_op=torch.resolve_conj,
        arg_func=resolve_conj_arg,
        dtypes=[torch.cfloat],
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_unique():
    def unique_kwargs(dtype, batch, size):
        return {"sorted": True, "return_inverse": True, "return_counts": False}

    bench = Benchmark(
        op_name="unique",
        torch_op=torch.unique,
        arg_func=unary_int_arg,
        dtypes=INT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=unique_kwargs,
    )
    bench.run()


def test_multinomial_with_replacement():
    def multinomial_args(dtype, batch, size):
        dist = torch.rand((batch, size), dtype=dtype, device="cuda")
        n_samples = 10000
        return (dist, n_samples, True)

    bench = Benchmark(
        op_name="multinomial",
        torch_op=torch.multinomial,
        arg_func=multinomial_args,
        dtypes=(torch.float16, torch.float32),
        batch=POINTWISE_BATCH,
        sizes=SIZES,
    )
    bench.run()


def test_perf_pad():
    def padding_kwargs(dtype, batch, size):
        input = torch.randn((batch, size), device="cuda", dtype=dtype)
        rank = input.ndim
        pad_params = tuple(torch.randint(0, 10, [rank * 2]))
        pad_value = float(torch.randint(0, 1024, [1]))
        return {
            "input": input,
            "pad": pad_params,
            "mode": "constant",
            "value": pad_value,
        }

    bench = Benchmark(
        op_name="padding",
        torch_op=torch.nn.functional.pad,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=padding_kwargs,
    )
    bench.run()


def test_perf_arange():
    bench = Benchmark(
        op_name="arange",
        torch_op=torch.arange,
        arg_func=None,
        dtypes=FLOAT_DTYPES,
        batch=POINTWISE_BATCH,
        sizes=SIZES,
        kwargs_func=arange_kwargs,
    )
    bench.run()
