import inspect
import os
import time

import torch
import triton

import flag_gems

from .conftest import CPU_MODE

WARMUP = 100
REPETITION = 1000
torch.backends.cuda.matmul.allow_tf32 = False


class Benchmark:
    def __init__(
        self,
        op_name,
        torch_op,
        arg_func,
        dtypes,
        batch,
        sizes,
        is_backward=False,
        kwargs_func=None,
    ):
        self.op_name = op_name
        if is_backward:
            self.op_name += " backward"
        self.torch_op = torch_op
        self.arg_func = arg_func
        self.kwargs_func = kwargs_func
        self.dtypes = dtypes
        self.batch = batch
        self.sizes = sizes
        self.gems_op = None
        self.is_backward = is_backward
        self.mock_code = ""
        self.gems_op_map = {
            flag_gems.gelu_and_mul: "flag_gems.gelu_and_mul",
            flag_gems.skip_layer_norm: "flag_gems.skip_layer_norm",
            flag_gems.silu_and_mul: "flag_gems.silu_and_mul",
            flag_gems.skip_rms_norm: "flag_gems.skip_rms_norm",
        }

        self.arg_func_map = {
            binary_args: "binary_args",
            binary_int_args: "binary_int_args",
            ternary_args: "ternary_args",
            unary_arg: "unary_arg",
            unary_int_arg: "unary_int_arg",
            where_args: "where_args",
            cumsum_args: "cumsum_args",
            layer_norm_args: "layer_norm_args",
            cross_entropy_loss_args: "cross_entropy_loss_args",
            mv_args: "mv_args",
            normal_arg: "normal_arg",
            resolve_neg_arg: "resolve_neg_arg",
            resolve_conj_arg: "resolve_conj_arg",
            skip_layernorm_args: "skip_layernorm_args",
            skip_rmsnorm_args: "skip_rmsnorm_args",
            stack_args: "stack_args",
            hstack_args: "hstack_args",
            cat_args: "cat_args",
            cat_int_args: "cat_int_args",
            repeat_interleave_self_int_arg: "repeat_interleave_self_int_arg",
            repeat_interleave_tensor_arg: "repeat_interleave_tensor_arg",
            gather_args: "gather_args",
        }

        self.kwags_func_map = {
            flip_kwargs: "flip_kwargs",
            rand_kwargs: "rand_kwargs",
            randn_kwargs: "randn_kwargs",
            ones_kwargs: "ones_kwargs",
            zeros_kwargs: "zeros_kwargs",
            full_kwargs: "full_kwargs",
            full_like_kwargs: "full_like_kwargs",
            arange_kwargs: "arange_kwargs",
            embedding_kwargs: "embedding_kwargs",
            fill_kwargs: "fill_kwargs",
            cat_kwargs: "cat_kwargs",
        }

        self.torch_op_map = {
            torch_op_gelu_and_mul: "torch_op_gelu_and_mul",
            torch_op_silu_and_mul: "torch_op_silu_and_mul",
            torch_op_skip_layernorm: "torch_op_skip_layernorm",
            torch_op_skip_rmsnorm: "torch_op_skip_rmsnorm",
        }

    def set_gems(self, gems_op):
        self.gems_op = gems_op

    def profile(self, op, *args, **kwargs):
        fn = lambda: op(*args, **kwargs)
        if self.is_backward:
            out = fn()
            dout = torch.randn_like(out)
            fn = lambda: out.backward(dout, retain_graph=True)
        if CPU_MODE:
            for i in range(WARMUP):
                fn()
            torch.cuda.synchronize()
            start = time.time()
            for i in range(REPETITION):
                fn()
            torch.cuda.synchronize()
            end = time.time()
            latency = (end - start) / REPETITION * 1000
        else:
            latency = triton.testing.do_bench(
                fn,
                warmup=WARMUP,
                rep=REPETITION,
                return_mode="median",
                mock_code=self.mock_code,
                flag_gems_op_name=self.op_name,
            )
        # average latency in ms
        return latency

    def run(self):
        mode_str = "cpu" if CPU_MODE else "cuda"
        print("")
        for dtype in self.dtypes:
            print(
                f"Operator {self.op_name} Performance Test (dtype={dtype}, mode={mode_str})"
            )
            print("Size    Torch Latency (ms)    Gems Latency (ms)    Gems Speedup")
            print("---------------------------------------------------------------")
            for size in self.sizes:
                args = ()
                if self.arg_func is not None:
                    args = self.arg_func(dtype, self.batch, size)
                if self.is_backward:
                    args = tuple(
                        a.clone().requires_grad_()
                        if torch.is_tensor(a) and torch.is_floating_point(a)
                        else a
                        for a in args
                    )

                kwargs = {}
                if self.kwargs_func is not None:
                    kwargs = self.kwargs_func(dtype, self.batch, size)

                def getMapCode():
                    mapCode = ""
                    if self.gems_op is not None:
                        mapCode += f" {self.gems_op_map.get(self.gems_op)}\n"
                    else:
                        mapCode += "None\n"

                    mapCode += (
                        "FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]\n"
                        "INT_DTYPES = [torch.int16, torch.int32]"
                    )

                    for func in self.torch_op_map.keys():
                        mapCode += f"\n{inspect.getsource(func)}"

                    for func in self.arg_func_map.keys():
                        mapCode += f"\n{inspect.getsource(func)}"

                    for func in self.kwags_func_map.keys():
                        mapCode += f"\n{inspect.getsource(func)}"

                    return mapCode

                def getTorchOpDesc():
                    torchOpDesc = ""
                    if self.torch_op_map.get(self.torch_op):
                        torchOpDesc = f"{self.torch_op.__name__}"
                    else:
                        if hasattr(self.torch_op, "__module__"):
                            torchOpDesc = f"{self.torch_op.__module__}."
                            if hasattr(self.torch_op, "__name__"):
                                torchOpDesc += f"{self.torch_op.__name__ }"
                            else:
                                torchOpDesc += f"{self.torch_op.__class__.__name__}"
                        else:
                            torchOpDesc = f"torch.Tensor.{self.torch_op.__name__}"
                    assert torchOpDesc != ""
                    return torchOpDesc

                self.mock_code = """
import torch
import flag_gems
gems_op = """
                self.mock_code += getMapCode()

                torchOpDesc = getTorchOpDesc()

                self.mock_code += f"""
class BenchmarkMock:
    def __init__(self):
        if 'fused_gems' in globals():
            self.op = gems_op
        else:
            self.op = {torchOpDesc}
        self.arg_func = {self.arg_func_map.get(self.arg_func, "None")}
        self.kwargs_func = {self.kwags_func_map.get(self.kwargs_func, "None")}
        self.dtype = {dtype}
        self.batch = {self.batch}
        self.size = {size}

    def run(self):
        args = ()
        if self.arg_func is not None:
            args = self.arg_func(self.dtype, self.batch, self.size)
        kwargs = {{}}
        if self.kwargs_func is not None:
            kwargs = self.kwargs_func(self.dtype, self.batch, self.size)
        fn = lambda: self.op(*args, **kwargs)
        if 'use_gems' in globals():
            with flag_gems.use_gems():
                fn()
        else:
            fn()


def main():
    bm = BenchmarkMock()
    bm.run()


if __name__ == '__main__':
    main()
                """

                torch_perf = self.profile(self.torch_op, *args, **kwargs)
                if self.gems_op:
                    self.mock_code = "fused_gems=True\n" + self.mock_code
                    gems_perf = self.profile(self.gems_op, *args, **kwargs)
                    # print(self.mock_code)
                    # exit()
                else:
                    self.mock_code = "use_gems=True\n" + self.mock_code
                    with flag_gems.use_gems():
                        gems_perf = self.profile(self.torch_op, *args, **kwargs)
                speedup = torch_perf / gems_perf
                print(
                    f"{size: <8}{torch_perf: >18.6}{gems_perf: >21.6}{speedup: >16.3}"
                )


FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]

DEFAULT_BATCH = 1
POINTWISE_BATCH = 1024
REDUCTION_BATCH = 1024
XPU_POINTWISE_BATCH = 12
XPU_REDUCTION_BATCH = 12
BLAS_BATCH = 16
SIZES = [i * 64 for i in range(1, 22, 5)]

if bool(os.environ.get("TRITONXPU_FIRST_CATCH", False)):
    SIZES = [64]


def unary_arg(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    return (inp,)


def unary_int_arg(dtype, batch, size):
    inp = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cuda"
    )
    return (inp,)


def binary_args(dtype, batch, size):
    if dtype in FLOAT_DTYPES:
        inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
        inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
    elif dtype in INT_DTYPES:
        inp1 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            [batch, size],
            dtype=dtype,
            device="cuda",
        )
        inp2 = torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            [batch, size],
            dtype=dtype,
            device="cuda",
        )
    return inp1, inp2


def binary_int_args(dtype, batch, size):
    inp1 = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cuda"
    )
    inp2 = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cuda"
    )
    return inp1, inp2


def ternary_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
    inp3 = torch.randn([batch, size], dtype=dtype, device="cuda")
    return inp1, inp2, inp3


def where_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
    condition = inp1 > 0
    return condition, inp1, inp2


def flip_kwargs(dtype, batch, size):
    return {"dims": [0, 1]}


def cumsum_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    return inp, 1


def layer_norm_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    weight = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    bias = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    return (
        inp,
        [
            size,
        ],
        weight,
        bias,
    )


def cross_entropy_loss_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    target = torch.randint(
        0,
        size,
        [
            batch,
        ],
        device="cuda",
    )
    return inp, target


def mv_args(dtype, batch, size):
    inp1 = torch.randn([size, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([size], dtype=dtype, device="cuda")
    return inp1, inp2


def rand_kwargs(dtype, batch, size):
    return {"size": (batch, size), "dtype": dtype, "device": "cuda"}


def randn_kwargs(dtype, batch, size):
    return {"size": (batch, size), "dtype": dtype, "device": "cuda"}


def normal_arg(dtype, batch, size):
    loc = torch.full(size=(size, batch), fill_value=3.0, dtype=dtype, device="cuda")
    scale = torch.full(size=(size, batch), fill_value=10.0, dtype=dtype, device="cuda")
    return loc, scale


def ones_kwargs(dtype, batch, size):
    return {"size": (batch, size), "dtype": dtype, "device": "cuda"}


def zeros_kwargs(dtype, batch, size):
    return {"size": (batch, size), "dtype": dtype, "device": "cuda"}


def full_kwargs(dtype, batch, size):
    return {
        "size": (batch, size),
        "fill_value": 3.1415926,
        "dtype": dtype,
        "device": "cuda",
    }


def full_like_kwargs(dtype, batch, size):
    return {
        "input": torch.randn([batch, size], dtype=dtype, device="cuda"),
        "fill_value": 3.1415926,
    }


def arange_kwargs(dtype, batch, size):
    return {
        "end": batch * size,
        "device": "cuda",
        "dtype": dtype,
    }


def resolve_neg_arg(dtype, batch, size):
    x = torch.randn(size=(batch, size), dtype=dtype, device="cuda")
    y = x.conj()
    z = y.imag
    return (z,)


def resolve_conj_arg(dtype, batch, size):
    x = torch.randn(size=(size, batch), dtype=dtype, device="cuda")
    return (x.conj(),)


def embedding_kwargs(dtype, batch, size):
    input = torch.randint(0, batch, (batch,), device="cuda")
    weight = torch.randn((batch + 1, size), device="cuda", dtype=dtype)
    return {"input": input, "weight": weight}


def torch_op_gelu_and_mul(x, y):
    return torch.mul(torch.nn.functional.gelu(x), y)


def torch_op_silu_and_mul(x, y):
    return torch.mul(torch.nn.functional.silu(x), y)


def skip_layernorm_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    residual = torch.randn([batch, size], dtype=dtype, device="cuda")
    weight = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    bias = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    return (
        inp,
        residual,
        [
            size,
        ],
        weight,
        bias,
    )


def torch_op_skip_layernorm(inp, residual, layer_shape, weight, bias):
    return torch.layer_norm(inp + residual, layer_shape, weight, bias)


def skip_rmsnorm_args(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    residual = torch.randn([batch, size], dtype=dtype, device="cuda")
    weight = torch.randn(
        [
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    return (
        inp,
        residual,
        [
            size,
        ],
        weight,
        1e-5,
    )


def torch_op_skip_rmsnorm(x, residual, layer_shape, weight, eps):
    x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + eps)
    return weight * hidden_states


def fill_kwargs(dtype, batch, size):
    value = 1.0
    input = torch.empty(batch * size, dtype=dtype, device="cuda")
    return {
        "input": input,
        "value": value,
    }


def stack_args(dtype, batch, size):
    inp = torch.randn(size=(batch, size), dtype=dtype, device="cuda")
    return {(inp,) * 3}


def hstack_args(dtype, batch, size):
    inp = torch.randn(size=(batch, size), dtype=dtype, device="cuda")
    return {(inp,) * 3}


def cat_args(dtype, batch, size):
    inp1 = torch.randn([batch, size], dtype=dtype, device="cuda")
    inp2 = torch.randn([batch, size], dtype=dtype, device="cuda")
    return [[inp1, inp2]]


def cat_kwargs(dtype, batch, size):
    return {"dim": 0}


def cat_int_args(dtype, batch, size):
    inp1 = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cuda"
    )
    inp2 = torch.randint(
        low=0, high=0x7FFF, size=[batch, size], dtype=dtype, device="cuda"
    )
    return [[inp1, inp2]]


def repeat_interleave_self_int_arg(dtype, batch, size):
    inp = torch.randn([batch, size], dtype=dtype, device="cuda")
    repeats = 2
    return inp, repeats


def repeat_interleave_tensor_arg(dtype, batch, size):
    repeats = torch.randint(
        low=0,
        high=0x7F,
        size=[
            size,
        ],
        dtype=dtype,
        device="cuda",
    )
    return (repeats,)


def gather_args(dtype, batch, size):
    inp_shape = [batch, size]
    inp = torch.randn(inp_shape, dtype=dtype, device="cuda")
    import random

    dim = random.choice([0, 1])
    size_dim = inp_shape[dim]
    index_shape = [
        random.randint(1, inp_shape[0]),
        random.randint(1, inp_shape[1]),
    ]
    index = torch.empty(tuple(index_shape), dtype=torch.long, device="cuda")

    m, n = index_shape

    index_size_dim = index_shape[dim]
    # make unique indices
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            ii = [i, j]
            ii[dim] = slice(0, index.size(dim) + 1)
            index[tuple(ii)] = torch.randperm(size_dim)[0:index_size_dim]

    return (inp, dim, index)
