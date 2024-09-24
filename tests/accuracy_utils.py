import torch

from .conftest import TO_CPU

major, minor = torch.__version__.split(".")[:2]
skip_expr = major < "2" or minor < "2"
skip_reason = "PyTorch < 2.2.0 does not support"


RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

POINTWISE_SHAPES = [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)]
DISTRIBUTION_SHAPES = [(20, 320, 15)]
REDUCTION_SHAPES = [(4096, 256 * i) for i in range(1, 10, 2)]
MNK_SHAPES = [16, 40, 256]

DIM_POINTWISE_SHAPES = [
    (1024, 1024, 1),
    (16, 1024, 256),
    (16, 7, 128, 64, 64),
    (20, 320, 15),
]
DIMS = [[0], [-2], [2], [0, 2], [2, 1], [0, -1, 1]]

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
FLOAT_DTYPES = [torch.float16, torch.float32]
ALL_FLOAT_DTYPES = [torch.float16, torch.float32, torch.float64, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]
ALL_INT_DTYPES = [torch.int16, torch.int32, torch.int64]

SCALARS = [0.001, -0.999, 100.001, -111.999]
DIM_LIST = [0, 1]
DIMS_LIST = [0, 1, [0, 1], [1, 0]]


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    ref_inp = inp
    if TO_CPU:
        ref_inp = ref_inp.to("cpu")
    if upcast:
        ref_inp = ref_inp.to(torch.float64)
    return ref_inp


def gems_assert_close(a, b, dtype, equal_nan=False, reduce_dim=1):
    if TO_CPU:
        a = a.to("cpu")
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def gems_assert_close_groupnorm(a, b, dtype, equal_nan=False, reduce_dim=1):
    if TO_CPU:
        a = a.to("cpu")
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    if dtype == torch.float16:
        atol = 1e-2
    if dtype == torch.bfloat16:
        atol = 2e-1
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def gems_assert_close_layernorm(a, b, dtype, equal_nan=False, reduce_dim=1):
    if TO_CPU:
        a = a.to("cpu")
    b = b.to(dtype)
    atol = 1e-4 * reduce_dim
    rtol = RESOLUTION[dtype]
    if dtype == torch.float16:
        atol = 1e-2
    if dtype == torch.bfloat16:
        atol = 5e-2
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def gems_assert_equal(a, b):
    if TO_CPU:
        a = a.to("cpu")
    assert torch.equal(a, b)


# XPU Setting
# some reduction op needs specific BLOCK_SIZE params
# REDUCTION_SHAPES = [(4096, 256)]
XPU_REDUCTION_SHAPES_M = [(12, 256)]  # SHAPE[0] <= CLUSTE_NUM   GRIDX <= CLUSTE_NUM
XPU_REDUCTION_SHAPES_N = [(4096, 1)]  # SHAPE[1] == 1            GRIDY == 1
XPU_POINTWISE_2D_SHAPES_8192 = [
    (16, 1024, 8)
]  # SHAPE[-1] * SHAPE[-2] <= 8192(core_num * buffer_size limit)

ALL_FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
ALL_INT_DTYPES = [torch.int32, torch.int16]  # miss torch.int64

# vendor-test shape
KEY_OPS_SHAPES = [(1024, 32), (1024, 96), (1024, 8192), (1024, 20480), (1024, 32768)]
