import torch
import triton
import triton.language as tl
import logging
from ..utils import libentry, MLU_GRID_MAX

def cfggen():
    block_m = [1, 2, 4, 8]
    configs = [
        triton.Config({"BLOCK_M": m, "BLOCK_N": 1024}, num_warps=4) for m in block_m
    ]
    return configs


@libentry()
@triton.autotune(configs=cfggen(), key=["M", "N"])
@triton.jit
def var_mean_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X it should compute.
    num_prog = tl.num_programs(0)
    task_num = tl.cdiv(M, BLOCK_M)
    iter_num = tl.cdiv(task_num, num_prog)
    if task_num % num_prog != 0:
        iter_num = iter_num + 1
    for i in range(0, iter_num):
        pid = (i * num_prog + tl.program_id(0)) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        X_ptr = X + pid * N
        Var_ptr = Var + pid
        Mean_ptr = Mean + pid
        row_mask = pid < M

        # Compute mean
        _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            a = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=1) / N
        mean = mean[:, None]

        # Compute variance
        _var = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask

            x = tl.load(X_ptr + cols, mask, other=0.0).to(tl.float32)
            x = tl.where(mask, x - mean, 0.0)
            _var += x * x
        var = tl.sum(_var, axis=1) / (N - correction)
        var = var[:, None]
        # Write mean / var
        tl.store(Mean_ptr, mean, row_mask)
        tl.store(Var_ptr, var, row_mask)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    logging.debug("GEMS VAR MEAN")
    if correction is None:
        correction = 1.0

    if dim is None:
        dim = list(range(x.ndim))
    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    order = [i for i in range(x.ndim) if i not in dim] + dim
    x = x.permute(order).contiguous()
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    var = torch.empty(shape, dtype=x.dtype, device=x.device)
    mean = torch.empty(shape, dtype=x.dtype, device=x.device)

    grid = lambda META: (min(triton.cdiv(M, META["BLOCK_M"]), MLU_GRID_MAX),)
    var_mean_kernel[grid](x, var, mean, M, N, correction)
    if not keepdim:
        var = var.squeeze(dim=dim)
        mean = mean.squeeze(dim=dim)
    return var, mean
