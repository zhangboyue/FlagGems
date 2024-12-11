import logging
import math

import torch
import triton
import triton.language as tl

from .. import runtime
from ..utils import libentry
from ..utils import triton_lang_extension as tle
from ..utils.type_utils import get_accumulator_dtype

# --------------------------------------------------------------------------------------------------------------------
#   Kernel implementation
# --------------------------------------------------------------------------------------------------------------------
# TODO.boyue add autotune decorator back
@libentry()
@triton.jit(do_not_specialize=["epsilon"])
def kernel_bn_forward(y_ptr, x_ptr, weight_ptr, bias_ptr, epsilon,
                      lock_ptr, sum_x_ptr, sum_xx_ptr, n_completed_ptr,
                      stride_n, stride_c, stride_h, stride_w, N, C, H, W,
                      BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr):
    """
    Forward of batch norm kernel. Tiling along axis-(H, W, N).
    """
    pid_n, pid_c, pid_hw = tl.program_id(axis=0), tl.program_id(axis=1), tl.program_id(axis=2)

    # pointer arithmetic, and mask calculation
    ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    ofst_c = pid_c*BLOCK_C + tl.arange(0, BLOCK_C)
    ofst_hw = pid_hw*BLOCK_HW + tl.arange(0, BLOCK_HW)

    x_ofst = (stride_n*ofst_n)[:, None, None] + (stride_c*ofst_c)[None, :, None] + (stride_w*ofst_hw)[None, None, :]

    mask_n = ofst_n < N
    mask_c = ofst_c < C
    mask_hw = ofst_hw < H*W
    x_mask = mask_n[:, None, None] * mask_c[None, :, None] * mask_hw[None, None, :]

    # do partial sum
    x = tl.load(x_ptr + x_ofst, mask=x_mask, other=0.0)
    x = tl.permute(x, (0, 2, 1)).reshape(BLOCK_N*BLOCK_HW, BLOCK_C)
    sum_x = tl.sum(x, axis=0)
    sum_xx = tl.sum(x*x, axis=0)

    # do full reduction
    while tl.atomic_cas(lock_ptr, 0, 1) != 0:
        pass
    accum_x = tl.load(sum_x_ptr + ofst_c, mask=mask_c)
    accum_xx = tl.load(sum_xx_ptr + ofst_c, mask=mask_c)
    tl.store(sum_x_ptr + ofst_c, accum_x + sum_x, mask=mask_c)
    tl.store(sum_xx_ptr + ofst_c, accum_xx + sum_xx, mask=mask_c)
    n_completed = tl.load(n_completed_ptr)
    tl.store(n_completed_ptr, n_completed + 1)
    tl.atomic_xchg(lock_ptr, 0)

    # synchronize
    n_blocks = tl.cdiv(N, BLOCK_N)*tl.cdiv(C, BLOCK_C)*tl.cdiv(H*W, BLOCK_HW)
    while tl.load(n_completed_ptr, volatile=True) < n_blocks:
        pass

    # do element wise operation
    weight = tl.load(weight_ptr + ofst_c, mask=mask_c, other=0.0)
    bias = tl.load(bias_ptr + ofst_c, mask=mask_c, other=0.0)
    sum_x = tl.load(sum_x_ptr + ofst_c, mask=mask_c, other=0.0)
    sum_xx = tl.load(sum_xx_ptr + ofst_c, mask=mask_c, other=0.0)
    mean_x = sum_x/(N*H*W)
    var_x = sum_xx/(N*H*W) - mean_x*mean_x
    inv_stdvar = 1.0/(tl.sqrt(var_x) + epsilon)
    x = x.reshape(BLOCK_N, BLOCK_HW, BLOCK_C).permute(0, 2, 1)
    normed_x = (x - mean_x[None, :, None])*inv_stdvar[None, :, None]
    normed_x = normed_x*weight[None, :, None] + bias[None, :, None]

    tl.store(y_ptr + x_ofst, normed_x, mask=x_mask)

@triton.jit
def kernel_bn_backward_grad_weight(result_ptr, grad_y_ptr, normed_x_ptr, lock_ptr, n_completed_ptr,
                                   N, C, H, W, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    indices = BLOCK_SIZE*pid + tl.arange(0, BLOCK_SIZE)
    mask = indices < N*C*H*W
    grad_y = tl.load(grad_y_ptr + indices, mask=mask, other=0.0)
    normed_x = tl.load(normed_x_ptr + indices, mask=mask, other=0.0)
    psum = tl.sum(grad_y*normed_x)

    while tl.atomic_cas(lock_ptr, 0, 1) != 0:
        pass
    accum = tl.load(result_ptr)
    tl.store(result_ptr, psum + accum)
    tl.atomic_xchg(lock_ptr, 0)

@triton.jit
def kernel_bn_backward_grad_bias(result_ptr, grad_y_ptr, lock_ptr, n_completed_ptr,
                                 N, C, H, W, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    indices = BLOCK_SIZE*pid + tl.arange(0, BLOCK_SIZE)
    mask = indices < N*C*H*W
    grad_y = tl.load(grad_y_ptr + indices, mask=mask, other=0.0)
    psum = tl.sum(grad_y)
    while tl.atomic_cas(lock_ptr, 0, 1) != 0:
        pass
    accum = tl.load(result_ptr)
    tl.store(result_ptr, psum + accum)
    tl.atomic_xchg(lock_ptr, 0)

@triton.jit
def _kernel_mean_method0(Y, X, SHAPE_N, SHAPE_C, SHAPE_HW, stride_n, stride_hw, stride_c,
                         BLOCK_N: tl.constexpr, BLOCK_B: tl.constexpr, BLOCK_HW: tl.constexpr):
    """
    Desc
    ====
        calculate mean of the input tensor. shape relation of input and output (N, C, HW) -> (C,)
        Kernel of method0 does not split channel dimmension. As consequence, not synchronization between blocks
        is required.

    Parameter
    =========
        Y : output tensor, shape=(SHAPE_C,)
        X : input tensor, shape=(SHAPE_N, SHAPE_C, SHAPE_HW)
        (SHAPE_N, SHAPE_C, SHAPE_HW): shape parameters
        (stride_n, stride_c, stride_hw): stride of input tensoor
        (BLOCK_N, BLOCK_C, BLOCK_HW): shape of a block
    """
    pid_n, pid_c, pid_hw = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    ofst_c = pid_c*BLOCK_C + tl.arange(0, BLOCK_C)
    ofst_hw = pid_hw*BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_n = ofst_n < SHAPE_BATCH
    mask_c = ofet_c < SHAPE_C
    mask_hw = ofst_hw < SHAPE_HW

    ofst_x = ofst_n*stride_n[:, None, None] + ofst_c*stride_c[None, :, None] + ofst_hw*stride_hw[None, None, :]
    mask_x = mask_n[:, None, None] * mask_c[None, :, None] * mask_hw[None, None, :]

    x = tl.load(X + ofst_x, mask=mask_x, other=0.0)
    x_permute = tl.permute(x, (1, 0, 2)).reshape(BLOCK_C, BLOCK_N*BLOCK_HW)
    mean = tl.sum(x, axis=1)/(SHAPE_N*SHAPE_HW)

    ofst_y, mask_y = ofst_c, mask_c
    tl.store(Y + ofst_y, mean, mask=mask_y)

@triton.jit
def _kernel_mean_method1(Y: torch.Tensor, X: torch.Tensor, n_completed: torch.Tensor, mutex: torch.Tensor,
                         SHAPE_N, SHAPE_C, SHAPE_HW, stride_n, stride_c, stride_hw,
                         BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr):
    """
    Desc
    ====
        calculate mean of the input tensor. shape relation of input and output (N, C, HW) -> (C,)
        Kernel of method1 split channel dimmension, which implies that synchronization between block
        is required.

    Parameter
    =========
        Y : output tensor, shape=(SHAPE_C,)
        X : input tensor, shape=(SHAPE_N, SHAPE_C, SHAPE_HW)
        n_completed : auxilary flag of integer type, showing how many partial sums has been accumulated.
                      shape=(SHAPE_C,)
        mutex: a software mutex lock for synchronization between blocks. shape=(SHAPE_C,)
        (SHAPE_N, SHAPE_C, SHAPE_HW): shape parameters
        (BLOCK_N, BLOCK_C, BLOCK_HW): shape of a block
    """
    pid_n, pid_c, pid_hw = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    ofst_c = pid_c*BLOCK_C + tl.arange(0, BLOCK_C)
    ofst_hw = pid_hw*BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_n = ofst_n < SHAPE_BATCH
    mask_c = ofst_c < SHAPE_C
    mask_hw = ofst_hw < SHAPE_HW

    ofst_x = ofst_n*stride_n[:, None, None] + ofst_c*stride_c[None, :, None] + ofst_hw*stride_hw[None, None, :]
    mask_x = mask_n[:, None, None] * mask_c[None, :, None] * mask_hw[None, None, :]

    x = tl.load(X + ofst_x, mask=mask_x, other=0.0)
    x_permute = tl.permute(x, (1, 0, 2)).reshape(BLOCK_C, BLOCK_N*BLOCK_HW)
    p_sum = tl.sum(x_permute, axis=1)

    # add partial sum to global accumulation
    while tl.atomic_cas(mutex + pid_c, 0, 1) == 1:
        continue

    ofst_y, mask_y = ofst_c, mask_c
    accum = tl.load(Y + ofst_y, mask=mask_y)
    tl.store(Y + ofst_y, accum + p_sum, mask=mask_y)
    tl.atomic_add(n_completed + pid_c, 1)


    # software approach, to synchronize between block
    while tl.load(n_completed + pid_c, volatile=True) < tl.cdiv(SHAPE_C, BLOCK_C):
        continue

    # now partial sum by each block has been accumualted, continue calculating mean result
    # this is only required by block-0
    if pid_c == 0:
        accum = tl.load(Y + ofst_y, mask=mask)
        mean_x = accum/(BLOCK_N*BLOCK_HW)
        tl.store(Y + ofst_y, mask=mask_y)

@triton.jit
def _kernel_bn_elementwise_prod(C: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                                SHAPE_N, SHAPE_C, SHAPE_HW, stride_n, stride_c, stride_hw,
                                BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr):
    """
    Desc
    ====
        does element-wise production of two input tensors of the same shape. channel first layout is assumed

    Kernel Parameters
    =================
        C: result tensor, shape=(BLOCK_N, BLOCK_C, BLOCK_HW)
        A: 1st operand, shape=(BLOCK_N, BLOCK_C, BLOCK_HW)
        B: 2nd operand, shape=(BLOCK_N, BLOCK_C, BLOCK_HW)
        (SHAPE_N, SHAPE_C, SHAPE_HW): shape of the global tensor
        (stride_n, stride_c, stride_hw): stride of the global tensor, assuming same physical layout of A, B and C
        (BLOCK_N, BLOCK_C, BLOCK_HW): block shape of tiled tensor, processed by each kernel
    """
    pid_n, pid_c, pid_hw = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    ofst_c = pid_c*BLOCK_C + tl.arange(0, BLOCK_C)
    ofst_hw = pid_hw*BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_n = ofst_n < SHAPE_N
    mask_c = ofst_c < SHAPE_C
    mask_hw = ofst_hw < SHAPE_HW

    ofst = stride_n*ofst_n[:, None, None] +\
           stride_c*ofst_c[None, :, None] +\
           stride_hw*ofst_hw[None, None, :]
    mask = mask_n[:, None, None] * mask_c[None, :, None] * mask_hw[None, None, :]

    a_tile = tl.load(A + ofst, mask=mask, other=0.0)
    b_tile = tl.load(B + ofst, mask=mask)
    tl.store(C + ofst, a_tile*b_tile, mask=mask)

@triton.jit
def _kernel_bn_grad_input(grad_x, y, grad_y, grad_y_mean, prod_mean, gamma,
                          SHAPE_N, SHAPE_C, SHAPE_HW, stride_n, stride_c, stride_hw,
                          BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr):
    """
    Desc
    ====
        backward gradient w.r.t input, channel first layout is assumed.
        grad_x = 1/delta (grad_y - mean(grad_y)[None, :, None] - y*(mean(y*grad_y)[None, :, None]))

    Kernel Parameters
    =================
        grad_x: output tensor, shape=(BLOCK_N, BLOCK_C, BLOCK_HW)
        y, input tensor: normalzed x saved to context during forward pass
        grad_y: input tensor, gradient of the BN operator result
        grad_y_mean: mean of gradident y, calculated by backward kernel
        prod_mean: mean(y*grad_y), calculated by backward kernel
        (SHAPE_N, SHAPE_C, SHAPE_HW): shape of the global tensor
        (stride_n, stride_c, stride_hw): stride of the global tensor, assuming same physical layout of A, B and C
        (BLOCK_N, BLOCK_C, BLOCK_HW): block shape of tiled tensor, processed by each kernel
    """
    pid_n, pid_c, pid_hw = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ofst_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    ofst_c = pid_c*BLOCK_C + tl.arange(0, BLOCK_C)
    ofst_hw = pid_hw*BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_n = ofst_n < SHAPE_BATCH
    mask_c = ofet_c < SHAPE_C
    mask_hw = ofst_hw < SHAPE_HW

    ofst = ofst_n*stride_n[:, None, None] + ofst_c*stride_c[None, :, None] + ofst_hw*stride_hw[None, None, :]
    mask = mask_n[:, None, None] * mask_c[None, :, None] * mask_hw[None, None, :]

    _y = tl.load(y + ofst, mask=mask)
    _grad_y = tl.load(grad_y + ofst, mask=mask)
    _grad_y_mean = tl.load(C + ofst_c, mask=mask_c)

    _grad_x = _grad_y - _grad_y_mean[None, :, None] - _y*prod_mean[None, :, None]
    tl.store(grad_x + ofst, _grad_x/gammra[None, :, None], mask=mask)

# --------------------------------------------------------------------------------------------------------------------
#   Operator class
# -------------------------------------------------------------------------------------------------------------------
class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, epsilon=1e-05):
        logging.debug("BN forward")
        assert (len(weight.shape) == len(bias.shape)) and (len(weight.shape) == 1)
        assert (len(x.shape) == 4) and (x.shape[1] == bias.shape[0]) and (weight.shape[0] == bias.shape[0])
        N, C, H, W = x.shape

        y = torch.empty(N, C, H, W, dtype=torch.float32, device='cuda')
        sum_x = torch.zeros(C, dtype=torch.float32, device='cuda')
        sum_xx = torch.zeros(C, dtype=torch.float32, device='cuda')
        lock = torch.zeros(1, dtype=torch.int32, device='cuda')
        n_completed = torch.zeros(1, dtype=torch.int32, device='cuda')

        # Static tiling. TODO: update to flexble tiling according to input shape
        BLOCK_N, BLOCK_C, BLOCK_HW = 4, 8, 128
        cdiv = lambda a, b: (a + b - 1) // b
        grid = lambda meta: (cdiv(N, meta["BLOCK_N"]), cdiv(C, meta["BLOCK_C"]), cdiv(H*W, meta["BLOCK_HW"]))
        kernel_bn_forward[grid](y, x, weight, bias, epsilon, lock, sum_x, sum_xx, n_completed, *x.stride(),
                                N, C, H, W, BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW)

        #TODO.boyue, save normed_y, instead of y
        ctx.save_for_backward(y, weight, bias)
        return y

    @staticmethod
    def backward(ctx, out_grad, mean_grad, rstd_grad):
        pass
        return in_grad, None, weight_grad, bias_grad, None, None

    class TileAssist():
        def __init__(self, smem_size=40*1024, datum_size=4):
            self.smem_size = smem_size
            self.datum_size = datum_size
            self.BATCH_IDX, self.CHANNEL_IDX, self.SPATIAL_IDX = range(0, 3)
            self.trial_prio = [self.CHANNEL_IDX, self.SPATIAL_IDX, self.BATCH_IDX]
            self.merged_shape = None

        def calc_tiling(self, shape):
            """Calculate tiling for a given shape, assume dim size specified in order of
               (BATCH, CHANNEL, H, W)
            """
            assert(len(shape) == 4)
            self.merged_shape = shape[0], shape[1], shape[2]*shape[3]
            SHAPE_HW_MIN = 8
            tile_shape = [1, 1, SHAPE_HW_MIN]

            for dim_idx in self.trial_prio:
               solution = self.trial_split(dim_idx, tile_shape)
               if solution:
                   self.merged_shape = None
                   return solution

            self.merged_shape = None
            return tile_shape

        def trial_split(self, dim_idx, tile_shape):
            assert(dim_idx < len(tile_shape))
            assert(self.merged_shape and (dim_idx < len(self.merged_shape)))
            next_power2 = lambda x: 2**math.ceil(math.log2(x))
            tile_shape[dim_idx] = next_power2(self.merged_shape[dim_idx])

            product = lambda a, b: a*b
            tile_size = lambda shape: reduce(product, shape)*self.datum_size
            if tile_size(tile_shape) <= self.smem_size:
                return None

            while tile_size(tile_shape) > self.smem_size:
                tile_shape[dim_idx] = (tile_shape[dim_idx] // 2)

            return tile_shape

# --------------------------------------------------------------------------------------------------------------------
#   Main entry of the operator
# -------------------------------------------------------------------------------------------------------------------
def batch_norm(x, normalized_shape, weight, bias, eps=1e-5, cudnn_enable=True):
    return BatchNorm.apply(x, normalized_shape, weight, bias, eps, cudnn_enable)
