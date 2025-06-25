import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def naive_softmax(x):
    """
    Args:
        - x (torch.Tensor): Input tensor of shape (M, N)
    
    Returns:
        - ret (torch.Tensor): Softmax of x, same shape as x
    
    Compute row-wise softmax of X using native pytorch
    We subtract the maximum element in order to avoid overflows. (Softmax is invariant to this shift)
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


"""
kernel summary:
- set BLOCK_SIZE as the next power of two greater than n_elements
- have each block process a few batches/rows of the input
- use a loop to process each row that the block is responsible for.
"""
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, batch_size, n_elements, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, batch_size, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_elements, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_elements
        mask = col_offsets < n_elements
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)





properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    """
    Args:
        - x (torch.Tensor): Batched input tensor of shape (batch_size, n_elements)
    Returns:
        - y (torch.Tensor): Batched softmax of x of shape (batch_size, n_elements)
    """
    batch_size, n_elements = x.shape
    y = torch.empty_like(x)

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_elements)

    # Another trick we can use is to ask the compiler to use more threads per row by increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural way so you don't have to come up with manual heuristics yourself.
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2



    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), batch_size, n_elements, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles() # finalizes the kernel, so we can get the metadata
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared


    # count how many blocks/programs we can fit in shared memory
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, batch_size)

    # Dispatch the kernel, creating as many blocks as we can fit in shared memory
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), batch_size, n_elements, BLOCK_SIZE, num_stages)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)