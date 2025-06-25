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

def get_device_stats():
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]
    target = triton.runtime.driver.active.get_current_target()

    device_stats = {
        "NUM_SM": NUM_SM,
        "NUM_REGS": NUM_REGS,
        "SIZE_SMEM": SIZE_SMEM,
        "WARP_SIZE": WARP_SIZE,
        "target": target
    }
    return device_stats


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, batch_size, n_elements, 
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    
    program_id = tl.program_id(axis=0)
    n_programs = tl.num_programs(0)

    for row_idx in tl.range(program_id, batch_size, n_programs, num_stages = num_stages):
        input_row_start_ptr = input_ptr + row_idx * input_row_stride
        output_row_start_ptr = output_ptr + row_idx * output_row_stride

        input_row_ptrs = input_row_start_ptr + tl.arange(0, BLOCK_SIZE)
        output_row_ptrs = output_row_start_ptr + tl.arange(0, BLOCK_SIZE)

        input_row = tl.load(input_row_ptrs)
        input_row_minus_max = input_row - tl.max(input_row, axis=0)
        numerator = tl.exp(input_row_minus_max)
        denominator = tl.sum(numerator, axis = 0)
        softmax_output = numerator / denominator

        tl.store(output_row_ptrs, softmax_output)






def softmax(x):
    """
    Args:
        - x (torch.Tensor): Batched input tensor of shape (batch_size, n_elements)
    Returns:
        - y (torch.Tensor): Batched softmax of x of shape (batch_size, n_elements)
    """
    batch_size, n_elements = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = n_elements
    num_stages = 2

    kernel = softmax_kernel
    kernel[(32, 1, 1)](y, x, x.stride(0), y.stride(0), batch_size, n_elements, BLOCK_SIZE, num_stages)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(10, 128, device=DEVICE)
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    print(f"triton vs torch:")
    print(f"triton: {y_triton[0,:5]}")
    print(f"torch:  {y_torch[0,:5]}")