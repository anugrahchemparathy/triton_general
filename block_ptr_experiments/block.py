import triton
import triton.language as tl
import torch

@triton.jit
def block_ptr_test(X_ptr, X_stride_0, X_stride_1):
    pid = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base = X_ptr,
        shape = (6, 3),
        strides = (X_stride_0, X_stride_1),
        offsets = (0, 0),
        block_shape = (4, 1),
        order = (1, 0) 
        # order = order of the strides in triton (meant for Hopper TMA optimizations) 
        # ref: https://www.mengyibai.com/p/order-in-triton-make-block-ptr/
    )


    x_block = tl.full((4, 1), 1.0, tl.float32)
    tl.store(x_block_ptr, x_block)


    # with tl.advance, we can move the block pointer in any direction
    # this moves the block pointer (+x, +y) slots in the memory, 
        # not (+x, +y) blocks
    x_block_ptr = tl.advance(base = x_block_ptr, offsets = (0, 2))
    x_block = tl.full((4, 1), 1.0, tl.float32)
    tl.store(x_block_ptr, x_block)


    # tl.advance also allows negative offsets and the new block pointer can overlap with a previous block pointer
    # also, the block pointer seems to automatically handle masking 
        # so perhaps it won't write out of bounds of the shape you passed in originally?
        # upon testing this, it seems not the case. you can still access beyond the original shape
        # I wonder if this has other issues, like accidentally writing to memory of a nearby tensor.
    x_block_ptr = tl.advance(base = x_block_ptr, offsets = (4, -2))
    x_block = tl.full((4, 1), 1.0, tl.float32)
    tl.store(x_block_ptr, x_block)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.zeros(8, 3).to(device)
    X_stride_0, X_stride_1 = X.stride()

    block_ptr_test[(1,)](X, X_stride_0, X_stride_1)
    print(X)
