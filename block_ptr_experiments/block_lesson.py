import triton
import triton.language as tl
import torch
"""
Written by: Anugrah Chemparathy

A quick lesson on using block pointers in triton for future reference.
"""


@triton.jit
def kernel_1(X_ptr, X_stride_0, X_stride_1):
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


    x_block = tl.full((4, 1), 2.0, tl.float32)
    tl.store(x_block_ptr, x_block)



    # with tl.advance, we can move the block pointer in any direction
    # this moves the block pointer (+x, +y) slots in the memory, 
        # not (+x, +y) blocks
    x_block_ptr = tl.advance(base = x_block_ptr, offsets = (0, 1))
    x_block = tl.full((4, 1), 2.0, tl.float32)
    tl.store(x_block_ptr, x_block)

    x_block_ptr = tl.advance(base = x_block_ptr, offsets = (0, 1))
    x_block = tl.full((4, 1), 3.0, tl.float32)
    tl.store(x_block_ptr, x_block)


    # 2 points here:
    # tl.advance also allows negative offsets and the new block pointer can overlap with a previous block pointer
    # if you pass in a tuple for boundary check, it will not write out of bounds of the shape you passed in originally
        # you should be very careful you pass in a tuple (0,) and not (0)
        # the boundary check takes in tuples of dimensions that you want to check e.g. (0,), or (0,1), ...
    x_block_ptr = tl.advance(base = x_block_ptr, offsets = (4, -2))
    x_block = tl.full((4, 1), 4.0, tl.float32)
    # tl.store(x_block_ptr, x_block)
    tl.store(x_block_ptr, x_block, boundary_check = (0,))


@triton.jit
def kernel_2(X_ptr, X_stride_0, X_stride_1):
    pid = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base = X_ptr,
        shape = (6, 3),
        strides = (X_stride_0, X_stride_1),
        offsets = (4, 0),
        block_shape = (4, 1),
        order = (1, 0) 
    )


    # also when loading, you can pass in a boundary_check tuple and padding_option
        # together these allow you to ensure you aren't overstepping the shape you 
        # passed in and the extra slots get padded with zeros or nans, depending
        # on the padding_option you pass in.
    x_block_ptr = tl.advance(base = x_block_ptr, offsets = (0, 1))
    x_block = tl.load(x_block_ptr, boundary_check=(0,), padding_option = "zero")
    
    # be careful with this padding though! it still exists, and will get written if
        # aren't careful!
    x_block_ptr = tl.advance(base = x_block_ptr, offsets = (-1, 0))
    tl.store(x_block_ptr, x_block, boundary_check = (0,))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.ones(7, 3).to(device)
    X[6,:] = 9.0
    X_stride_0, X_stride_1 = X.stride()

    print(X)
    kernel_1[(1,)](X, X_stride_0, X_stride_1)
    print(X)
    kernel_2[(1,)](X, X_stride_0, X_stride_1)
    print(X)
