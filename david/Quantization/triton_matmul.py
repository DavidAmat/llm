import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    A simple matrix multiplication kernel: (M x K) * (K x N) -> (M x N).

    A_ptr: float32 pointer to matrix A
    B_ptr: float32 pointer to matrix B
    C_ptr: float32 pointer to matrix C (output)
    M, N, K: dimensions of the matrices
    stride_am, stride_ak, etc.: strides to index A, B, C in memory
    BLOCK_M, BLOCK_N, BLOCK_K: tile/block sizes for the Triton kernel
    """

    # Program ID: each "program" processes one tile of the output matrix.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Each block of threads computes a [BLOCK_M x BLOCK_N] sub-matrix of C.
    # Compute the row and col bounds for this tile.
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # Create a pointer for loading/storing values in C.
    # We'll handle boundary conditions carefully below.
    # Range of row indices and col indices this program is responsible for:
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = col_start + tl.arange(0, BLOCK_N)

    # Initialize accumulator for our block of C. We'll store partial sums here.
    c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in BLOCK_K increments
    # We assume (K is multiple of BLOCK_K) for simplicity. If not, you'd handle remainder logic.
    for k_block_start in range(0, K, BLOCK_K):
        # Offsets for A and B
        # A is [M x K], B is [K x N]
        # We'll load a [BLOCK_M x BLOCK_K] sub-tile from A and
        # a [BLOCK_K x BLOCK_N] sub-tile from B.
        offs_k = k_block_start + tl.arange(0, BLOCK_K)

        # Create 2D mesh for the sub-block
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Load the sub-blocks
        a_block = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b_block = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Multiply and accumulate
        c += tl.dot(a_block, b_block)

    # Now we store the [BLOCK_M x BLOCK_N] tile results into C.
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    # We only write where (offs_m < M) & (offs_n < N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def triton_matmul(A, B, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    """
    Multiply two float32 matrices A and B using a Triton kernel and return C = A * B.
    
    A: [M x K]
    B: [K x N]
    BLOCK_M, BLOCK_N, BLOCK_K: tile sizes for the Triton kernel
    """
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "This MVP supports float32 only"
    assert A.is_cuda and B.is_cuda, "Tensors must be on GPU"

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions"

    # Create output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Grid: how many program blocks we launch in each dimension
    # We'll need enough blocks to cover M and N in increments of BLOCK_M / BLOCK_N
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)

    # Launch the Triton kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return C


if __name__ == "__main__":
    # Basic test: multiply random matrices on an RTX 4090 or similar GPU
    device = "cuda"  # or "cuda:0"

    # Create random input data
    # For MVP, let's keep them fairly small, but you can scale up.
    M, K, N = 256, 128, 256
    A_torch = torch.randn((M, K), device=device, dtype=torch.float32)
    B_torch = torch.randn((K, N), device=device, dtype=torch.float32)

    # Run Triton matmul
    C_triton = triton_matmul(A_torch, B_torch)

    # Compare with PyTorch matmul for correctness
    C_ref = A_torch @ B_torch
    max_diff = (C_triton - C_ref).abs().max()

    print(f"Triton matmul result shape = {C_triton.shape}")
    print(f"Reference matmul result shape = {C_ref.shape}")
    print(f"Max difference between Triton and PyTorch = {max_diff.item():.6f}")
