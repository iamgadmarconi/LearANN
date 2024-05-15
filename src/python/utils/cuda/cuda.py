import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

from utils.utils import get_base_path


def get_function():
    # kernel_path = os.path.join(get_base_path(), 'cuda', 'math.ptx')
    # with open(kernel_path, "rb") as f:
    #     kernel_code = f.read()
    # mod = SourceModule(kernel_code.decode('utf-8'))
    mod = SourceModule("""
    __global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < N && col < N) {
            float value = 0;
            for (int k = 0; k < N; ++k) {
                value += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = value;
        }
    }

    __global__ void dotProductKernel(float* A, float* B, float* C, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            C[idx] = A[idx] * B[idx];
        }
    }
    """)
    matrix_mul = mod.get_function("matrixMulKernel")
    dot_product = mod.get_function("dotProductKernel")
    return {"matrix_mul": matrix_mul, "dot_product": dot_product}

# Helper function to perform matrix multiplication on GPU
def gpu_matrix_mul(A, B, N):
    function_dict = get_function()
    matrix_mul = function_dict["matrix_mul"]
    
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    # Allocate memory on the GPU
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(C.nbytes)

    # Copy data to the GPU
    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)

    # Define block and grid sizes
    block_size = (16, 16, 1)
    grid_size = (N // block_size[0] + 1, N // block_size[1] + 1, 1)

    # Launch the kernel
    matrix_mul(A_gpu, B_gpu, C_gpu, np.int32(N), block=block_size, grid=grid_size)

    # Copy the result back to the host
    cuda.memcpy_dtoh(C, C_gpu)

    return C

# Helper function to perform dot product on GPU
def gpu_dot_product(A, B, N):
    function_dict = get_function()
    dot_product = function_dict["dot_product"]
    
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    C = np.zeros(N, dtype=np.float32)

    # Allocate memory on the GPU
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(C.nbytes)

    # Copy data to the GPU
    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)

    # Define block and grid sizes
    block_size = (256, 1, 1)
    grid_size = (N // block_size[0] + 1, 1, 1)

    # Launch the kernel
    dot_product(A_gpu, B_gpu, C_gpu, np.int32(N), block=block_size, grid=grid_size)

    # Copy the result back to the host
    cuda.memcpy_dtoh(C, C_gpu)

    return C

# Test the GPU matrix multiplication and dot product
def test_mat_mult():
    N = 32  # Size of the matrix
    A = np.random.randn(N, N)
    B = np.random.randn(N, N)
    
    C = gpu_matrix_mul(A, B, N)
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Matrix C (A * B):\n", C)

    A_vec = np.random.randn(N)
    B_vec = np.random.randn(N)
    
    D = gpu_dot_product(A_vec, B_vec, N)
    print("Vector A:\n", A_vec)
    print("Vector B:\n", B_vec)
    print("Vector D (A . B):\n", D)
