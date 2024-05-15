import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

from utils.utils import get_base_path


def get_function():
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

    __global__ void reluKernel(float* x, float* y, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            y[idx] = max(0.0f, x[idx]);
        }
    }

    __global__ void reluGradKernel(float* x, float* grad, float* y, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            y[idx] = (x[idx] > 0) ? grad[idx] : 0.0f;
        }
    }

    __global__ void sigmoidKernel(float* x, float* y, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            y[idx] = 1.0f / (1.0f + expf(-x[idx]));
        }
    }

    __global__ void sigmoidGradKernel(float* x, float* grad, float* y, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            float sigmoid_val = 1.0f / (1.0f + expf(-x[idx]));
            y[idx] = grad[idx] * sigmoid_val * (1 - sigmoid_val);
        }
    }

    __global__ void tanhKernel(float* x, float* y, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            y[idx] = tanhf(x[idx]);
        }
    }

    __global__ void tanhGradKernel(float* x, float* grad, float* y, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            float tanh_val = tanhf(x[idx]);
            y[idx] = grad[idx] * (1 - tanh_val * tanh_val);
        }
    }

    __global__ void mseLossKernel(float* pred, float* target, float* loss, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            float diff = pred[idx] - target[idx];
            atomicAdd(loss, diff * diff / N);
        }
    }

    __global__ void mseGradKernel(float* pred, float* target, float* grad, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            grad[idx] = 2 * (pred[idx] - target[idx]) / N;
        }
    }
    """)
    matrix_mul = mod.get_function("matrixMulKernel")
    dot_product = mod.get_function("dotProductKernel")
    relu = mod.get_function("reluKernel")
    relu_grad = mod.get_function("reluGradKernel")
    sigmoid = mod.get_function("sigmoidKernel")
    sigmoid_grad = mod.get_function("sigmoidGradKernel")
    tanh = mod.get_function("tanhKernel")
    tanh_grad = mod.get_function("tanhGradKernel")
    mse_loss = mod.get_function("mseLossKernel")
    mse_grad = mod.get_function("mseGradKernel")
    return {
        "matrix_mul": matrix_mul,
        "dot_product": dot_product,
        "relu": relu,
        "relu_grad": relu_grad,
        "sigmoid": sigmoid,
        "sigmoid_grad": sigmoid_grad,
        "tanh": tanh,
        "tanh_grad": tanh_grad,
        "mse_loss": mse_loss,
        "mse_grad": mse_grad
    }

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


def gpu_activation(activation_type, x, N):
    function_dict = get_function()
    x = x.astype(np.float32)
    y = np.zeros_like(x)
    
    # Allocate memory on the GPU
    x_gpu = cuda.mem_alloc(x.nbytes)
    y_gpu = cuda.mem_alloc(y.nbytes)
    
    # Copy data to the GPU
    cuda.memcpy_htod(x_gpu, x)
    
    # Define block and grid sizes
    block_size = (256, 1, 1)
    grid_size = (N // block_size[0] + 1, 1, 1)
    
    # Launch the kernel
    function_dict[activation_type](x_gpu, y_gpu, np.int32(N), block=block_size, grid=grid_size)
    
    # Copy the result back to the host
    cuda.memcpy_dtoh(y, y_gpu)
    
    return y


def gpu_activation_grad(activation_type, x, grad, N):
    function_dict = get_function()
    x = x.astype(np.float32)
    grad = grad.astype(np.float32)
    y = np.zeros_like(x)
    
    # Allocate memory on the GPU
    x_gpu = cuda.mem_alloc(x.nbytes)
    grad_gpu = cuda.mem_alloc(grad.nbytes)
    y_gpu = cuda.mem_alloc(y.nbytes)
    
    # Copy data to the GPU
    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(grad_gpu, grad)
    
    # Define block and grid sizes
    block_size = (256, 1, 1)
    grid_size = (N // block_size[0] + 1, 1, 1)
    
    # Launch the kernel
    function_dict[activation_type](x_gpu, grad_gpu, y_gpu, np.int32(N), block=block_size, grid=grid_size)
    
    # Copy the result back to the host
    cuda.memcpy_dtoh(y, y_gpu)
    
    return y


def gpu_mse_loss(pred, target, N):
    function_dict = get_function()
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    loss = np.zeros(1, dtype=np.float32)
    
    # Allocate memory on the GPU
    pred_gpu = cuda.mem_alloc(pred.nbytes)
    target_gpu = cuda.mem_alloc(target.nbytes)
    loss_gpu = cuda.mem_alloc(loss.nbytes)
    
    # Copy data to the GPU
    cuda.memcpy_htod(pred_gpu, pred)
    cuda.memcpy_htod(target_gpu, target)
    
    # Define block and grid sizes
    block_size = (256, 1, 1)
    grid_size = (N // block_size[0] + 1, 1, 1)
    
    # Launch the kernel
    function_dict["mse_loss"](pred_gpu, target_gpu, loss_gpu, np.int32(N), block=block_size, grid=grid_size)
    
    # Copy the result back to the host
    cuda.memcpy_dtoh(loss, loss_gpu)
    
    return loss[0]


def gpu_mse_grad(pred, target, N):
    function_dict = get_function()
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    grad = np.zeros_like(pred)
    
    # Allocate memory on the GPU
    pred_gpu = cuda.mem_alloc(pred.nbytes)
    target_gpu = cuda.mem_alloc(target.nbytes)
    grad_gpu = cuda.mem_alloc(grad.nbytes)
    
    # Copy data to the GPU
    cuda.memcpy_htod(pred_gpu, pred)
    cuda.memcpy_htod(target_gpu, target)
    
    # Define block and grid sizes
    block_size = (256, 1, 1)
    grid_size = (N // block_size[0] + 1, 1, 1)
    
    # Launch the kernel
    function_dict["mse_grad"](pred_gpu, target_gpu, grad_gpu, np.int32(N), block=block_size, grid=grid_size)
    
    # Copy the result back to the host
    cuda.memcpy_dtoh(grad, grad_gpu)
    
    return grad


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
