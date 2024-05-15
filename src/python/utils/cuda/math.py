from pycuda.compiler import SourceModule


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
    __global__ void matrixVectorMulKernel(float* A, float* B, float* C, int rows, int cols) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows) {
            float value = 0;
            for (int j = 0; j < cols; ++j) {
                value += A[row * cols + j] * B[j];
            }
            C[row] = value;
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
    matrixVectorMulKernel = mod.get_function("matrixVectorMulKernel")

    return {
        "matrix_mul": matrix_mul,
        "matrix_vector_mul": matrixVectorMulKernel,
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