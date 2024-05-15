// math.cu
extern "C" {
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
    __global__ void elementwiseMulKernel(float* A, float* B, float* C, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            C[idx] = A[idx] * B[idx];
        }
    }
    __global__ void matrixMulTransposeKernel(float* A, float* B, float* C, int rows, int cols) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows) {
            float value = 0;
            for (int j = 0; j < cols; ++j) {
                value += A[j * rows + row] * B[j];
            }
            C[row] = value;
        }
    }
    __global__ void sumKernel(float* input, float* output, int N) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // Load input into shared memory
        sdata[tid] = (i < N) ? input[i] : 0;
        __syncthreads();

        // Perform reduction in shared memory
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write result for this block to global memory
        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
        }
    }
}