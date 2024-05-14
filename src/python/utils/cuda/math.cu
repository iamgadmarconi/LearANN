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

    __global__ void dotProductKernel(float* A, float* B, float* C, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            C[idx] = A[idx] * B[idx];
        }
    }
}