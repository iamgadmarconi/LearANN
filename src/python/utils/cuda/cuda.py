try:
    import pycuda.driver as cuda
    import pycuda.autoinit

    import numpy as np

    from utils.cuda.math import get_function

    CUDA_FUNCTIONS = get_function()

    # Helper function to perform matrix multiplication on GPU
    def gpu_matrix_mul(A, B, N):

        matrix_mul = CUDA_FUNCTIONS["matrix_mul"]
        
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

        dot_product = CUDA_FUNCTIONS["dot_product"]
        
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
        CUDA_FUNCTIONS[activation_type](x_gpu, y_gpu, np.int32(N), block=block_size, grid=grid_size)
        
        # Copy the result back to the host
        cuda.memcpy_dtoh(y, y_gpu)
        
        return y


    def gpu_activation_grad(activation_type, x, grad, N):
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
        CUDA_FUNCTIONS[activation_type](x_gpu, grad_gpu, y_gpu, np.int32(N), block=block_size, grid=grid_size)
        
        # Copy the result back to the host
        cuda.memcpy_dtoh(y, y_gpu)
        
        return y


    def gpu_mse_loss(pred, target, N):
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
        CUDA_FUNCTIONS["mse_loss"](pred_gpu, target_gpu, loss_gpu, np.int32(N), block=block_size, grid=grid_size)
        
        # Copy the result back to the host
        cuda.memcpy_dtoh(loss, loss_gpu)
        
        return loss[0]


    def gpu_mse_grad(pred, target, N):
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
        CUDA_FUNCTIONS["mse_grad"](pred_gpu, target_gpu, grad_gpu, np.int32(N), block=block_size, grid=grid_size)
        
        # Copy the result back to the host
        cuda.memcpy_dtoh(grad, grad_gpu)
        
        return grad


    def gpu_matrix_vector_mul(A, B):
        rows, cols = A.shape
        C = np.zeros((rows, B.shape[1]), dtype=np.float32)

        A = A.astype(np.float32)
        B = B.astype(np.float32)

        # Allocate memory on the GPU
        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        C_gpu = cuda.mem_alloc(C.nbytes)

        # Copy data to the GPU
        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)

        # Define block and grid sizes
        block_size = (256, 1, 1)
        grid_size = (rows // block_size[0] + 1, 1, 1)
        matrix_vector_mul = CUDA_FUNCTIONS["matrix_vector_mul"]
        matrix_vector_mul(A_gpu, B_gpu, C_gpu, np.int32(rows), np.int32(cols), block=block_size, grid=grid_size)

        # Copy the result back to the host
        cuda.memcpy_dtoh(C, C_gpu)
        return C

    def gpu_elementwise_mul(A, B):
        N = A.size
        C = np.zeros_like(A)

        A = A.astype(np.float32)
        B = B.astype(np.float32)

        A_gpu = cuda.mem_alloc(A.nbytes)
        B_gpu = cuda.mem_alloc(B.nbytes)
        C_gpu = cuda.mem_alloc(C.nbytes)

        cuda.memcpy_htod(A_gpu, A)
        cuda.memcpy_htod(B_gpu, B)

        block_size = 256
        grid_size = (N + block_size - 1) // block_size

        elementwise_mul = CUDA_FUNCTIONS["elementwise_mul"]
        elementwise_mul(A_gpu, B_gpu, C_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1, 1))

        cuda.memcpy_dtoh(C, C_gpu)
        return C


    def gpu_matrix_mul(A, B, transposed=False):
        if transposed:
            rows, cols = B.shape
            C = np.zeros((rows, A.shape[1]), dtype=np.float32)

            A = A.astype(np.float32)
            B = B.astype(np.float32)

            A_gpu = cuda.mem_alloc(A.nbytes)
            B_gpu = cuda.mem_alloc(B.nbytes)
            C_gpu = cuda.mem_alloc(C.nbytes)

            cuda.memcpy_htod(A_gpu, A)
            cuda.memcpy_htod(B_gpu, B)

            block_size = 256
            grid_size = (rows // block_size[0] + 1, 1, 1)

            matrix_mul_transpose = CUDA_FUNCTIONS["matrix_mul_transpose"]
            matrix_mul_transpose(A_gpu, B_gpu, C_gpu, np.int32(rows), np.int32(cols), block=block_size, grid=grid_size)

            cuda.memcpy_dtoh(C, C_gpu)
        else:
            rows, cols = A.shape
            C = np.zeros((rows, B.shape[1]), dtype=np.float32)

            A = A.astype(np.float32)
            B = B.astype(np.float32)

            A_gpu = cuda.mem_alloc(A.nbytes)
            B_gpu = cuda.mem_alloc(B.nbytes)
            C_gpu = cuda.mem_alloc(C.nbytes)

            cuda.memcpy_htod(A_gpu, A)
            cuda.memcpy_htod(B_gpu, B)

            block_size = 256
            grid_size = (rows // block_size[0] + 1, 1, 1)

            matrix_vector_mul = CUDA_FUNCTIONS["matrix_vector_mul"]
            matrix_vector_mul(A_gpu, B_gpu, C_gpu, np.int32(rows), np.int32(cols), block=block_size, grid=grid_size)

            cuda.memcpy_dtoh(C, C_gpu)
        
        return C


    def gpu_sum(input):
        N = input.size
        block_size = 256
        grid_size = (N + block_size - 1) // block_size

        output = np.zeros(grid_size, dtype=np.float32)
        input_gpu = cuda.mem_alloc(input.nbytes)
        output_gpu = cuda.mem_alloc(output.nbytes)

        cuda.memcpy_htod(input_gpu, input)  

        sum_kernel = CUDA_FUNCTIONS["sum"]
        shared_memory_size = block_size * np.dtype(np.float32).itemsize
        sum_kernel(input_gpu, output_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1, 1), shared=shared_memory_size)

        cuda.memcpy_dtoh(output, output_gpu)
        return output.sum()

except ImportError:
    Warning("PyCUDA is not installed. Please install it using `pip install pycuda`.")

