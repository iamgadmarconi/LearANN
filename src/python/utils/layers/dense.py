import numpy as np
import numba

from utils.cuda.cuda import *


class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.weight = self.weights.astype(np.float32)
        self.biases = np.zeros((output_size, 1), dtype=np.float32)
        
        # Predefine activation and gradient functions
        self._init_activation(activation)

    def _init_activation(self, activation):
        if activation == 'relu':
            self.activation = self.relu
            self.activation_grad = self.relu_grad
        elif activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_grad = self.sigmoid_grad
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_grad = self.tanh_grad
        else:
            self.activation = self.linear
            self.activation_grad = self.linear_grad

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.input = x
        self.z = np.dot(self.weights, x) + self.biases
        return self.activation(self.z)

    def backward(self, grad_output):
        # Ensure grad_output is 2D for matrix operations
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)

        # Calculate the gradient of the activation function
        grad_z = grad_output * self.activation_grad(self.z)

        # Calculate gradients for weights and biases
        self.grad_weights = np.dot(grad_z, self.input.T)
        self.grad_biases = grad_z.sum(axis=1, keepdims=True)

        # Propagate the gradient backwards
        grad_input = np.dot(self.weights.T, grad_z)
        return grad_input


    # Activation functions and their gradients
    @staticmethod
    @numba.njit
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    @numba.njit
    def relu_grad(z):
        return np.where(z > 0, 1.0, 0.0)

    @staticmethod
    @numba.njit
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    @numba.njit
    def sigmoid_grad(z):
        sig = 1 / (1 + np.exp(-z))
        return sig * (1 - sig)

    @staticmethod
    @numba.njit
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    @numba.njit
    def tanh_grad(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    @numba.njit
    def linear(z):
        return z

    @staticmethod
    @numba.njit
    def linear_grad(z):
        return np.ones_like(z)


class GPULayer(Layer):
    def __init__(self, input_size, output_size, activation='relu'):
        super().__init__(input_size, output_size, activation)

        self.activation_type = activation

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.input = x
        self.z = gpu_matrix_vector_mul(self.weights, x)
        self.z += self.biases
        return gpu_activation(self.activation_type, self.z, self.z.size)

    def backward(self, grad_output):
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)
        grad_z = gpu_elementwise_mul(grad_output, self.activation_grad(self.z))
        self.grad_weights = gpu_matrix_vector_mul(grad_z, self.input.T)
        self.grad_biases = gpu_sum(grad_z)
        grad_input = gpu_matrix_vector_mul(self.weights.T, grad_z)
        return grad_input


