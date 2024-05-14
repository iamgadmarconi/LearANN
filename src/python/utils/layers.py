import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size, 1))
        
        # Predefine activation and gradient functions
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
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)
        grad_z = grad_output * self.activation_grad(self.z)
        grad_z = grad_z.reshape(self.output_size, -1)
        self.grad_weights = np.dot(grad_z, self.input.T)
        self.grad_biases = grad_z.sum(axis=1, keepdims=True)
        grad_input = np.dot(self.weights.T, grad_z)
        return grad_input

    # Activation functions and their gradients
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_grad(z):
        sig = 1 / (1 + np.exp(-z))
        return sig * (1 - sig)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_grad(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def linear_grad(z):
        return np.ones_like(z)
