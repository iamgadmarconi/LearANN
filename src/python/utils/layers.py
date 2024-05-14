import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size, 1))
        self.activation = activation

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.input = x
        self.z = np.dot(self.weights, x) + self.biases
        return self.apply_activation(self.z)

    def apply_activation(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            return z

    def backward(self, grad_output):
        if grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)
        if self.activation == 'relu':
            grad_z = grad_output * (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = self.apply_activation(self.z)
            grad_z = grad_output * sig * (1 - sig)
        elif self.activation == 'tanh':
            tanh = self.apply_activation(self.z)
            grad_z = grad_output * (1 - tanh ** 2)
        else:
            grad_z = grad_output

        grad_z = grad_z.reshape(self.output_size, -1)
        self.grad_weights = np.dot(grad_z, self.input.T)
        self.grad_biases = grad_z.sum(axis=1, keepdims=True)
        grad_input = np.dot(self.weights.T, grad_z)
        return grad_input