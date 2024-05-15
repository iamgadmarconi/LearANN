import numpy as np


from utils.cuda.cuda import *


class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size, 1), dtype=np.float32)
        
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


class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights = np.random.randn(4 * hidden_size, input_size + hidden_size).astype(np.float32) * np.sqrt(2. / (input_size + hidden_size))
        self.biases = np.zeros((4 * hidden_size, 1), dtype=np.float32)

    def forward(self, x, h_prev, c_prev):
        batch_size = x.shape[1]

        assert x.shape[0] == self.input_size, f"Expected input size {self.input_size}, got {x.shape[0]}"
        assert h_prev.shape == (self.hidden_size, batch_size), f"Expected hidden state shape {(self.hidden_size, batch_size)}, got {h_prev.shape}"
        assert c_prev.shape == (self.hidden_size, batch_size), f"Expected cell state shape {(self.hidden_size, batch_size)}, got {c_prev.shape}"

        combined = np.vstack((x, h_prev))  # Shape: (input_size + hidden_size, batch_size)
        assert combined.shape == (self.input_size + self.hidden_size, batch_size), f"Expected combined shape {(self.input_size + self.hidden_size, batch_size)}, got {combined.shape}"

        gates = np.dot(self.weights, combined) + self.biases
        assert gates.shape == (4 * self.hidden_size, batch_size), f"Expected gates shape {(4 * self.hidden_size, batch_size)}, got {gates.shape}"

        i_gate = self.sigmoid(gates[:self.hidden_size])
        f_gate = self.sigmoid(gates[self.hidden_size:self.hidden_size*2])
        o_gate = self.sigmoid(gates[self.hidden_size*2:self.hidden_size*3])
        g_gate = np.tanh(gates[self.hidden_size*3:])

        assert i_gate.shape == (self.hidden_size, batch_size), f"Expected i_gate shape {(self.hidden_size, batch_size)}, got {i_gate.shape}"
        assert f_gate.shape == (self.hidden_size, batch_size), f"Expected f_gate shape {(self.hidden_size, batch_size)}, got {f_gate.shape}"
        assert o_gate.shape == (self.hidden_size, batch_size), f"Expected o_gate shape {(self.hidden_size, batch_size)}, got {o_gate.shape}"
        assert g_gate.shape == (self.hidden_size, batch_size), f"Expected g_gate shape {(self.hidden_size, batch_size)}, got {g_gate.shape}"

        c = f_gate * c_prev + i_gate * g_gate
        h = o_gate * np.tanh(c)

        self.input = combined
        self.h_prev = h_prev
        self.c_prev = c_prev
        self.i_gate = i_gate
        self.f_gate = f_gate
        self.o_gate = o_gate
        self.g_gate = g_gate
        self.c = c

        return h, c

    def backward(self, dh_next, dc_next):
        do = dh_next * np.tanh(self.c)
        dc = dc_next + dh_next * self.o_gate * (1 - np.tanh(self.c) ** 2)
        di = dc * self.g_gate
        dg = dc * self.i_gate
        df = dc * self.c_prev

        di_input = di * self.i_gate * (1 - self.i_gate)
        df_input = df * self.f_gate * (1 - self.f_gate)
        do_input = do * self.o_gate * (1 - self.o_gate)
        dg_input = dg * (1 - self.g_gate ** 2)

        d_combined = np.vstack((di_input, df_input, do_input, dg_input))

        self.grad_weights = np.dot(d_combined, self.input.T)
        self.grad_biases = d_combined.sum(axis=1, keepdims=True)
        d_combined_input = np.dot(self.weights.T, d_combined)

        dx = d_combined_input[:self.input_size]
        dh_prev = d_combined_input[self.input_size:]

        return dx, dh_prev, dc * self.f_gate

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
