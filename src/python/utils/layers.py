import numpy as np
import numba


from utils.cuda.cuda import *


class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
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
        return (z > 0).astype(float)

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


class LSTMCell(Layer):
    def __init__(self, input_size, hidden_size, activation='relu', gate_activation='sigmoid'):
        super().__init__(input_size, hidden_size, activation)
        self.hidden_size = hidden_size
        self.weights = np.random.randn(4 * hidden_size, input_size + hidden_size).astype(np.float32) * np.sqrt(2. / (input_size + hidden_size))
        self.biases = np.zeros((4 * hidden_size, 1), dtype=np.float32)
        self._init_gate_activations(gate_activation)


    def _init_gate_activations(self, gate_activation):
        if gate_activation == 'relu':
            self.gate_activation = self.relu
            self.gate_activation_grad = self.relu_grad
        elif gate_activation == 'sigmoid':
            self.gate_activation = self.sigmoid
            self.gate_activation_grad = self.sigmoid_grad
        elif gate_activation == 'tanh':
            self.gate_activation = self.tanh
            self.gate_activation_grad = self.tanh_grad
        else:
            self.gate_activation = self.linear
            self.gate_activation_grad = self.linear_grad

    def forward(self, x, h_prev, c_prev):
        batch_size = x.shape[1]

        combined = np.vstack((x, h_prev))
        gates = np.dot(self.weights, combined) + self.biases

        i_gate = self.sigmoid(gates[:self.hidden_size])
        f_gate = self.sigmoid(gates[self.hidden_size:self.hidden_size*2])
        o_gate = self.sigmoid(gates[self.hidden_size*2:self.hidden_size*3])
        g_gate = self.tanh(gates[self.hidden_size*3:])

        c = f_gate * c_prev + i_gate * g_gate
        h = o_gate * self.activation(c)

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
        batch_size = dh_next.shape[1]

        # Adjust dh_next and dc_next if their dimensions do not match self.hidden_size
        if dh_next.shape[0] != self.hidden_size:
            dh_next = np.resize(dh_next, (self.hidden_size, batch_size))
        if dc_next.shape[0] != self.hidden_size:
            dc_next = np.resize(dc_next, (self.hidden_size, batch_size))

        if dh_next.ndim == 1:
            dh_next = dh_next.reshape(self.hidden_size, -1)
        if dc_next.ndim == 1:
            dc_next = dc_next.reshape(self.hidden_size, -1)

        do = dh_next * np.tanh(self.c)
        dc = dc_next + dh_next * self.o_gate * (1 - np.tanh(self.c) ** 2)
        di = dc * self.g_gate
        dg = dc * self.i_gate
        df = dc * self.c_prev

        di_input = di * self.sigmoid(self.i_gate)
        df_input = df * self.sigmoid(self.f_gate)
        do_input = do * self.tanh(self.o_gate)
        dg_input = dg * self.sigmoid(self.g_gate)

        d_combined = np.vstack((di_input, df_input, do_input, dg_input))

        self.grad_weights = np.dot(d_combined, self.input.T)
        self.grad_biases = d_combined.sum(axis=1, keepdims=True)
        d_combined_input = np.dot(self.weights.T, d_combined)

        dx = d_combined_input[:self.input_size]
        dh_prev = d_combined_input[self.input_size:self.input_size + self.hidden_size]
        dc_prev = dc * self.f_gate

        return dx, dh_prev, dc_prev
