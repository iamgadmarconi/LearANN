import numpy as np
import numba
from numba import jit
from numba.experimental import jitclass

from utils.layers.dense import Layer


spec_lstm = [
    ('hidden_size', numba.int32),
    ('weights', numba.float32[:, :]),
    ('weight', numba.float32[:, :]),
    ('biases', numba.float32[:, :]),
    ('gate_activation', numba.types.FunctionType(numba.float32(numba.float32)), numba.types.FunctionType(numba.float32(numba.float32))),
    ('i_gate', numba.float32[:]),
    ('f_gate', numba.float32[:]),
    ('o_gate', numba.float32[:]),
    ('g_gate', numba.float32[:]),
    ('c', numba.float32[:]),
    ('input', numba.float32[:, :]),
    ('h_prev', numba.float32[:, :]),
    ('c_prev', numba.float32[:, :]),
    ('grad_weights', numba.float32[:, :]),
    ('grad_biases', numba.float32[:, :])
]


# @jitclass(spec_lstm)
class LSTMCell(Layer):
    def __init__(self, input_size, hidden_size, activation='relu', gate_activation='sigmoid'):
        super().__init__(input_size, hidden_size, activation)
        self.hidden_size = hidden_size
        self.weights = np.random.randn(4 * hidden_size, input_size + hidden_size).astype(np.float32) * np.sqrt(2. / (input_size + hidden_size))
        self.weight = self.weights.astype(np.float32)
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

        combined = np.vstack((h_prev, x))  # Note the order change to match dimensions
        gates = np.dot(self.weights, combined) + self.biases

        i_gate = self.sigmoid(gates[:self.hidden_size])
        f_gate = self.sigmoid(gates[self.hidden_size:self.hidden_size*2])
        o_gate = self.sigmoid(gates[self.hidden_size*2:self.hidden_size*3])
        g_gate = self.tanh(gates[self.hidden_size*3:])

        c = f_gate * c_prev + i_gate * g_gate
        h = o_gate * self.tanh(c)

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

        do = dh_next * self.activation_grad(np.tanh(self.c))  # Assuming tanh is used for output activation
        dc = dc_next + dh_next * self.o_gate * self.activation_grad(np.tanh(self.c))

        di = dc * self.g_gate
        df = dc * self.c_prev
        dg = dc * self.i_gate

        di_input = di * self.sigmoid_grad(self.i_gate)
        df_input = df * self.sigmoid_grad(self.f_gate)
        do_input = do * self.sigmoid_grad(self.o_gate)
        dg_input = dg * self.tanh_grad(self.g_gate)

        d_combined = np.vstack((di_input, df_input, do_input, dg_input))

        self.grad_weights = np.dot(d_combined, self.input.T)
        self.grad_biases = np.sum(d_combined, axis=1, keepdims=True)
        d_combined_input = np.dot(self.weights.T, d_combined)

        dx = d_combined_input[self.hidden_size:, :]
        dh_prev = d_combined_input[:self.hidden_size, :]
        dc_prev = dc * self.f_gate

        return dx, dh_prev, dc_prev
