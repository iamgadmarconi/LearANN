import numpy as np
import numba
from numba.experimental import jitclass


spec_adagrad = [
    ('learning_rate', numba.float32),
    ('epsilon', numba.float32),
    ('cumulative_grads', numba.types.DictType(numba.types.unicode_type, numba.float32[:, :])),
    ('_initialized', numba.boolean)
]


@jitclass(spec=spec_adagrad)
class Adagrad:

    def __init__(self, lr, epsilon=1e-8):
        self.learning_rate = lr
        self.epsilon = epsilon
        self.cumulative_grads = {}

    def initialize_parameters(self, params):
        if not self._initialized:
            self._initialized = True
            for name, shape in params.items():
                self.cumulative_grads[name] = np.zeros(shape)

    def update(self, param, grad_param, name):
        self.cumulative_grads[name] += grad_param ** 2
        param -= self.learning_rate * grad_param / (np.sqrt(self.cumulative_grads[name]) + self.epsilon)
        return param
