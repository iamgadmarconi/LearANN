import numpy as np
import numba
from numba.experimental import jitclass
from numba.typed import Dict

spec = [
    ('lr', numba.float32),
    ('_initialized', numba.boolean)
]

spec_adagrad = [
    ('learning_rate', numba.float32),
    ('epsilon', numba.float32),
    ('cumulative_grads', numba.types.DictType(numba.types.unicode_type, numba.float32[:, :])),
    ('_initialized', numba.boolean)
]

# @jitclass(spec=spec)
class Optimizer:
    def __init__(self, lr):
        self.learning_rate = lr

        self._initialized = False

    def update(self, param, grad_param, name):
        raise NotImplementedError

    def initialize_parameters(self, params):
        raise NotImplementedError




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

class GradientDescent(Optimizer):

    def __init__(self, lr):
        super().__init__(lr)

    @numba.njit
    def update(self, param, grad_param, name):
        param -= self.learning_rate * grad_param
        return param
