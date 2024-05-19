import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True)
def _update_adam(param, grad_param, m, v, t, lr, beta1, beta2, epsilon, idx):
    t += 1
    m_new = beta1 * m + (1 - beta1) * grad_param
    v_new = beta2 * v + (1 - beta2) * (grad_param ** 2)
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, t, m_new, v_new


class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self._initialized = False
        self.param_index = {}  # To map parameter names to indices
        self.param_shapes = [] # Store shapes for each parameter

    def initialize_parameters(self, param_shapes):
        if not self._initialized:
            self._initialized = True

            self.param_index = {name: i for i, name in enumerate(param_shapes.keys())}
            self.param_shapes = list(param_shapes.values())

            if not all(isinstance(shape, tuple) for shape in self.param_shapes):
                raise ValueError("param_shapes should be a list of tuples representing parameter shapes.")

            self.m = [np.zeros(shape, dtype=np.float32) for shape in self.param_shapes]
            self.v = [np.zeros(shape, dtype=np.float32) for shape in self.param_shapes]

    def update(self, param, grad_param, name):
        idx = self.param_index[name]
        param = param.astype(np.float32)
        grad_param = grad_param.astype(np.float32)
        
        param, self.t, m_new, v_new = _update_adam(param, grad_param, self.m[idx], self.v[idx], self.t, self.learning_rate, self.beta1, self.beta2, self.epsilon, idx)
        self.m[idx] = m_new
        self.v[idx] = v_new
        return param
