import numba
import numpy as np


@numba.jit(nopython=True, fastmath=True)
def _update_adam(param, grad_param, m, v, t, lr, beta1, beta2, epsilon, idx):
    grad_param = grad_param.astype(np.float32)
    t += 1
    m[idx] = beta1 * m[idx] + (1 - beta1) * grad_param
    v[idx] = beta2 * v[idx] + (1 - beta2) * (grad_param ** 2)
    m_hat = m[idx] / (1 - beta1 ** t)
    v_hat = v[idx] / (1 - beta2 ** t)
    param -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, t


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

            # Create an index mapping
            self.param_index = {name: i for i, name in enumerate(param_shapes.keys())}
            self.param_shapes = list(param_shapes.values())

            # Ensure param_shapes_list is a list of tuples
            if not all(isinstance(shape, tuple) for shape in self.param_shapes):
                raise ValueError("param_shapes should be a list of tuples representing parameter shapes.")

            # Initialize m and v with the correct shapes
            self.m = [np.zeros(shape, dtype=np.float32) for shape in self.param_shapes]
            self.v = [np.zeros(shape, dtype=np.float32) for shape in self.param_shapes]

    def update(self, param, grad_param, name):
        if not self._initialized:
            raise ValueError("Optimizer parameters not initialized. Call `initialize_parameters` first.")
        idx = self.param_index[name]
        param = param.astype(np.float32)
        grad_param = grad_param.astype(np.float32)
        param, self.t = _update_adam(param, grad_param, self.m, self.v, self.t, self.learning_rate, self.beta1, self.beta2, self.epsilon, idx)
        return param
