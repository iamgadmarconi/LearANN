import numpy as np
from numba import jit, prange
from line_profiler import profile


@jit(nopython=True, fastmath=True, cache=True, nogil=True, parallel=True)
def _update_adam(param, grad_param, m, v, t, lr, beta1, beta2, epsilon):
    t += 1
    
    # In-place updates and efficient calculations
    beta1_t = beta1 ** t
    beta2_t = beta2 ** t
    
    for i in prange(param.shape[0]):
        m[i] = beta1 * m[i] + (1 - beta1) * grad_param[i]
        v[i] = beta2 * v[i] + (1 - beta2) * (grad_param[i] ** 2)
        
        m_hat = m[i] / (1 - beta1_t)
        v_hat = v[i] / (1 - beta2_t)
        
        param[i] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return param, t, m, v

# @jit(nopython=True, fastmath=True, cache=True, nogil=True, parallel=True)
# def _update_adam(param, grad_param, m, v, t, lr, beta1, beta2, epsilon):
#     t += 1
#     # In-place multiplication updates
#     m[:] = beta1 * m + (1 - beta1) * grad_param
#     v[:] = beta2 * v + (1 - beta2) * (grad_param ** 2)

#     m_hat = m / (1 - beta1 ** t)
#     v_hat = v / (1 - beta2 ** t)

#     # In-place subtraction for parameter update
#     param[:] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
#     return param, t, m, v

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

    @profile
    def update(self, param, grad_param, name):
        idx = self.param_index[name]
        
        # Ensure the parameters are already in the correct type
        param = param.astype(np.float32, copy=False)
        grad_param = grad_param.astype(np.float32, copy=False)

        param, self.t, m_new, v_new = _update_adam(param, grad_param, self.m[idx], self.v[idx], self.t, self.learning_rate, self.beta1, self.beta2, self.epsilon)
        self.m[idx] = m_new
        self.v[idx] = v_new
        return param

# For profiling the Adam optimizer. 
# Currently, the _update_params function is severly bottleneckeing the performance of the optimizer.
# The for-loop is used for parallelization, and provides ~100% speedup on a 10-core CPU.
if __name__ == "__main__":
    param_shapes = {'w1': (3, 3), 'b1': (3,), 'w2': (3, 3), 'b2': (3,)}
    optimizer = Adam(lr=0.001)
    optimizer.initialize_parameters(param_shapes)

    params = {
        'w1': np.random.randn(3, 3).astype(np.float32),
        'b1': np.random.randn(3).astype(np.float32),
        'w2': np.random.randn(3, 3).astype(np.float32),
        'b2': np.random.randn(3).astype(np.float32)
    }

    grads = {
        'w1': np.random.randn(3, 3).astype(np.float32),
        'b1': np.random.randn(3).astype(np.float32),
        'w2': np.random.randn(3, 3).astype(np.float32),
        'b2': np.random.randn(3).astype(np.float32)
    }

    for name in params.keys():
        updated_param = optimizer.update(params[name], grads[name], name)
        print(f"Updated {name}: {updated_param}")
