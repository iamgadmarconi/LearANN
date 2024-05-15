import numpy as np


class Optimizer:
    def __init__(self, lr):
        self.learning_rate = lr

        self._initialized = False

    def update(self, param, grad_param, name):
        raise NotImplementedError

    def initialize_parameters(self, params):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def initialize_parameters(self, params):
        if not self._initialized:
            self._initialized = True
            for name, shape in params.items():
                self.m[name] = np.zeros(shape)
                self.v[name] = np.zeros(shape)

    def update(self, param, grad_param, name):
        self.t += 1
        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad_param
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad_param ** 2)
        m_hat = self.m[name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[name] / (1 - self.beta2 ** self.t)
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param
    

class Adagrad(Optimizer):

    def __init__(self, lr, epsilon=1e-8):
        super().__init__(lr)
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

    def update(self, param, grad_param, name):
        param -= self.learning_rate * grad_param
        return param