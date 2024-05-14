import numpy as np


class Optimizer:
    def __init__(self, lr):
        self.learning_rate = lr


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def initialize_moments(self, shape, name):
        if name not in self.m:
            self.m[name] = np.zeros(shape)
            self.v[name] = np.zeros(shape)

    def update(self, param, grad_param, name):
        self.initialize_moments(param.shape, name)
        self.t += 1
        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad_param
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad_param ** 2)
        m_hat = self.m[name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[name] / (1 - self.beta2 ** self.t)
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return param