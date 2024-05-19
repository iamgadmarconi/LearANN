import numpy as np


class Optimizer:
    def __init__(self, lr):
        self.learning_rate = lr

        self._initialized = False

    def update(self, param, grad_param, name):
        raise NotImplementedError

    def initialize_parameters(self, params):
        raise NotImplementedError





